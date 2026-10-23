import logging
import time
from datetime import UTC, datetime
from unittest.mock import Mock

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    DsmcSpec,
    FixedLatentStepSpec,
    GaussianReferenceTemperingSpec,
    MarginalParticleGibbsSpec,
    ParticleAGradSpec,
    SamplerSpec,
    TwistedParticleAGradSpec,
)
from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
from grassmoth.models.ssm.compile.state_space import AdditiveRecurrences, StateSpace
from grassmoth.models.ssm.inference.samplers.mpgibbs import runner
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    MarginalParticleGibbsKernel,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
from grassmoth.models.ssm.preparation.binding import BoundPanel
from grassmoth.models.ssm.preparation.preflight import validate_sampler_support
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program

"""Particle configuration and initialization contracts without compiling a sampler."""


pytestmark = pytest.mark.contract


@pytest.mark.parametrize("spec_type", [ParticleAGradSpec, TwistedParticleAGradSpec])
@pytest.mark.parametrize("unsupported", ["interval", "replay", "tempering"])
def test_guided_preflight_rejects_unsupported_models_before_compute(
    monkeypatch, spec_type, unsupported
):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(
            2,
            (),
            (0, 1),
            (0,) if unsupported == "replay" else (0, 1),
            (),
            (AdditiveRecurrences((1,)),) if unsupported == "replay" else (),
        ),
        observations=(
            Mock(support=Mock(support_kind="interval" if unsupported == "interval" else "point")),
        ),
    )
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=spec_type(),
            tempering=GaussianReferenceTemperingSpec() if unsupported == "tempering" else None,
        )
    )
    failure = validate_sampler_support(compiled, sampler)
    assert failure is not None
    assert unsupported in failure.message


@pytest.mark.parametrize("spec_type", [ParticleAGradSpec, TwistedParticleAGradSpec])
def test_guided_configuration_roundtrip_and_continuation_identity(spec_type):
    sampler = SamplerSpec(algorithm=MarginalParticleGibbsSpec(path_update=spec_type(kappa=1)))
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert sampler.kernel_identity != SamplerSpec().kernel_identity
    assert (
        sampler.kernel_identity
        != SamplerSpec(
            algorithm=MarginalParticleGibbsSpec(path_update=spec_type(kappa=0))
        ).kernel_identity
    )
    with pytest.raises(ValidationError, match="latent_block_coords"):
        spec_type.model_validate({"latent_block_coords": 1})


@pytest.mark.parametrize(
    "settings",
    [
        {"drift": "global"},
        {"amala_kappa": 0.75},
        {"amala_grad_clip": 2},
        {"latent_block_coords": 3},
        {"latent_step": {"kind": "fixed", "delta": 0}},
        {"latent_step": {"kind": "adapted", "target": 1}},
    ],
)
def test_dsmc_boundary_owns_discrete_choices_and_scale_policy(settings):
    with pytest.raises(ValidationError):
        DsmcSpec.model_validate(settings)


@pytest.mark.parametrize(
    "field",
    [
        "parameter_proposal",
        "latent_smoother",
        "drift",
        "adaptation_scheme",
        "init_method",
        "latent_init_method",
        "auto_preconditioner_method",
    ],
)
def test_sampler_boundary_rejects_unknown_modes(field):
    raw = (
        {"path_update": {"method": "particle_rwm", field: "bogus"}}
        if field == "drift"
        else {field: "bogus"}
    )
    with pytest.raises(ValidationError, match=field):
        MarginalParticleGibbsSpec.model_validate(raw)


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("position_shape", "initialization.positions must have shape"),
        ("path_shape", "leading dimension num_chains"),
        ("position_finite", r"non-finite for chain\(s\) \[1\]"),
        ("path_finite", r"non-finite for chain\(s\) \[1\]"),
    ],
)
def test_runner_rejects_invalid_initial_chains_before_any_step(monkeypatch, invalid, error):
    positions = np.zeros((2, 1), dtype=np.float32)
    paths = np.zeros((2, 3, 1), dtype=np.float32)
    if invalid == "position_shape":
        positions = positions[:1]
    elif invalid == "path_shape":
        paths = paths[:1]
    elif invalid == "position_finite":
        positions[1, 0] = np.inf
    else:
        paths[1, 1, 0] = np.nan

    target = Mock(
        spec=ParticleTarget,
        observations=jnp.zeros((3, 1)),
        times=jnp.arange(3.0),
        initial_position=jnp.zeros(1),
        independent_mask=jnp.ones((3, 1), dtype=bool),
        context=Mock(return_value=jnp.array(0.0)),
        replay_path=lambda _context, path: path,
        initial_path=Mock(side_effect=AssertionError("Supplied paths must not be regenerated")),
        initial_moments=Mock(return_value=(jnp.zeros(1), jnp.eye(1))),
        log_posterior_from_context=lambda z, _context, path, _obs: (
            jnp.sum(z) + jnp.sum(path),
            jnp.sum(path),
        ),
    )
    kernel = Mock(
        spec=MarginalParticleGibbsKernel,
        parameter_gradient_fn=jnp.zeros_like,
        target_accept=jnp.asarray([0.35]),
        exact_constraints=None,
        initial_param_step_size=jnp.asarray([0.01]),
        initial_parameter_chol=jnp.eye(1),
        path_update=DsmcSpec(latent_step=FixedLatentStepSpec(delta=0.1)),
        min_scale=jnp.asarray([1e-6]),
        max_scale=jnp.asarray([1.0]),
    )
    step = Mock(side_effect=AssertionError("Invalid initial states must fail before sampling"))
    monkeypatch.setattr(runner, "compile_projected_sweeps", step)

    # Execute the synthetic initializer directly so mocks stay outside JAX tracing.
    def compiler(function, *_args, **_kwargs):
        return Mock(call=function)

    with pytest.raises(ValueError, match=error):
        runner.run_marginal_particle_gibbs(
            target,
            kernel=kernel,
            compute_budget=COMPUTE_BUDGET,
            num_warmup=0,
            num_samples=1,
            num_chains=2,
            key=random.PRNGKey(0),
            adaptation_rate=0.1,
            initialization=Mock(
                spec=ParticleInitialization,
                positions=jnp.asarray(positions),
                paths=jnp.asarray(paths),
            ),
            clock=time.monotonic,
            compiler=compiler,
            prepare_program=prepare_program,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
        )
    step.assert_not_called()
    target.initial_path.assert_not_called()


def test_initialization_failure_stops_production_before_sampling(monkeypatch):
    from importlib import import_module

    fitting = import_module("grassmoth.models.ssm.inference.samplers.mpgibbs.fit")
    target = Mock(spec=ParticleTarget, initial_position=jnp.zeros(1))
    monkeypatch.setattr(
        fitting, "build_particle_problem", Mock(return_value=Mock(runtime=target, public_sites=()))
    )
    failure = InitializationFailure.unavailable("No finite Gaussian starts")
    initialize = Mock(return_value=failure)
    sample = Mock(side_effect=AssertionError("Failed initialization must stop the fit"))
    monkeypatch.setattr(fitting, "prepare_particle_initialization", initialize)
    monkeypatch.setattr(fitting, "run_marginal_particle_gibbs", sample)
    result = fitting.fit_marginal_particle_gibbs(
        Mock(spec=PriorRuntimeBundle),
        Mock(
            spec=BoundPanel,
            observations=jnp.zeros((3, 1)),
            compiled_dynamical_model=Mock(
                spec=CompiledDynamicalModel,
                state_space=StateSpace(
                    size=1,
                    inputs=(),
                    initial=(0,),
                    transition=(0,),
                    static=(),
                    recurrence=(),
                ),
            ),
        ),
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_program,
        prepare_program=prepare_program,
        sampler=SamplerSpec(),
        clock=time.monotonic,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert result is failure
    initialize.assert_called_once()
    sample.assert_not_called()
