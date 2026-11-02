from dataclasses import replace
from unittest.mock import Mock

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanSpec,
    FixedLatentStepSpec,
    GuidedPathSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
from grassmoth.models.ssm.compile.state_space import (
    AdditiveRecurrences,
    SequentialRecurrences,
    StateSpace,
)
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
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.interval_particle import interval_target

"""Particle configuration and initialization contracts without compiling a sampler."""


pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "step", [{"kind": "fixed", "delta": 0.1}, {"kind": "adapted", "target": 0.5}]
)
def test_auxiliary_kalman_request_roundtrip_and_identity(step):
    sampler = SamplerSpec.model_validate(
        {"path_update": {"method": "auxiliary_kalman", "latent_step": step}}
    )
    assert isinstance(sampler.path_update, AuxiliaryKalmanSpec)
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert sampler.kernel_identity != SamplerSpec().kernel_identity
    with pytest.raises(ValidationError, match="num_particles"):
        AuxiliaryKalmanSpec.model_validate({"method": "auxiliary_kalman", "num_particles": 256})


@pytest.mark.parametrize(
    ("support", "recurrences", "metrics", "block_length", "accepted"),
    [
        ("point", (), (), 300, True),
        ("interval", (), (), 300, False),
        ("point", (AdditiveRecurrences((0,)),), (), 300, False),
        ("interval", (), (), None, True),
        ("point", (AdditiveRecurrences((0,)),), (), None, True),
        ("point", (), ("particle_weights",), None, False),
        ("point", (), ("particle_identity",), None, False),
    ],
)
def test_auxiliary_kalman_admission(
    monkeypatch, support, recurrences, metrics, block_length, accepted
):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(1, (), (0,), (0,), (), recurrences),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(
        path_update=AuxiliaryKalmanSpec(block_length=block_length), diagnostic_metrics=metrics
    )
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted


@pytest.mark.parametrize(
    "recurrences", [(), (AdditiveRecurrences((0,)),), (SequentialRecurrences((0,)),)]
)
def test_twisted_preflight_accepts_interval_readings_and_recurrences(monkeypatch, recurrences):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(1, (), (0,), () if recurrences else (0,), (), recurrences),
        observations=(Mock(support=Mock(support_kind="interval")),),
    )
    sampler = SamplerSpec(
        path_update=GuidedPathSpec(method="twisted_particle_agrad"),
        compiler_reparameterization=True,
    )
    assert validate_sampler_support(compiled, sampler) is None


@pytest.mark.parametrize("method", ["particle_mgrad", "twisted_particle_agrad"])
def test_guided_configuration_roundtrip_and_continuation_identity(method):
    sampler = SamplerSpec(path_update=GuidedPathSpec(method=method, kappa=1))
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    # Particle-mGRAD with kappa=1 is the default path update.
    assert (sampler.kernel_identity == SamplerSpec().kernel_identity) == (
        method == "particle_mgrad"
    )
    assert (
        sampler.kernel_identity
        != SamplerSpec(path_update=GuidedPathSpec(method=method, kappa=0)).kernel_identity
    )


@pytest.mark.parametrize(
    ("support", "recurrences", "accepted"),
    [("point", (), True), ("interval", (), False), ("point", (AdditiveRecurrences((0,)),), True)],
)
def test_mgrad_admission_rejects_only_interval_readings(
    monkeypatch, support, recurrences, accepted
):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(1, (), (0,), () if recurrences else (0,), (), recurrences),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(path_update=GuidedPathSpec(method="particle_mgrad"))
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted
    if failure is not None:
        assert "Particle-mGRAD" in failure.message


@pytest.mark.parametrize(
    ("support", "recurrences", "accepted"),
    [("point", (), True), ("interval", (), False), ("point", (AdditiveRecurrences((0,)),), False)],
)
def test_scale_chart_admission(monkeypatch, support, recurrences, accepted):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(1, (), (0,), () if recurrences else (0,), (), recurrences),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(compiler_reparameterization=True, scale_chart=True)
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted
    if failure is not None:
        assert "scale chart" in failure.message


@pytest.mark.parametrize(
    ("method", "kappa", "support", "recurrences", "accepted"),
    [
        ("twisted_particle_agrad", 1, "point", (), True),
        ("twisted_particle_agrad", 0, "point", (), False),
        ("particle_mgrad", 1, "point", (), False),
        ("twisted_particle_agrad", 1, "interval", (), False),
        ("twisted_particle_agrad", 1, "point", (AdditiveRecurrences((0,)),), False),
    ],
)
def test_prefix_mixture_admission(monkeypatch, method, kappa, support, recurrences, accepted):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(1, (), (0,), () if recurrences else (0,), (), recurrences),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(
        path_update=GuidedPathSpec(method=method, kappa=kappa, prefix_mixture=True)
    )
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert isinstance(sampler.path_update, GuidedPathSpec)
    ordinary = sampler.revised(path_update=sampler.path_update.revised(prefix_mixture=False))
    assert sampler.kernel_identity != ordinary.kernel_identity
    assert GuidedPathSpec().prefix_mixture is False
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted
    if failure is not None:
        assert "Prefix-mixture" in failure.message


@pytest.mark.parametrize(
    "settings",
    [
        {"latent_step": {"kind": "fixed", "delta": 0}},
        {"latent_step": {"kind": "adapted", "target": 1}},
    ],
)
def test_path_update_boundary_owns_scale_policy(settings):
    with pytest.raises(ValidationError):
        GuidedPathSpec.model_validate(settings)


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
        {"path_update": {"method": "particle_mgrad", field: "bogus"}}
        if field == "drift"
        else {field: "bogus"}
    )
    with pytest.raises(ValidationError, match=field):
        SamplerSpec.model_validate(raw)


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
    grid = np.arange(3.0)
    context = interval_target(grid, np.zeros((3, 1)), grid[:, None], ("last",)).context
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
        context=context,
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
        initial_chart=None,
        initial_param_step_size=jnp.asarray([0.01]),
        initial_parameter_chol=jnp.eye(1),
        path_update=GuidedPathSpec(latent_step=FixedLatentStepSpec(delta=0.1)),
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
            sampler=SamplerSpec(num_warmup=0, num_samples_per_chain=1, num_chains=2),
            key=random.PRNGKey(0),
            initialization=Mock(
                spec=ParticleInitialization,
                positions=jnp.asarray(positions),
                paths=jnp.asarray(paths),
            ),
            execution_context=replace(EXECUTION_CONTEXT, compiler=compiler),
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
        sampler=SamplerSpec(),
        execution_context=EXECUTION_CONTEXT,
    )
    assert result is failure
    initialize.assert_called_once()
    sample.assert_not_called()


def test_computed_translation_is_the_only_translation_contract():
    ordinary = SamplerSpec()
    computed = ordinary.revised(compiler_reparameterization=True)
    assert ordinary.kernel_identity != computed.kernel_identity
    assert SamplerSpec.model_validate_json(computed.model_dump_json()) == computed
