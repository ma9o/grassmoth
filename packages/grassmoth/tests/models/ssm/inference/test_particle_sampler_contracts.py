from dataclasses import replace
from unittest.mock import Mock

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    ConditionalParameterProposalSpec,
    FixedLatentStepSpec,
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    ParameterMeanProposalSpec,
    ParameterMixtureProposalSpec,
    PathUpdateSpec,
    SamplerSpec,
    WholePathUpdateSpec,
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
from grassmoth_test_support.values import revise_fixture

"""Particle configuration and initialization contracts without compiling a sampler."""


pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "step", [{"kind": "fixed", "delta": 0.1}, {"kind": "adapted", "target": 0.5}]
)
def test_auxiliary_kalman_request_roundtrip_and_identity(step):
    sampler = SamplerSpec.model_validate(
        {
            "algorithm": {
                "method": "mpgibbs",
                "path_update": {
                    "method": {"kind": "auxiliary_kalman"},
                    "update": {"kind": "blocked", "block_length": 300},
                    "proposal": {"kind": "conditional_parameter"},
                    "latent_step": step,
                },
            }
        }
    )
    assert isinstance(sampler.algorithm, MarginalParticleGibbsSpec)
    assert isinstance(sampler.algorithm.path_update.method, AuxiliaryKalmanMethodSpec)
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert sampler.kernel_identity != SamplerSpec().kernel_identity
    with pytest.raises(ValidationError, match="num_particles"):
        AuxiliaryKalmanMethodSpec.model_validate({"kind": "auxiliary_kalman", "num_particles": 256})
    with pytest.raises(ValidationError, match="power of two"):
        GuidedMethodSpec(kind="particle_mgrad", num_particles=24)
    with pytest.raises(ValidationError, match="power of two"):
        MarginalParticleGibbsSpec(n_parameter_particles=3)


@pytest.mark.parametrize(
    ("support", "recurrences", "metrics", "block_length", "accepted"),
    [
        ("point", (), (), 300, True),
        ("interval", (), (), 300, True),
        ("point", (AdditiveRecurrences((0,)),), (), 300, True),
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
        state_space=StateSpace(
            1,
            (),
            (0,),
            (0,),
            (),
            recurrences,
            initial_covariance="dense",
            transition_covariance="dense",
        ),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(
                method=AuxiliaryKalmanMethodSpec(),
                update=WholePathUpdateSpec()
                if (block_length) is None
                else BlockedPathUpdateSpec(block_length=block_length),
                proposal=ParameterMeanProposalSpec()
                if (block_length) is None
                else ConditionalParameterProposalSpec(),
            ),
            diagnostic_metrics=metrics,
        )
    )
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted


@pytest.mark.parametrize(
    "recurrences", [(), (AdditiveRecurrences((0,)),), (SequentialRecurrences((0,)),)]
)
def test_guided_preflight_accepts_interval_readings_and_recurrences(monkeypatch, recurrences):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(
            1,
            (),
            (0,),
            () if recurrences else (0,),
            (),
            recurrences,
            initial_covariance="dense",
            transition_covariance="dense",
        ),
        observations=(Mock(support=Mock(support_kind="interval")),),
    )
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(
                method=GuidedMethodSpec(kind="twisted_particle_agrad"),
                proposal=ParameterMeanProposalSpec(),
            ),
        )
    )
    assert validate_sampler_support(compiled, sampler) is None


@pytest.mark.parametrize(
    ("particles", "labels", "accepted"),
    [
        (64, 64, True),
        (128, 64, True),
        (256, 32, True),
        (512, 16, True),
        (1024, 4, True),
        (2048, 2, True),
        (64, 128, False),
        (256, 64, False),
        (512, 32, False),
        (1024, 8, False),
        (2048, 4, False),
        (4096, 2, False),
    ],
)
def test_mgrad_rejects_particle_banks_beyond_its_fused_tail(particles, labels, accepted):
    def sampler(kind):
        return SamplerSpec(
            algorithm=MarginalParticleGibbsSpec(
                n_parameter_particles=labels,
                path_update=PathUpdateSpec(
                    method=GuidedMethodSpec(kind=kind, num_particles=particles)
                ),
            )
        )

    assert (sampler("particle_mgrad").support_failure is None) == accepted
    assert sampler("twisted_particle_agrad").support_failure is None


@pytest.mark.parametrize(("particles", "accepted"), [(4096, True), (8192, False)])
def test_mgrad_smoother_counts_one_label_under_nuts(particles, accepted):
    sampler = SamplerSpec(
        algorithm=NutsDASpec(
            smoother=PathUpdateSpec(method=GuidedMethodSpec(num_particles=particles))
        )
    )
    assert (sampler.support_failure is None) == accepted


@pytest.mark.parametrize(
    ("size", "method", "accepted"),
    [
        (16, "particle_mgrad", True),
        (17, "particle_mgrad", False),
        (17, "twisted_particle_agrad", True),
    ],
)
def test_mgrad_rejects_models_beyond_its_fused_tail(monkeypatch, size, method, accepted):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    coordinates = tuple(range(size))
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(size, (), coordinates, coordinates, (), (), "diagonal", "diagonal"),
        observations=(Mock(support=Mock(support_kind="point")),),
    )
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(method=GuidedMethodSpec(kind=method))
        )
    )
    assert (validate_sampler_support(compiled, sampler) is None) == accepted


@pytest.mark.parametrize("method", ["particle_mgrad", "twisted_particle_agrad"])
def test_guided_configuration_roundtrip_and_continuation_identity(method):
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(method=GuidedMethodSpec(kind=method, kappa=1))
        )
    )
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    # Particle-mGRAD with kappa=1 is the default path update.
    assert (sampler.kernel_identity == SamplerSpec().kernel_identity) == (
        method == "particle_mgrad"
    )
    assert (
        sampler.kernel_identity
        != SamplerSpec(
            algorithm=MarginalParticleGibbsSpec(
                path_update=PathUpdateSpec(method=GuidedMethodSpec(kind=method, kappa=0))
            )
        ).kernel_identity
    )


@pytest.mark.parametrize(
    ("method", "kappa", "support", "recurrences", "accepted"),
    [
        ("twisted_particle_agrad", 1, "point", (), True),
        ("twisted_particle_agrad", 0, "point", (), False),
        ("particle_mgrad", 1, "point", (), True),
        ("particle_mgrad", 0, "point", (), True),
        ("particle_mgrad", 1, "interval", (), False),
        ("particle_mgrad", 1, "point", (AdditiveRecurrences((0,)),), False),
        ("twisted_particle_agrad", 1, "interval", (), False),
        ("twisted_particle_agrad", 1, "point", (AdditiveRecurrences((0,)),), False),
    ],
)
def test_prefix_mixture_admission(monkeypatch, method, kappa, support, recurrences, accepted):
    from grassmoth.models.ssm import compiled_access

    monkeypatch.setattr(compiled_access, "diffusion_families", lambda _compiled: ())
    compiled = Mock(
        spec=CompiledDynamicalModel,
        state_space=StateSpace(
            1,
            (),
            (0,),
            () if recurrences else (0,),
            (),
            recurrences,
            initial_covariance="dense",
            transition_covariance="dense",
        ),
        observations=(Mock(support=Mock(support_kind=support)),),
    )
    sampler = SamplerSpec(
        algorithm=MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(
                method=GuidedMethodSpec(kind=method, kappa=kappa),
                proposal=ParameterMixtureProposalSpec(),
            )
        )
    )
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert isinstance(sampler.algorithm, MarginalParticleGibbsSpec)
    assert isinstance(sampler.algorithm.path_update.method, GuidedMethodSpec)
    ordinary = revise_fixture(
        sampler,
        algorithm=revise_fixture(
            sampler.algorithm,
            path_update=revise_fixture(
                sampler.algorithm.path_update, proposal=ParameterMeanProposalSpec()
            ),
        ),
    )
    assert sampler.kernel_identity != ordinary.kernel_identity
    assert isinstance(PathUpdateSpec().proposal, ParameterMixtureProposalSpec)
    failure = validate_sampler_support(compiled, sampler)
    assert (failure is None) == accepted
    if failure is not None:
        assert "mixture" in failure.message


@pytest.mark.parametrize("method", ["particle_mgrad", "twisted_particle_agrad", "auxiliary_kalman"])
def test_path_proposal_default_depends_on_method_and_preserves_explicit_choice(method):
    path = PathUpdateSpec.model_validate({"method": {"kind": method}})
    assert path.proposal == (
        ParameterMixtureProposalSpec()
        if method == "particle_mgrad"
        else ParameterMeanProposalSpec()
    )
    assert path.support_failure is None
    explicit = PathUpdateSpec.model_validate(
        {"method": {"kind": method}, "proposal": {"kind": "parameter_mean"}}
    )
    assert explicit.proposal == ParameterMeanProposalSpec()
    assert PathUpdateSpec.model_validate_json(path.model_dump_json()) == path


@pytest.mark.parametrize(
    "settings",
    [
        {"latent_step": {"kind": "fixed", "delta": 0}},
        {"latent_step": {"kind": "adapted", "target": 1}},
    ],
)
def test_path_update_boundary_owns_scale_policy(settings):
    with pytest.raises(ValidationError):
        PathUpdateSpec.model_validate(settings)


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
        {
            "path_update": {
                "method": {"kind": "particle_mgrad"},
                "update": {"kind": "whole_path"},
                "proposal": {"kind": "parameter_mean"},
                field: "bogus",
            }
        }
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
        initial_param_step_size=jnp.asarray([0.01]),
        initial_parameter_chol=jnp.eye(1),
        path_update=PathUpdateSpec(latent_step=FixedLatentStepSpec(delta=0.1)),
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
                    initial_covariance="dense",
                    transition_covariance="dense",
                ),
            ),
        ),
        sampler=SamplerSpec(),
        execution_context=EXECUTION_CONTEXT,
    )
    assert result is failure
    initialize.assert_called_once()
    sample.assert_not_called()


@pytest.mark.parametrize("engine", ["mpgibbs", "nuts_da"])
@pytest.mark.parametrize(
    ("method", "scope", "proposal", "weights", "supported"),
    [
        ("particle_mgrad", "whole_path", "parameter_mean", None, False),
        ("twisted_particle_agrad", "whole_path", "parameter_mean", None, True),
        ("auxiliary_kalman", "whole_path", "parameter_mean", None, True),
        ("twisted_particle_agrad", "whole_path", "parameter_mixture", "running_prefix", True),
        ("auxiliary_kalman", "blocked", "conditional_parameter", None, True),
        ("particle_mgrad", "whole_path", "parameter_mixture", "running_prefix", True),
        ("particle_mgrad", "whole_path", "parameter_mixture", "prior", False),
        ("twisted_particle_agrad", "whole_path", "parameter_mixture", "prior", False),
        ("auxiliary_kalman", "whole_path", "parameter_mixture", "prior", False),
        ("auxiliary_kalman", "whole_path", "parameter_mixture", "running_prefix", False),
        ("particle_mgrad", "blocked", "parameter_mean", None, False),
        ("twisted_particle_agrad", "blocked", "conditional_parameter", None, False),
        ("auxiliary_kalman", "blocked", "parameter_mean", None, False),
        ("auxiliary_kalman", "blocked", "parameter_mixture", "prior", False),
        ("auxiliary_kalman", "whole_path", "conditional_parameter", None, False),
    ],
)
def test_common_path_schema_preserves_requests_and_reports_support(
    engine, method, scope, proposal, weights, supported
):
    path = {
        "method": {"kind": method},
        "update": {"kind": scope, **({"block_length": 17} if scope == "blocked" else {})},
        "proposal": {"kind": proposal, **({"weights": weights} if weights else {})},
        "latent_step": {"kind": "fixed", "delta": 0.3},
    }
    sampler = SamplerSpec.model_validate(
        {
            "algorithm": {
                "method": engine,
                "smoother" if engine == "nuts_da" else "path_update": path,
            }
        }
    )
    assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
    assert (sampler.support_failure is None) == supported
    if not supported:
        assert sampler.support_failure is not None
        assert sampler.support_failure.message


def test_continuation_identity_distinguishes_update_and_proposal_laws():
    base = PathUpdateSpec(method=AuxiliaryKalmanMethodSpec())
    blocked = revise_fixture(
        base,
        update=BlockedPathUpdateSpec(block_length=17),
        proposal=ConditionalParameterProposalSpec(),
    )
    mixture = revise_fixture(base, proposal=ParameterMixtureProposalSpec(weights="prior"))
    variants = (
        base,
        blocked,
        revise_fixture(blocked, update=BlockedPathUpdateSpec(block_length=23)),
        mixture,
        revise_fixture(mixture, proposal=ParameterMixtureProposalSpec(weights="running_prefix")),
    )
    assert len({variant.kernel_identity for variant in variants}) == len(variants)
