import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.reports.posterior_diagnostics import (
    GaussianInitializationDiagnostics,
    ParticleSamplerDiagnostics,
)
from grassmoth.artifacts.sampling import (
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    PathUpdateSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_test_support.compute import EXECUTION_CONTEXT, compile_program
from grassmoth_test_support.inference import (
    compile_fit_fixture,
    nonlinear_fit_fixture,
)
from grassmoth_test_support.values import revise_fixture
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem

"""The exact centred initialization chart and its handoff to production sampling."""


pytestmark = pytest.mark.inference(concern="warmup")


def test_initialization_chart_excludes_fixed_states_and_replays_static_and_accumulated_states():
    model = _model()
    driver_spec = next(item for item in model.constructs if item.name == "driver")
    driver_spec = revise_fixture(
        driver_spec,
        indicators=tuple(
            revise_fixture(
                indicator,
                likelihood=revise_fixture(
                    indicator.likelihood,
                    law=DeltaLawSpec(v=state(driver_spec.id)),
                    standardized=False,
                ),
            )
            for indicator in driver_spec.indicators
        ),
    )
    model = model.with_entities(edges=replace_constructs(model.edges, (driver_spec,)))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    column = next(
        index
        for index, law in enumerate(compiled.observations)
        if isinstance(law.law, DeltaLawSpec)
    )
    observed = jnp.full((4, len(compiled.observations)), jnp.nan).at[2, column].set(0.8)
    compiled, problem = _problem(model, observed)
    indices = {item.name: index for index, item in enumerate(compiled.states)}
    baseline, driver, total = (indices[name] for name in ("baseline", "driver", "total"))
    target = problem.runtime
    assert target.exact_constraints is not None
    values = target.exact_constraints.values
    parameterization = build_joint_coordinates(target, compiler=compile_program)
    _, path = jax.jit(parameterization.decode)(jnp.linspace(-0.5, 0.5, parameterization.dimension))
    assert parameterization.dimension == target.initial_position.size + int(
        jnp.sum(target.independent_mask & jnp.isnan(values))
    )
    np.testing.assert_array_equal(path[:, baseline], jnp.repeat(path[0, baseline], 4))
    np.testing.assert_allclose(path[1:, total], jnp.cumsum(path[:-1, driver]), atol=1e-7)
    assert path[2, driver] == values[2, driver]


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("method", "metrics"),
    [
        ("particle_mgrad", ("particle_identity", "parameter_movement", "particle_weights")),
        ("twisted_particle_agrad", ()),
    ],
)
def test_production_fit_consumes_gaussian_initialization(method, metrics):
    prepared, panel = nonlinear_fit_fixture()
    result = fit(
        prepared.prior_runtime_bundle,
        panel,
        sampler=SamplerSpec(
            num_warmup=2,
            num_samples_per_chain=3,
            num_chains=2,
            seed=7,
            algorithm=MarginalParticleGibbsSpec(
                diagnostic_metrics=metrics,
                path_update=PathUpdateSpec(method=GuidedMethodSpec(kind=method, num_particles=4)),
            ),
        ),
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(result, MCMCPosterior)
    evidence = result.diagnostics.sampler_diagnostics
    assert isinstance(evidence, ParticleSamplerDiagnostics)
    assert result.diagnostics.phase_extra_fields is not None
    retained = result.diagnostics.phase_extra_fields["post_warmup"]
    if metrics:
        assert evidence.path_weights is not None
        assert evidence.path_weights.phase == "post_warmup"
        assert len(evidence.path_weights.filters) == len(panel.times)
        assert evidence.path_weights.filters[0].candidates == 4
        assert retained["particle_weight_ess"].shape == (2, 3, len(panel.times))
    else:
        assert evidence.path_weights is None
        assert not any(key.startswith("particle_weight_") for key in retained)
    assert isinstance(evidence.initialization, GaussianInitializationDiagnostics)
    assert evidence.initialization.scale == 0.1
    assert evidence.parameter_preconditioned
    assert isinstance(result.draws.latent_paths, np.ndarray)
    assert result.draws.latent_paths.shape == (6, len(panel.times), 1)
    assert result.diagnostics.latent_chains is not None
    assert np.shares_memory(result.draws.latent_paths, result.diagnostics.latent_chains.paths)
    assert len(result.diagnostics.latent_chains.diagnostics) == 1
    assert isinstance(result.diagnostics.all_complete_log_posterior_history, np.ndarray)
    assert all(
        isinstance(value, np.ndarray)
        for value in result.diagnostics.phase_extra_fields["all"].values()
    )
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
