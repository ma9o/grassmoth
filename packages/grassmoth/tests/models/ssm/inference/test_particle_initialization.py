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
    DsmcSpec,
    GuidedPathSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_test_support.compute import EXECUTION_CONTEXT, compile_program
from grassmoth_test_support.inference import (
    compile_fit_fixture,
    nonlinear_fit_fixture,
)
from grassmoth_test_support.interval_particle import gaussian_interval_case
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem

"""The exact centred initialization chart and its handoff to production sampling."""


pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("nonlinear", [False, True])
def test_initialization_chart_preserves_exact_density_and_directional_gradients(nonlinear):
    if nonlinear:
        prepared, panel = nonlinear_fit_fixture()
        target = build_particle_problem(
            prepared.prior_runtime_bundle,
            panel,
            trace_key=jax.random.key(7),
            reparam=None,
            compiler=compile_program,
        ).runtime
    else:
        target, _, _ = gaussian_interval_case()
    parameterization = build_joint_coordinates(target, compiler=compile_program)
    point = jnp.linspace(-0.4, 0.5, parameterization.dimension)

    def native_coordinates(q):
        parameters, path = parameterization.decode(q)
        return jnp.concatenate((parameters, path.ravel()[parameterization.free_indices]))

    jacobian = jax.jacfwd(native_coordinates)(point)
    _, log_jacobian = jnp.linalg.slogdet(jacobian)
    parameters, path = parameterization.decode(point)
    expected = target.log_posterior(parameters, path, target.observations, target.times)
    value, gradient = jax.jit(jax.value_and_grad(parameterization.log_density))(point)
    np.testing.assert_allclose(value, expected + log_jacobian, rtol=2e-6)
    direction = jax.random.normal(jax.random.key(321), point.shape)
    direction = direction / jnp.linalg.norm(direction)
    h = 1e-3
    finite_difference = (
        parameterization.log_density(point + h * direction)
        - parameterization.log_density(point - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=0.02, atol=2e-3)


def test_initialization_chart_excludes_fixed_states_and_replays_static_and_accumulated_states():
    model = _model()
    driver_spec = next(item for item in model.constructs if item.name == "driver")
    driver_spec = driver_spec.revised(
        indicators=tuple(
            indicator.revised(
                likelihood=indicator.likelihood.revised(
                    law=DeltaLawSpec(v=state(driver_spec.id)), standardized=False
                )
            )
            for indicator in driver_spec.indicators
        )
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
@pytest.mark.parametrize("leaf", ["amala", "amala_plus", "agrad", "twisted"])
def test_production_fit_consumes_gaussian_initialization(leaf):
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
                diagnostic_metrics=("particle_identity", "parameter_movement", "particle_weights")
                if leaf != "amala"
                else (),
                path_update=GuidedPathSpec(method="particle_agrad", num_particles=4)
                if leaf == "agrad"
                else (
                    GuidedPathSpec(method="twisted_particle_agrad", num_particles=4)
                    if leaf == "twisted"
                    else DsmcSpec(
                        num_particles=4,
                        method=(
                            "particle_rwm"
                            if (leaf) == "none"
                            else "particle_amala"
                            if (leaf) == "amala"
                            else "particle_amala_plus"
                        ),
                    )
                ),
            ),
        ),
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(result, MCMCPosterior)
    evidence = result.diagnostics.sampler_diagnostics
    assert isinstance(evidence, ParticleSamplerDiagnostics)
    assert result.diagnostics.phase_extra_fields is not None
    retained = result.diagnostics.phase_extra_fields["post_warmup"]
    if leaf == "amala_plus":
        assert evidence.path_weights is not None
        assert evidence.path_weights.kind == "dsmc"
        assert evidence.path_weights.phase == "post_warmup"
        assert evidence.path_weights.terminal.candidates == 4
        assert retained["particle_weight_ess"].shape == (2, 3, len(evidence.path_weights.tree) + 1)
    elif leaf in ("agrad", "twisted"):
        assert evidence.path_weights is not None
        assert evidence.path_weights.kind == "sequential"
        assert len(evidence.path_weights.filters) == len(panel.times)
        assert retained["particle_weight_ess"].shape == (2, 3, len(panel.times))
    else:
        assert evidence.path_weights is None
        assert not any(key.startswith("particle_weight_") for key in retained)
    assert isinstance(evidence.initialization, GaussianInitializationDiagnostics)
    assert evidence.initialization.scale == 0.1
    assert evidence.parameter_preconditioned
    assert result.draws.latent_paths is not None
    assert result.draws.latent_paths.shape == (6, len(panel.times), 1)
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
