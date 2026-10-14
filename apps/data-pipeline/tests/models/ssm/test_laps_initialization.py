"""Exact joint LAPS initialization and its paired handoff to production sampling."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import state
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior, fit
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.inference.warmup import laps_ensemble
from nof1_causal_lab.models.ssm.inference.warmup.laps import (
    LapsInitializationResult,
    build_laps_coordinates,
    prepare_laps_initialization,
)
from nof1_causal_lab.models.ssm.inference.warmup.laps_ensemble import run_laps_ensemble
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.sampler_config import (
    LapsInitializationSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.inference_fixtures import bind_panel_fixture, compile_fit_fixture
from tests.interval_particle_fixtures import gaussian_interval_case
from tests.model_fixtures import load_model_fixture
from tests.models.ssm.test_exact_state_space import _model, _problem

pytestmark = pytest.mark.inference(concern="warmup")


def _nonlinear_fixture():
    model = load_model_fixture(
        "ecosystem_consumer/nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model.json"
    )
    prepared = compile_fit_fixture(model)
    panel = bind_panel_fixture(
        prepared.compiled_dynamical_model,
        jnp.array([[0.2, 1.0], [jnp.nan, 2.0], [0.3, jnp.nan], [-0.2, 0.0]]),
        jnp.array([0.0, 0.05, 0.17, 0.9]),
    )
    return prepared, panel


@pytest.mark.parametrize("nonlinear", [False, True])
def test_laps_chart_preserves_exact_density_and_directional_gradients(nonlinear):
    if nonlinear:
        prepared, panel = _nonlinear_fixture()
        target = build_particle_problem(
            prepared.prior_runtime_bundle,
            panel,
            scheme="euler_maruyama",
            trace_key=jax.random.key(7),
            reparam=None,
        ).runtime
    else:
        target, _, _ = gaussian_interval_case()
    chart = build_laps_coordinates(target)
    point = jnp.linspace(-0.4, 0.5, chart.sampling_dimension)

    def native_coordinates(q):
        parameters, path = chart.decode(q)
        return jnp.concatenate((parameters, path.ravel()[chart.free_indices]))

    jacobian = jax.jacfwd(native_coordinates)(point)
    _, log_jacobian = jnp.linalg.slogdet(jacobian)
    parameters, path = chart.decode(point)
    expected = target.log_posterior(parameters, path, target.observations, target.times)
    value, gradient = jax.jit(jax.value_and_grad(chart.log_density))(point)
    np.testing.assert_allclose(value, expected + log_jacobian, rtol=2e-6)
    direction = jax.random.normal(jax.random.key(321), point.shape)
    direction = direction / jnp.linalg.norm(direction)
    h = 1e-3
    finite_difference = (
        chart.log_density(point + h * direction) - chart.log_density(point - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=0.02, atol=2e-3)


def test_laps_chart_excludes_fixed_states_and_replays_static_and_accumulated_states():
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
    chart = build_laps_coordinates(target)
    _, path = jax.jit(chart.decode)(jnp.linspace(-0.5, 0.5, chart.sampling_dimension))
    assert chart.dimension == target.initial_position.size + int(
        jnp.sum(target.independent_mask & jnp.isnan(values))
    )
    np.testing.assert_array_equal(path[:, baseline], jnp.repeat(path[0, baseline], 4))
    np.testing.assert_allclose(path[1:, total], jnp.cumsum(path[:-1, driver]), atol=1e-7)
    assert path[2, driver] == values[2, driver]


def test_laps_ensemble_recovers_analytic_joint_moments_with_paired_paths():
    target, mean, covariance = gaussian_interval_case()
    result = prepare_laps_initialization(
        target,
        spec=LapsInitializationSpec(
            ensemble_size=1024, num_unadjusted_steps=2000, num_adjusted_steps=80
        ),
        key=jax.random.key(312),
        num_chains=1024,
        compute_budget=COMPUTE_BUDGET,
    )
    assert isinstance(result, LapsInitializationResult)
    joint = np.concatenate((result.positions, result.paths[:, :, 0]), axis=1)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), (joint - mean).T).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=0.15)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(mean.size), atol=0.2)
    assert result.diagnostics.adjusted_acceptance > 0.1
    assert np.linalg.eigvalsh(np.asarray(result.parameter_chol @ result.parameter_chol.T)).min() > 0


def test_nonfinite_laps_initialization_returns_a_failure_without_substitution():
    result = run_laps_ensemble(
        lambda point: jnp.asarray(jnp.nan) + jnp.sum(point),
        dimension=2,
        scientific_dimension=2,
        key=jax.random.key(41),
        spec=LapsInitializationSpec(ensemble_size=8, num_unadjusted_steps=10, num_adjusted_steps=1),
        compute_budget=COMPUTE_BUDGET,
    )
    assert isinstance(result, InitializationFailure)
    assert "non-finite" in result.message


def test_laps_batch_width_preserves_members_and_global_adaptation(monkeypatch):
    execute = laps_ensemble._execute
    spec = LapsInitializationSpec(
        ensemble_size=17, num_unadjusted_steps=20, num_adjusted_steps=3, steps_per_sample=2
    )

    def run():
        return run_laps_ensemble(
            lambda point: -0.5 * jnp.sum((point - jnp.array([0.3, -0.7, 1.2])) ** 2),
            dimension=3,
            scientific_dimension=3,
            key=jax.random.key(718),
            spec=spec,
            compute_budget=COMPUTE_BUDGET,
        )

    full = run()

    def bounded(function, initial, budget, *, max_batch_size):
        return execute(function, initial, budget, max_batch_size=min(4, max_batch_size))

    monkeypatch.setattr(laps_ensemble, "_execute", bounded)
    batched = run()
    assert isinstance(full, laps_ensemble.LapsEnsemble)
    assert isinstance(batched, laps_ensemble.LapsEnsemble)
    assert batched.positions.shape == (17, 3)
    np.testing.assert_allclose(batched.positions, full.positions, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        batched.diagnostics.adjusted_acceptance,
        full.diagnostics.adjusted_acceptance,
        rtol=2e-4,
    )
    assert batched.diagnostics.unadjusted_steps == full.diagnostics.unadjusted_steps


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("leaf", ["amala_exact", "paid_mix"])
def test_production_fit_consumes_laps_package(leaf):
    prepared, panel = _nonlinear_fixture()
    result = fit(
        prepared.prior_runtime_bundle,
        panel,
        compute_budget=COMPUTE_BUDGET,
        sampler=SamplerSpec(
            num_warmup=2,
            num_samples_per_chain=3,
            num_chains=2,
            num_particles=4,
            seed=7,
            marginal_particle_gibbs=MarginalParticleGibbsSpec(
                dsmc_leaf_proposal=leaf,
                laps=LapsInitializationSpec(
                    ensemble_size=128, num_unadjusted_steps=1000, num_adjusted_steps=20
                ),
            ),
        ),
        reparam=None,
        clock=time.monotonic,
    )
    assert isinstance(result, ParticleMCMCPosterior)
    evidence = result.diagnostics.marginal_particle_gibbs
    assert evidence is not None
    assert evidence.initialization.ensemble_size == 128
    assert evidence.parameter_preconditioned
    assert result.draws.latent_paths is not None
    assert result.draws.latent_paths.shape == (6, len(panel.times), 1)
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
