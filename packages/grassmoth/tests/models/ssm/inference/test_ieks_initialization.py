"""Data-conditioned particle starts preserve exact support and Gaussian jitter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import GaussianInitializationSpec
from grassmoth.models.ssm.inference.joint_coordinates import (
    CentredParameterization,
    build_joint_coordinates,
)
from grassmoth.models.ssm.inference.warmup import initialization
from grassmoth.models.ssm.inference.warmup.ieks import ieks_path
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth_test_support.compute import EXECUTION_CONTEXT, compile_program
from grassmoth_test_support.interval_particle import interval_target
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import _correlated_target

pytestmark = pytest.mark.inference(concern="warmup")


def _ar_covariance(steps):
    innovations = np.eye(steps)
    for index in range(1, steps):
        innovations[index, :index] = 0.7 * innovations[index - 1, :index]
    return innovations @ np.diag([1.0, *([0.3] * (steps - 1))]) @ innovations.T


@pytest.fixture
def gaussian_case():
    readings = np.array([np.nan, np.nan, 2.0, 1.5, 1.0])
    times = np.arange(float(readings.size))
    target = interval_target(times, readings[:, None], times[:, None], ("last",))
    covariance = _ar_covariance(readings.size)
    observed = np.isfinite(readings)
    posterior_mean = covariance[:, observed] @ np.linalg.solve(
        covariance[np.ix_(observed, observed)] + 0.2 * np.eye(observed.sum()), readings[observed]
    )
    return target, posterior_mean[:, None]


def _seed(target):
    coordinates = build_joint_coordinates(target, compiler=compile_program)
    assert isinstance(coordinates.parameterization, CentredParameterization)
    initial_path = coordinates.parameterization.path_center
    context = target.context(target.initial_position, target.times)
    return jax.jit(lambda path: ieks_path(target, context, path))(initial_path), initial_path


def test_linear_gaussian_seed_matches_dense_smoother_including_unobserved_past(gaussian_case):
    target, expected = gaussian_case
    actual, initial = _seed(target)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(initial, np.zeros_like(initial))
    assert actual[0, 0] > 0.1


def test_correlated_gaussian_seed_conditions_exact_pins_and_omits_interval_readings():
    target = _correlated_target(pinned=0.3)
    actual, _ = _seed(target)
    steps, dimension = actual.shape
    transition = np.array([[0.8, 0.1], [0.0, 0.7]])
    scale = np.exp(-0.2)
    rho = 0.6 * np.tanh(0.3)
    initial_chol = np.array([[scale, 0.0], [rho * scale, 0.7]])
    step_chol = np.array([[0.4 * scale, 0.0], [0.3 * rho, 0.5 * scale]])
    innovations = np.zeros((steps * dimension, steps * dimension))
    for row in range(steps):
        for column in range(row + 1):
            factor = initial_chol if column == 0 else step_chol
            innovations[2 * row : 2 * row + 2, 2 * column : 2 * column + 2] = (
                np.linalg.matrix_power(transition, row - column) @ factor
            )
    covariance = innovations @ innovations.T
    prior_mean = np.concatenate(
        [
            np.linalg.matrix_power(transition, index) @ np.array([0.2, -0.3])
            for index in range(steps)
        ]
    )
    # Two point Gaussian readings and an exact stochastic pin; the interval is omitted.
    observed = np.array([1, 2, 5])
    values = np.array([0.1, 0.3, -0.2])
    expected = prior_mean + covariance[:, observed] @ np.linalg.solve(
        covariance[np.ix_(observed, observed)] + np.diag([0.2, 0.0, 0.2]),
        values - prior_mean[observed],
    )

    np.testing.assert_allclose(actual, expected.reshape(steps, dimension), rtol=2e-5, atol=2e-5)
    assert actual[1, 0] == jnp.asarray(0.3)


def test_poisson_seed_improves_exact_readings_and_reaches_a_stationary_path():
    counts = np.array([5.0, 8.0, 4.0, 6.0])
    times = np.arange(float(counts.size))
    target = interval_target(
        times,
        counts[:, None],
        times[:, None],
        ("last",),
        families=(DistributionFamily.POISSON,),
    )
    actual, initial = _seed(target)
    context = target.context(target.initial_position, target.times)
    initial_readings = target.observation_log_probs(context, initial, target.observations).sum()
    seeded_readings = target.observation_log_probs(context, actual, target.observations).sum()
    path = np.asarray(actual[:, 0])
    negative_log_density_gradient = np.linalg.solve(_ar_covariance(counts.size), path) + (
        np.exp(path) - counts
    )

    assert np.isfinite(actual).all()
    assert seeded_readings > initial_readings + 1.0
    np.testing.assert_allclose(negative_log_density_gradient, 0.0, atol=2e-3)


def test_particle_initializer_consumes_seed_preserving_parameter_draws_and_path_jitter(
    gaussian_case,
):
    target, expected_seed = gaussian_case
    coordinates = build_joint_coordinates(target, compiler=compile_program)
    spec = GaussianInitializationSpec()
    key = jax.random.key(71)
    points = spec.scale * jax.random.normal(key, (3, coordinates.dimension))
    original_parameters, original_paths = jax.vmap(coordinates.decode)(points)

    result = prepare_particle_initialization(
        target, spec=spec, key=key, num_chains=3, execution_context=EXECUTION_CONTEXT
    )

    assert isinstance(result, ParticleInitialization)
    np.testing.assert_array_equal(result.positions, original_parameters)
    np.testing.assert_allclose(
        result.paths, original_paths + expected_seed[None], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        result.parameter_chol, spec.scale * np.eye(target.initial_position.size)
    )
    np.testing.assert_allclose(result.pilot_mean, np.mean(result.paths, axis=0), atol=1e-6)
    np.testing.assert_allclose(result.pilot_variance, np.var(original_paths, axis=0), atol=1e-6)


def test_nonfinite_seed_returns_typed_initialization_failure(gaussian_case, monkeypatch):
    target, _ = gaussian_case
    monkeypatch.setattr(
        initialization,
        "ieks_path",
        lambda _target, _context, initial_path: jnp.full_like(initial_path, jnp.nan),
    )

    result = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(71),
        num_chains=2,
        execution_context=EXECUTION_CONTEXT,
    )

    assert isinstance(result, InitializationFailure)
    assert "non-finite path seed" in result.message


def test_seed_preserves_static_and_deterministic_accumulator_support():
    readings = jnp.broadcast_to(jnp.arange(4.0)[:, None], (4, 4))
    compiled, problem = _problem(_model(), readings)
    target = problem.runtime
    actual, initial = _seed(target)
    indices = {item.name: index for index, item in enumerate(compiled.states)}
    baseline, driver, total = (indices[name] for name in ("baseline", "driver", "total"))
    context = target.context(target.initial_position, target.times)

    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:, baseline], initial[:, baseline])
    np.testing.assert_allclose(actual[1:, total], np.cumsum(actual[:-1, driver]), atol=1e-7)
    assert target.path_log_prob(context, actual, target.observations) > target.path_log_prob(
        context, initial, target.observations
    )
