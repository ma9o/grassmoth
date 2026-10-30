"""Independent dense controls for positive path curvature and singular recurrences."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.inference.warmup.computed_chart import computed_translation_slopes
from grassmoth_test_support.interval_particle import gaussian_interval_case
from packages.grassmoth.tests.models.ssm.inference.test_guided_recurrences import WINDOWS, _target
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import _correlated_target
from packages.grassmoth.tests.models.ssm.inference.test_translation_chart import _gaussian_joint

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("nonlinear", [False, True])
def test_point_gauss_newton_slopes_match_dense_positive_curvature(nonlinear):
    base, _, _ = _gaussian_joint()
    path = jnp.array([[0.2], [-0.1], [0.4]])
    position = jnp.array([0.3])

    def mean(x):
        return 0.7 * x - nonlinear * 0.2 * jnp.tanh(x) ** 3

    def model(context):
        return eqx.tree_at(
            lambda value: value.state_evolution,
            base.dynamical_model(context),
            dsx.GaussianStateEvolution(lambda x, *_: mean(x), lambda *_: jnp.array([[0.3]])),
        )

    target = replace(base, dynamical_model=model)
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    residual_jacobian = np.eye(3)
    residual_jacobian[1, 0] = -float(jax.grad(lambda x: mean(x))(path[0, 0]))
    residual_jacobian[2, 1] = -float(jax.grad(lambda x: mean(x))(path[1, 0]))
    precision = (
        residual_jacobian.T @ np.diag([1.0, 1 / 0.3, 1 / 0.3]) @ residual_jacobian + np.eye(3) / 0.2
    )
    expected = -np.linalg.solve(precision, np.ones((3, 1)) / 0.2)
    np.testing.assert_allclose(actual[:, 0], expected, rtol=3e-6, atol=1e-6)
    assert np.linalg.eigvalsh(precision).min() > 0


def test_overlapping_intervals_match_dense_joint_gaussian_curvature():
    target, _, _ = gaussian_interval_case(steps=9)
    position = jnp.array([0.3])
    path = jnp.linspace(-0.2, 0.4, 9)[:, None]
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    precision = np.asarray(
        jax.hessian(
            lambda joint: (
                -target.log_posterior(joint[:1], joint[1:, None], target.observations, target.times)
            )
        )(jnp.concatenate((position, path[:, 0])))
    )
    expected = -np.linalg.solve(precision[1:, 1:], precision[1:, :1])
    np.testing.assert_allclose(actual[:, 0], expected, rtol=3e-5, atol=3e-6)
    assert np.linalg.norm(actual) > 0.1


def test_deterministic_stages_and_interval_readings_contribute_to_translation():
    target = _target(False)
    position = jnp.array([0.15])
    path = jnp.array([[0.2, 0.03, -0.015], [0.1, 0.1, 0.0], [-0.1, 0.12, 0.03], [0.15, 0.08, 0.05]])
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    matrix = np.array([[0.65, 0.0, 0.25], [0.4 + 0.2 * 0.15, 0.8, 0.0], [0.0, 0.3, 0.5]])
    noise_map = np.zeros((12, 4))
    prior_slope = np.zeros((4, 3))
    prior_slope[0] = [0.2, 0.2, -0.1]
    noise_map[:3, 0] = [np.sqrt(0.65), 0.0, 0.0]
    for t in range(1, 4):
        noise_map[3 * t : 3 * t + 3] = matrix @ noise_map[3 * t - 3 : 3 * t]
        noise_map[3 * t, t] = np.sqrt(0.3)
        prior_slope[t] = matrix @ prior_slope[t - 1] + [0.0, 0.2 * path[t - 1, 0], 0.0]
    covariance = noise_map @ noise_map.T
    rows = []
    for t in range(4):
        for channel in range(3):
            if not np.isfinite(target.observations[t, channel]):
                continue
            row = np.zeros((4, 3))
            if channel == 2:
                row[:, 2] = WINDOWS[t]
            else:
                row[t, 0 if channel == 0 else 2] = 1.0
            rows.append(row.ravel())
    design = np.stack(rows)
    expected = prior_slope.ravel() - covariance @ design.T @ np.linalg.solve(
        design @ covariance @ design.T + 0.45 * np.eye(len(rows)), design @ prior_slope.ravel()
    )
    np.testing.assert_allclose(actual[..., 0], expected.reshape(4, 3), rtol=3e-5, atol=3e-6)
    assert np.linalg.norm(actual[:, 1:]) > 0.1
    point_only = replace(
        target, intervals=None, observations=target.observations.at[:, 2].set(jnp.nan)
    )
    point_slopes = jax.jit(lambda: computed_translation_slopes(point_only, position, path))()
    assert np.linalg.norm(actual - point_slopes) > 0.001


def test_gaussian_covariance_only_parameter_has_zero_mean_cross_term_even_with_pins():
    target = _correlated_target()
    position = target.initial_position
    context = target.context(position, target.times)
    path = target.initial_path(context, exact_constraints=target.exact_constraints)
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    np.testing.assert_array_equal(actual, np.zeros_like(actual))
