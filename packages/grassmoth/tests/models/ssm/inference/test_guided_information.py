"""Independent exact Gaussian laws for auxiliary guidance, including zero information."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from grassmoth.models.ssm.inference.samplers.particle.guided import _guided_information
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist, future_information

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("kind", ["poisson", "nonconcave"])
def test_non_gaussian_guidance_matches_true_local_score_and_has_psd_precision(kind):
    auxiliary = jnp.array([0.2])
    variance = 0.08

    def log_likelihood(value):
        x = value[0]
        if kind == "poisson":
            return 3 * x - jnp.exp(x)
        return -0.5 * (1 - x**2) ** 2

    def observe(value, index):
        del index
        return jnp.zeros(2), jax.grad(log_likelihood)(value)

    information = _guided_information(
        observe, auxiliary[None], jnp.array([[variance]]), jnp.ones((1, 1), bool)
    )

    def exact_joint(value):
        auxiliary_mean = value + variance * jax.grad(log_likelihood)(value)
        return log_likelihood(value) - jnp.sum((auxiliary - auxiliary_mean) ** 2) / (2 * variance)

    expected = jax.grad(exact_joint)(auxiliary)
    actual = jax.grad(lambda value: information.log_density(value, 0))(auxiliary)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert np.linalg.eigvalsh(information.precision[0]).min() >= 0


@pytest.mark.parametrize("delta", [0.04, 0.08, 0.12, 0.16, [0.04, 0.32, 0.09]])
def test_guidance_matches_exact_correlated_gaussian_product_with_fixed_coordinate(delta):
    covariance = np.array([[0.04, 0.01, 0.004], [0.01, 0.06, -0.002], [0.004, -0.002, 0.05]])
    precision = np.linalg.inv(covariance)
    reading = np.array([0.6, -0.3, 0.1])
    auxiliary = np.array([0.15, -0.4, 0.2])
    mask = np.array([True, True, False])
    variance = np.broadcast_to(np.asarray(delta) / 2, auxiliary.shape)

    def observe(value, index):
        del index
        return jnp.zeros(2), -jnp.asarray(precision) @ (value - jnp.asarray(reading))

    product = jax.jit(
        lambda: _guided_information(
            observe,
            jnp.asarray(auxiliary[None]),
            jnp.asarray(variance[None]),
            jnp.asarray(mask[None]),
        )
    )()

    def exact(value):
        score = -precision @ (value - reading)
        result = norm.logpdf(
            auxiliary[mask], (value + variance * score)[mask], np.sqrt(variance[mask])
        ).sum()
        return result + multivariate_normal.logpdf(reading, value, covariance)

    baseline = np.array([0.0, 0.0, auxiliary[2]])
    for value in (np.array([0.3, -0.2, auxiliary[2]]), np.array([-0.5, 0.4, auxiliary[2]])):
        expected = exact(value) - exact(baseline)
        actual = product.log_density(jnp.asarray(value), 0) - product.log_density(
            jnp.asarray(baseline), 0
        )
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=2e-6)
        np.testing.assert_array_equal(product.precision[0, 2], np.zeros(3))
        assert product.shift[0, 2] == 0


@pytest.mark.parametrize("steps", [1, 4])
def test_zero_auxiliary_information_is_finite_and_readings_remain(steps):
    variance = 0.04
    reading = jnp.array([0.7])
    auxiliary = jnp.full((steps, 1), 0.2)

    def observe(value, index):
        del index
        return jnp.zeros(2), (reading - value) / variance

    product = _guided_information(
        observe, auxiliary, jnp.full((steps, 1), variance), jnp.ones((steps, 1), bool)
    )
    np.testing.assert_allclose(product.precision, np.full((steps, 1, 1), 1 / variance))
    np.testing.assert_allclose(product.shift, np.full((steps, 1), 0.7 / variance))
    matrices = jnp.full((steps - 1, 1, 1), 0.9)
    offsets = jnp.full((steps - 1, 1), 0.1)
    covariances = jnp.full((steps - 1, 1, 1), 0.2)
    twist = jax.jit(future_information)(
        matrices,
        offsets,
        covariances,
        GaussianTwist(jnp.zeros((steps, 1, 1)), jnp.zeros((steps, 1))),
    )
    np.testing.assert_array_equal(twist.precision, np.zeros((steps, 1, 1)))
    np.testing.assert_array_equal(twist.shift, np.zeros((steps, 1)))
