"""Independent dense Gaussian references for retained-component mGRAD weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import _proposal_geometry
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import marginal_mixture_bank
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist

pytestmark = pytest.mark.inference(concern="sampling")


def test_mixture_bank_matches_dense_integral():
    covariance = jnp.array([[[0.2, 0.05], [0.05, 0.4]], [[1.1, -0.2], [-0.2, 0.8]]])
    variance = jnp.array([0.3, 0.7])
    auxiliary = jnp.array([-0.4, 0.6])
    mask = jnp.ones(2, dtype=bool)
    proposal = jax.vmap(_proposal_geometry, in_axes=(0, None, None, None, None))(
        covariance,
        GaussianTwist(jnp.diag(1 / variance), auxiliary / variance),
        mask,
        jnp.zeros((2, 2)),
        jnp.zeros(2),
    )
    means = jnp.array(
        [
            [[0.1, -0.2], [0.5, 0.3]],
            [[0.5, 0.1], [-0.4, 0.3]],
            [[0.3, -0.2], [0.0, 0.1]],
            [[0.5, 0.2], [0.4, -0.7]],
        ]
    )
    probabilities = jnp.log(jnp.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]]))
    prior = jnp.log(jnp.array([0.6, 0.4]))
    bank = marginal_mixture_bank(
        proposal,
        proposal.center(means, mask),
        probabilities,
        prior,
        jax.random.normal(jax.random.key(1), (3, 2)),
        jax.random.gumbel(jax.random.key(2), (4, 2)),
        jnp.array([0.2, -0.3]),
        mask,
    )
    gradients = jnp.array([[0.3, -0.1], [-0.4, 0.2], [0.1, 0.5], [-0.2, 0.4]])
    actual = np.asarray(bank.log_weights(gradients, variance, mask))
    factors = np.asarray(bank.geometry.chol)
    covariances = factors @ factors.swapaxes(-1, -2)
    maps = covariances / np.asarray(variance)[None, None, :]
    expected = []
    for index in range(4):
        others = np.arange(4) != index
        matrix = maps[others].reshape(-1, 2)
        joint_covariance = block_diag(*covariances[others]) + matrix @ np.diag(variance) @ matrix.T
        center = np.asarray(bank.particles[index] + variance * gradients[index])
        joint_mean = np.asarray(bank.prior_centers)[others].ravel() + matrix @ center
        expected.append(
            multivariate_normal.logpdf(
                np.asarray(bank.particles)[others].ravel(), joint_mean, joint_covariance
            )
            + float(bank.label_correction[index])
        )
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual - actual[0], expected - expected[0], atol=2e-5)
