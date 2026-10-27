"""Independently integrate the shared Gaussian auxiliary for a particle bank."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from grassmoth.models.ssm.inference.samplers.particle.marginal import marginal_geometry

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("kappa", [0, 1])
def test_marginal_bank_weights_match_dense_integrated_proposal(kappa):
    rng = np.random.default_rng(729)
    count, dimension = 5, 3
    prior_covariance = np.array([[0.7, 0.18, 0.0], [0.18, 0.4, 0.0], [0.0, 0.0, 1.0]])
    variance = np.array([0.13, 0.8, 0.4])
    mask = np.array([True, True, False])
    covariance = np.linalg.inv(np.linalg.inv(prior_covariance) + np.diag(mask / variance))
    gain = covariance @ np.diag(mask / variance)
    centers = rng.normal(size=(count, dimension)) * mask
    particles = rng.normal(size=(count, dimension)) * mask
    gradients = kappa * rng.normal(size=(count, dimension)) * mask
    factors = marginal_geometry(
        jnp.asarray(np.linalg.cholesky(covariance)),
        jnp.asarray(variance),
        jnp.asarray(mask),
        count,
    )
    actual = np.asarray(
        factors.log_weights(
            jnp.asarray(particles),
            jnp.asarray(centers),
            jnp.asarray(gradients),
            jnp.asarray(variance),
            jnp.asarray(mask),
        )
    )
    active_covariance = covariance[:2, :2]
    shared_covariance = (gain @ np.diag(variance) @ gain.T)[:2, :2]
    joint = np.kron(np.eye(count - 1), active_covariance) + np.kron(
        np.ones((count - 1, count - 1)), shared_covariance
    )
    expected = []
    for candidate in range(count):
        others = np.arange(count) != candidate
        auxiliary_mean = particles[candidate] + variance * gradients[candidate]
        means = centers[others] + gain @ auxiliary_mean
        expected.append(
            multivariate_normal.logpdf(
                particles[others, :2].ravel(), mean=means[:, :2].ravel(), cov=joint
            )
        )
    np.testing.assert_allclose(
        actual - logsumexp(actual), np.array(expected) - logsumexp(expected), atol=2e-5
    )
