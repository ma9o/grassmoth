"""The prefix proposal samples components but corrects with the complete proposal mixture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import _proposal_geometry
from grassmoth.models.ssm.inference.samplers.particle.prefix_mixture import _mixture_particles
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist


@pytest.mark.inference(concern="sampling")
def test_sampled_components_pay_normalized_mixture_including_reference():
    covariances = np.array([[[0.7, 0.12], [0.12, 1.3]], [[1.8, -0.2], [-0.2, 0.5]]])
    precision = np.diag([0.4, 0.8])
    expected_covariances = np.array(
        [np.linalg.inv(np.linalg.inv(q) + precision) for q in covariances]
    )
    proposal = jax.vmap(
        lambda covariance: _proposal_geometry(
            covariance,
            GaussianTwist(jnp.asarray(precision, jnp.float32), jnp.zeros(2)),
            jnp.ones(2, dtype=bool),
            jnp.zeros((2, 2)),
            jnp.zeros(2),
        )
    )(jnp.asarray(covariances, jnp.float32))
    centers = np.array(
        [
            [[-0.6, 0.2], [0.8, -0.3]],
            [[-0.3, 0.1], [0.5, -0.4]],
            [[0.0, 0.7], [0.4, -0.2]],
            [[-0.5, 0.3], [0.6, 0.1]],
        ]
    )
    weights = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    noise = np.array([[0.3, -0.2], [-0.8, 0.5], [0.1, 0.6]])
    component_noise = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    reference = np.array([0.1, -0.2])
    particles, density, _centers = _mixture_particles(
        proposal,
        jnp.asarray(centers, jnp.float32),
        jnp.asarray(np.log(weights), jnp.float32),
        jnp.asarray(noise, jnp.float32),
        jnp.asarray(component_noise, jnp.float32),
        jnp.asarray(reference, jnp.float32),
        jnp.ones(2, dtype=bool),
    )
    components = np.argmax(np.log(weights[1:]) + component_noise, axis=1)
    expected_free = np.array(
        [
            centers[i + 1, k] + np.linalg.cholesky(expected_covariances[k]) @ noise[i]
            for i, k in enumerate(components)
        ]
    )
    np.testing.assert_allclose(particles, np.vstack((reference, expected_free)), atol=2e-6)
    expected_density = np.array(
        [
            logsumexp(
                [
                    np.log(weights[i, k])
                    + multivariate_normal.logpdf(
                        particle, mean=centers[i, k], cov=expected_covariances[k]
                    )
                    for k in range(2)
                ]
            )
            for i, particle in enumerate(np.asarray(particles))
        ]
    )
    target_log_density = np.array([-1.2, -2.1, -0.9, -1.7])
    np.testing.assert_allclose(
        target_log_density - np.asarray(density),
        target_log_density - expected_density,
        rtol=2e-6,
        atol=2e-6,
    )
