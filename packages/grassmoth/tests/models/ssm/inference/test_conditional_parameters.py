"""Exact conditional parameter kernels with a changing latent trajectory."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.sampling import ConditionalNutsSpec, MetropolisHastingsSpec
from grassmoth.models.ssm.inference.samplers.pgibbs.parameters import (
    metropolis_kernel,
    nuts_kernel,
)
from grassmoth_test_support.interval_particle import gaussian_interval_case

pytestmark = [
    pytest.mark.inference(concern="sampling"),
    pytest.mark.inference(concern="recovery"),
]


@pytest.mark.parametrize("method", ["mh", "nuts"])
def test_conditional_parameter_update_preserves_joint_posterior_and_freezes_adaptation(method):
    target, mean, covariance = gaussian_interval_case()
    warmup, retained = 250, 1500
    preconditioner = jnp.sqrt(jnp.asarray(covariance[:1, :1]))
    kernel = (
        metropolis_kernel(target, MetropolisHastingsSpec(), preconditioner, warmup)
        if method == "mh"
        else nuts_kernel(target, ConditionalNutsSpec(), preconditioner, warmup)
    )
    center = jnp.asarray(mean)
    loading = jnp.asarray(covariance[1:, 0] / covariance[0, 0])
    conditional_chol = jnp.linalg.cholesky(
        jnp.asarray(covariance[1:, 1:])
        - jnp.outer(jnp.asarray(covariance[1:, 0]), jnp.asarray(covariance[0, 1:]))
        / covariance[0, 0]
    )

    def chain(key):
        init_key, run_key = jax.random.split(key)
        path = center[1:, None]
        initial = kernel.initialize(center[:1], path, init_key)

        def update(carry, step_key):
            state, path = carry
            theta_key, path_key = jax.random.split(step_key)
            density = target.log_posterior(
                kernel.position(state), path, target.observations, target.times
            )
            state, _ = kernel.step(state, path, density, theta_key)
            position = kernel.position(state)
            # Independently known x|theta completes an exact blocked Gibbs sweep.
            # Its changing path catches a stale NUTS energy/gradient cache.
            path = (
                center[1:]
                + loading * (position[0] - center[0])
                + conditional_chol @ jax.random.normal(path_key, (len(mean) - 1,))
            )[:, None]
            return (state, path), (jnp.concatenate((position, path[:, 0])), kernel.step_size(state))

        _, (draws, scales) = jax.lax.scan(
            update, (initial, path), jax.random.split(run_key, warmup + retained)
        )
        return draws[warmup:], scales[warmup:]

    draws, scales = jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(348), 4))
    whitened = np.linalg.solve(
        np.linalg.cholesky(covariance), (np.asarray(draws).reshape(-1, len(mean)) - mean).T
    ).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(len(mean)), atol=0.15)
    np.testing.assert_array_equal(scales, np.broadcast_to(scales[:, :1], scales.shape))
