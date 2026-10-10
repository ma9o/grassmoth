"""Construct a fixed auxiliary Gaussian at the particle sampler's initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import GaussianReference

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


def build_gaussian_reference(
    target: ParticleTarget,
    *,
    positions: jax.Array,
    paths: jax.Array,
    parameter_chol: jax.Array | None,
    scale: float,
) -> GaussianReference:
    """Freeze initialization moments and transition Jacobians on free coordinates.

    This Gaussian is an auxiliary distribution, never a production likelihood.
    Between-chain parameter dispersion supplements the proposal preconditioner.
    The exact target continues to own deterministic replay and every cold density.
    """
    parameter_mean = jnp.mean(positions, axis=0)
    dimension = parameter_mean.size
    base_chol = (
        jnp.eye(dimension, dtype=parameter_mean.dtype) if parameter_chol is None else parameter_chol
    )
    deviations = positions - parameter_mean
    parameter_covariance = base_chol @ base_chol.T + deviations.T @ deviations / positions.shape[0]
    context = target.context(parameter_mean, target.times)
    path_mean = target.replay_path(context, jnp.mean(paths, axis=0))
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask

    def transition(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        jacobian = jax.jacfwd(lambda state: target.transition_moments(context, state, index)[0])(
            previous
        )
        return jnp.asarray(jacobian), target.transition_moments(context, previous, index)[1]

    matrices, covariances = jax.vmap(transition)(path_mean[:-1], jnp.arange(1, target.times.size))
    state_dimension = path_mean.shape[-1]
    matrices = jnp.concatenate((jnp.zeros((1, state_dimension, state_dimension)), matrices))
    previous_free = jnp.concatenate((jnp.zeros_like(free[:1]), free[:-1]))
    matrices = jnp.where(free[:, :, None] & previous_free[:, None, :], matrices, 0.0)
    covariances = jnp.concatenate((target.initial_moments(context)[1][None], covariances))
    embedded_covariance = jnp.where(free[:, :, None] & free[:, None, :], scale**2 * covariances, 0)
    embedded_covariance = embedded_covariance + jax.vmap(jnp.diag)((~free).astype(path_mean.dtype))
    return GaussianReference(
        parameter_mean=parameter_mean,
        parameter_chol=scale * jnp.linalg.cholesky(parameter_covariance),
        path_mean=path_mean,
        autoregression=matrices,
        innovation_chol=jnp.linalg.cholesky(embedded_covariance),
        free_mask=free,
    )
