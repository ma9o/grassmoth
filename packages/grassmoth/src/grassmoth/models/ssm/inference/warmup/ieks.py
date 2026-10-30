"""Data-conditioned path centers used only to initialize particle chains.

Each Gauss--Newton iteration smooths an affine approximation of the dynamics
and a positive-semidefinite quadratic approximation of the exact point reading
densities. Neither approximation enters a sampling target or a diagnostic.
Strong nonlinearity, separated modes and clipped curvature can leave a poor
start. Interval readings are omitted: their cross-time information would
require window-memory augmentation, rather than independent point factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.execution.innovations import conditional_gaussian_factors
from grassmoth.models.ssm.inference.samplers.particle.twist import (
    GaussianTwist,
    future_information,
    symmetric,
)

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


def _smoothing_mean(
    matrices: jax.Array,
    offsets: jax.Array,
    covariances: jax.Array,
    information: GaussianTwist,
) -> jax.Array:
    """Integrate Gaussian future messages, then compose conditional affine means.

    Covariances may be singular on deterministic coordinates. The solves use
    I + Q J, so no artificial process noise or covariance inverse is needed.
    """
    future = future_information(matrices[1:], offsets[1:], covariances[1:], information)
    precision = information.precision + future.precision
    shift = information.shift + future.shift
    identity = jnp.eye(offsets.shape[-1], dtype=offsets.dtype)
    systems = identity + covariances @ precision
    conditioned_matrices = jnp.linalg.solve(systems, matrices)
    conditioned_offsets = jnp.linalg.solve(
        systems, (offsets + jnp.einsum("tij,tj->ti", covariances, shift))[..., None]
    )[..., 0]
    _, means = jax.lax.associative_scan(
        compose_affine_maps, (conditioned_matrices, conditioned_offsets)
    )
    return jnp.asarray(means)


def ieks_path(
    target: ParticleTarget, context: ParticleContext, initial_path: jax.Array
) -> jax.Array:
    """Run at most twenty damped smoothing iterations at fixed parameters.

    Exact replay enforces recurrences, static values and observed constraints
    after every trial. Backtracking accepts only improvement of the exact
    point-reading/path density. A failed numerical solve propagates a non-finite
    result to the initializer's typed failure boundary; it never selects a
    replacement flat path.
    """
    steps, dimension = initial_path.shape
    indices = jnp.arange(steps)
    density_mask = target.state_density_mask
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    initial_mean, initial_covariance = target.initial_moments(context)
    transition_covariances = jax.vmap(
        lambda previous, index: target.transition_moments(context, previous, index)[1]
    )(initial_path[:-1], indices[1:])
    covariances = jnp.concatenate((initial_covariance[None], transition_covariances))
    geometry = jax.vmap(
        lambda covariance, density, active: conditional_gaussian_factors(
            covariance, density_mask=density, innovation_mask=active
        )
    )(covariances, density_mask, free)
    conditioned_covariances = geometry.chol @ jnp.swapaxes(geometry.chol, -1, -2)
    conditioned_covariances = jnp.where(
        free[:, :, None] & free[:, None, :], conditioned_covariances, 0.0
    )
    pinned = geometry.centred_mask
    pinned_covariances = jnp.where(
        pinned[:, :, None] & pinned[:, None, :], covariances, 0.0
    ) + jax.vmap(jnp.diag)((~pinned).astype(initial_path.dtype))
    pinned_precisions = (
        jnp.linalg.solve(
            pinned_covariances,
            jnp.broadcast_to(jnp.eye(dimension, dtype=initial_path.dtype), covariances.shape),
        )
        * pinned[:, :, None]
        * pinned[:, None, :]
    )
    noise = target.prepare_noise_factors(context, target.observations)

    def reading(state: jax.Array, index: jax.Array) -> jax.Array:
        return target.observation_increment_with_noise(context, noise, state, index)

    def score(path: jax.Array) -> jax.Array:
        return jnp.asarray(
            target.prior_path_log_prob(context, path) + jnp.sum(jax.vmap(reading)(path, indices))
        )

    def iteration(path: jax.Array) -> jax.Array:
        def transition(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
            def mean(state: jax.Array) -> tuple[jax.Array, jax.Array]:
                value = target.transition_moments(context, state, index)[0]
                return value, value

            matrix, value = jax.jacfwd(mean, has_aux=True)(previous)
            return jnp.asarray(matrix), jnp.asarray(value - matrix @ previous)

        tail_matrices, tail_offsets = jax.vmap(transition)(path[:-1], indices[1:])
        matrices = jnp.concatenate(
            (jnp.zeros((1, dimension, dimension), dtype=path.dtype), tail_matrices)
        )
        offsets = jnp.concatenate((initial_mean[None], tail_offsets))
        # Conditioning stochastic pins retains their marginal density on the
        # previous state as an extra quadratic factor, including correlated noise.
        pin_precision = jnp.swapaxes(matrices, -1, -2) @ pinned_precisions @ matrices
        pin_shift = jnp.einsum(
            "tji,tjk,tk->ti", matrices, pinned_precisions, initial_path - offsets
        )
        identity = jnp.eye(dimension, dtype=path.dtype)
        conditioning = identity - geometry.regression
        conditioned_matrices = conditioning @ matrices
        conditioned_offsets = jnp.einsum("tij,tj->ti", conditioning, offsets) + jnp.einsum(
            "tij,tj->ti", geometry.regression, initial_path
        )
        # Static values and observed inputs have no stochastic path coordinate.
        static = (
            jnp.zeros(dimension, dtype=bool)
            .at[jnp.asarray(target.state_space.static + target.state_space.inputs, dtype=jnp.int32)]
            .set(True)
        )
        fixed = pinned | static[None, :]
        conditioned_matrices = jnp.where(fixed[:, :, None], 0.0, conditioned_matrices)
        conditioned_offsets = jnp.where(fixed, initial_path, conditioned_offsets)

        def point(state: jax.Array, index: jax.Array) -> GaussianTwist:
            gradient = jax.grad(reading)(state, index)
            hessian = symmetric(jax.hessian(reading)(state, index))
            eigenvalues, eigenvectors = jnp.linalg.eigh(-hessian)
            precision = symmetric((eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T)
            return GaussianTwist(precision, gradient + precision @ state)

        observed = jax.vmap(point)(path, indices)
        information = GaussianTwist(
            observed.precision.at[:-1].add(pin_precision[1:]),
            observed.shift.at[:-1].add(pin_shift[1:]),
        )
        proposed = _smoothing_mean(
            conditioned_matrices, conditioned_offsets, conditioned_covariances, information
        )
        direction = proposed - path
        current_score = score(path)

        def trial(
            carry: tuple[jax.Array, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            accepted, result, damping = carry
            candidate = target.replay_path(context, path + damping * direction)
            candidate_score = score(candidate)
            better = jnp.isfinite(candidate_score) & (candidate_score >= current_score)
            take = ~accepted & better
            return (
                accepted | better,
                jnp.where(take, candidate, result),
                damping * 0.5,
            ), None

        (_, result, _), _ = jax.lax.scan(
            trial, (jnp.asarray(False), path, jnp.asarray(1.0, dtype=path.dtype)), None, length=8
        )
        return jnp.where(jnp.all(jnp.isfinite(proposed)), result, jnp.full_like(result, jnp.nan))

    return jnp.asarray(jax.lax.fori_loop(0, 20, lambda _, path: iteration(path), initial_path))
