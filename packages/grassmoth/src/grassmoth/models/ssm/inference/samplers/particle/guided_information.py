"""Proposal-only information rules, held fixed at the auxiliary path.

These approximations never evaluate the exact reading or auxiliary corrections.
Point and augmented-memory geometry share the information choice, but retain
their coordinate-specific algebra and operation order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist, symmetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime

type InformationRule = Literal["unshifted", "actual_law"]


def _unshifted_information(
    auxiliary: jax.Array, variance: jax.Array, mask: jax.Array
) -> GaussianTwist:
    return GaussianTwist(
        jax.vmap(jnp.diag)(mask / variance),
        jnp.where(mask, auxiliary / variance, 0.0),
    )


def _guided_information(
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
) -> GaussianTwist:
    """Approximate readings times the actual auxiliary law, held fixed at u.

    For score g and Hessian H at u, the auxiliary mean's local affine map is
    B = I + diag(v) H, c = v * (g - H u). Its information is
    B.T diag(1/v) B and B.T ((u - c) / v), including when B vanishes. The reading factor uses
    the positive-semidefinite part of -H and retains its score at u.
    A Gaussian observation factor gives its exact product. Curvature clipping,
    nonlinear score extrapolation and separated modes can cost mixing; the
    exact particle corrections still use the true nonlinear model.
    """

    def point(value: jax.Array, v: jax.Array, free: jax.Array, index: jax.Array) -> GaussianTwist:
        def score(state: jax.Array) -> tuple[jax.Array, jax.Array]:
            gradient = observe(state, index)[1]
            return gradient, gradient

        hessian, gradient = jax.jacfwd(score, has_aux=True)(value)
        hessian = symmetric(jnp.asarray(hessian)) * free[:, None] * free[None, :]
        gradient = jnp.where(free, jnp.asarray(gradient), 0.0)
        matrix = jnp.diag(free.astype(value.dtype)) + v[:, None] * hessian
        residual = jnp.where(free, value - v * gradient + v * (hessian @ value), 0.0)
        auxiliary_precision = symmetric(matrix.T @ (matrix / v[:, None]))
        auxiliary_shift = matrix.T @ (residual / v)
        eigenvalues, eigenvectors = jnp.linalg.eigh(-hessian)
        observed_precision = symmetric(
            (eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        )
        observed_shift = gradient + observed_precision @ value
        return GaussianTwist(
            auxiliary_precision + observed_precision, auxiliary_shift + observed_shift
        )

    return jax.vmap(point)(auxiliary, variance, mask, jnp.arange(auxiliary.shape[0]))


def memory_information(
    rule: InformationRule,
    memory: SequentialObservationRuntime,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    label_log_probs: jax.Array,
    points: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    augmented_mask: jax.Array,
) -> GaussianTwist:
    """Apply the information rule in carried coordinates, without adding process noise."""
    steps, dimension = mask.shape

    def information_at(
        value: jax.Array, v: jax.Array, free: jax.Array, active: jax.Array, index: jax.Array
    ) -> GaussianTwist:
        if rule == "unshifted":
            precision = jnp.zeros((memory.dimension, memory.dimension), value.dtype)
            precision = precision.at[:dimension, :dimension].set(jnp.diag(free / v))
            return GaussianTwist(
                precision,
                jnp.zeros_like(value)
                .at[:dimension]
                .set(jnp.where(free, value[:dimension] / v, 0.0)),
            )

        def reading(state):
            return logsumexp(label_log_probs + memory.log_probs(state, index))

        hessian = symmetric(jax.hessian(reading)(value)) * active[:, None] * active[None, :]
        gradient = jnp.where(active, jax.grad(reading)(value), 0.0)
        eigenvalues, eigenvectors = jnp.linalg.eigh(-hessian)
        observed = symmetric((eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T)

        def auxiliary_mean(state):
            return memory.state(state) + v * observe(state, index)[1]

        mean = auxiliary_mean(value)
        matrix = jax.jacfwd(auxiliary_mean)(value) * free[:, None] * active[None, :]
        residual = jnp.where(free, memory.state(value) - mean + matrix @ value, 0.0)
        return GaussianTwist(
            symmetric(matrix.T @ (matrix / v[:, None])) + observed,
            matrix.T @ (residual / v) + gradient + observed @ value,
        )

    return jax.vmap(information_at)(points, variance, mask, augmented_mask, jnp.arange(steps))


def point_information(
    rule: InformationRule,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
) -> GaussianTwist:
    """Apply the same information choice in independent point coordinates."""
    if rule == "actual_law":
        return _guided_information(observe, auxiliary, variance, mask)
    return _unshifted_information(auxiliary, variance, mask)
