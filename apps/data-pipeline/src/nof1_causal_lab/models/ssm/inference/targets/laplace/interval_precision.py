"""Tridiagonal precision plus observed interval factors, for initialization only."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .shared import (
    _block_banded_logdet,
    _factor_block_profile_cholesky,
    _solve_block_profile_from_cholesky,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IntervalPrecision:
    """Precision B + U C Uᵀ, with block-tridiagonal B and interval Jacobian U."""

    diagonal: jax.Array
    upper: jax.Array
    jacobian: jax.Array
    information: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IntervalFactor:
    """Woodbury inverse factors and determinant of an interval precision."""

    diagonal_cholesky: jax.Array
    lower_cholesky: jax.Array
    inverse_jacobian: jax.Array
    correction: jax.Array
    logdet: jax.Array


def _base_solve(chol: jax.Array, lower: jax.Array, rhs: jax.Array) -> jax.Array:
    times = jnp.arange(chol.shape[0])
    return _solve_block_profile_from_cholesky(
        chol, lower, rhs, jnp.minimum(1, chol.shape[0] - 1 - times), jnp.minimum(1, times)
    )


def factor_interval_precision(precision: IntervalPrecision) -> IntervalFactor:
    """Factor using the determinant lemma without inverting interval information.

    C may be singular or have negative eigenvalues (e.g. Student-t curvature).
    The small system I + C Uᵀ B⁻¹ U therefore uses a general solve, not Cholesky.
    """
    diagonal, upper, jacobian, information = (
        precision.diagonal,
        precision.upper,
        precision.jacobian,
        precision.information,
    )
    times = jnp.arange(diagonal.shape[0])
    chol, lower = _factor_block_profile_cholesky(
        diagonal, upper, jnp.minimum(1, diagonal.shape[0] - 1 - times), jnp.minimum(1, times)
    )
    inverse_jacobian = _base_solve(chol, lower, jacobian)
    gram = jnp.einsum("tdk,tdl->kl", jacobian, inverse_jacobian)
    small = jnp.eye(information.shape[0], dtype=diagonal.dtype) + information @ gram
    correction = jnp.linalg.solve(small, information)
    correction = (correction + correction.T) / 2
    sign, logabsdet = jnp.linalg.slogdet(small)
    # Positive determinant alone permits an even number of negative eigenvalues.
    # This rank-sized congruence checks definiteness without a trajectory matrix.
    eigenvalues, eigenvectors = jnp.linalg.eigh((gram + gram.T) / 2)
    root = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0))[None, :]
    reduced = jnp.eye(information.shape[0], dtype=diagonal.dtype) + root.T @ information @ root
    positive = jnp.all(jnp.linalg.eigvalsh((reduced + reduced.T) / 2) > 0)
    logdet = _block_banded_logdet(chol) + jnp.where((sign > 0) & positive, logabsdet, jnp.nan)
    return IntervalFactor(chol, lower, inverse_jacobian, correction, logdet)


def solve_interval_precision(factor: IntervalFactor, rhs: jax.Array) -> jax.Array:
    """Apply the factored inverse to one trajectory-shaped right-hand side."""
    base = _base_solve(factor.diagonal_cholesky, factor.lower_cholesky, rhs)
    projected = jnp.einsum("tdk,td->k", factor.inverse_jacobian, rhs)
    return base - jnp.einsum("tdk,k->td", factor.inverse_jacobian, factor.correction @ projected)
