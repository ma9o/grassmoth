"""Shared covariance repair helpers for stable Cholesky factorization.

Generic over the square dimension: helpers operate on any covariance, latent
(``D``) or manifest (``M``), so they annotate it with the neutral axis ``N``
(see :mod:`grassmoth.models.ssm.shapes`). Instrumented for runtime shape
checks, hence no ``from __future__ import annotations`` (eager annotations keep
the jaxtyping imports resolvable for beartype).
"""

import jax.numpy as jnp

from grassmoth.models.ssm.shapes import Array, Float

# Defined locally so the covariance core does not depend on execution contracts.
CHOL_JITTER = 1e-8

INITIAL_STATE_COV_MIN_EIGENVALUE = 1e-6


def symmetrize(M: Float[Array, "*batch N N"]) -> Float[Array, "*batch N N"]:
    """Symmetrize a square matrix or stack of square matrices."""
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def symmetrize_with_jitter(
    M: Float[Array, "*batch N N"], *, jitter: float = CHOL_JITTER
) -> Float[Array, "*batch N N"]:
    """Symmetrize a square matrix or stack of square matrices and add diagonal jitter."""
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    return symmetrize(M) + eye * jitter
