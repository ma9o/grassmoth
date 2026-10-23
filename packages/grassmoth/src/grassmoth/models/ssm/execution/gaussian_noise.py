"""Exact Gaussian density factors for fixed noise and changing nonlinear means."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class GaussianNoiseFactors:
    """A whitening map and normalizer, with any leading batch axes retained."""

    whitener: jax.Array
    log_normalizer: jax.Array

    @classmethod
    def from_cholesky(cls, chol: jax.Array) -> GaussianNoiseFactors:
        """Prepare one exact noise law outside the repeated state evaluation."""
        dimension = chol.shape[-1]
        inverse = solve_triangular(chol, jnp.eye(dimension, dtype=chol.dtype), lower=True)
        constant = -jnp.sum(jnp.log(jnp.diag(chol))) - 0.5 * dimension * jnp.log(2 * jnp.pi)
        return cls(inverse, constant)

    def at(self, index: jax.Array | int) -> GaussianNoiseFactors:
        """Select one time from a prepared sequence."""
        return GaussianNoiseFactors(self.whitener[index], self.log_normalizer[index])

    def log_prob(self, residual: jax.Array) -> jax.Array:
        """Score a residual without a factorization or solve."""
        whitened = jnp.sum(self.whitener * residual[..., None, :], axis=-1)
        return self.log_normalizer - 0.5 * jnp.sum(whitened**2, axis=-1)
