"""Normalized Gaussian path proposals, with logarithmic parallel depth.

Corenflos & Särkkä (2025), https://arxiv.org/abs/2303.00301, sections 2.2/2.4.
Backward information followed by forward affine sampling is the time-reversed
factorization of their parallel filter/backward sampler. Evaluating the same
normalized conditional factors avoids a separate marginal-likelihood pass.
These Gaussian laws are proposals only; the nonlinear target pays the MH ratio.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.inference.samplers.particle.twist import (
    GaussianTwist,
    future_information,
    symmetric,
)


class GaussianPathTerms(NamedTuple):
    """Affine dynamics and unnormalized local information for one proposal."""

    matrices: jax.Array
    offsets: jax.Array
    covariances: jax.Array
    information: GaussianTwist


class GaussianPathProposal(NamedTuple):
    """Forward conditionals of the complete Gaussian smoothing distribution."""

    matrices: jax.Array
    offsets: jax.Array
    cholesky: jax.Array
    free: jax.Array

    def sample(self, key: jax.Array) -> jax.Array:
        """Compose affine draws simultaneously across the time grid."""
        noise = jax.random.normal(key, self.offsets.shape, dtype=self.offsets.dtype)
        increments = self.offsets + jnp.einsum("tij,tj->ti", self.cholesky, noise) * self.free
        _, path = jax.lax.associative_scan(compose_affine_maps, (self.matrices, increments))
        return jnp.asarray(path)

    def log_density(self, path: jax.Array) -> jax.Array:
        """Use the exact factors used to draw, including their determinants."""
        previous = jnp.concatenate((jnp.zeros_like(path[:1]), path[:-1]))
        mean = self.offsets + jnp.einsum("tij,tj->ti", self.matrices, previous)
        residual = jnp.where(self.free, path - mean, 0.0)
        white = jax.vmap(
            lambda chol, value: jax.scipy.linalg.solve_triangular(chol, value, lower=True)
        )(self.cholesky, residual)
        return (
            -0.5 * jnp.sum(white**2)
            - jnp.sum(jnp.log(jnp.diagonal(self.cholesky, axis1=-2, axis2=-1)))
            - 0.5 * jnp.sum(self.free) * jnp.log(2.0 * jnp.pi)
        )


def gaussian_path_proposal(terms: GaussianPathTerms, free: jax.Array) -> GaussianPathProposal:
    """Condition every transition on all later information, then factor once.

    Zero-variance fixed coordinates get independent unit dummy noise solely for
    factorization. It is never sampled or scored, and contributes log(det)=0.
    """
    matrices, offsets, covariances, information = terms
    future = future_information(matrices[1:], offsets[1:], covariances[1:], information)
    precision = information.precision + future.precision
    shift = information.shift + future.shift
    identity = jnp.eye(offsets.shape[-1], dtype=offsets.dtype)
    systems = identity + covariances @ precision
    # Share the factorization across means and covariances at each time.
    rhs = jnp.concatenate(
        (
            matrices,
            (offsets + jnp.einsum("tij,tj->ti", covariances, shift))[..., None],
            covariances,
        ),
        axis=-1,
    )
    solved = jnp.linalg.solve(systems, rhs)
    dimension = offsets.shape[-1]
    covariance = symmetric(solved[..., dimension + 1 :])
    covariance = covariance + jax.vmap(jnp.diag)((~free).astype(offsets.dtype))
    return GaussianPathProposal(
        solved[..., :dimension], solved[..., dimension], jnp.linalg.cholesky(covariance), free
    )
