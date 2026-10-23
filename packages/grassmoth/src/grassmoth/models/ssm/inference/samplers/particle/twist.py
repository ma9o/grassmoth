"""Gaussian future-auxiliary information for exact twisted particle proposals.

The associative filtering operator is Särkkä & García-Fernández (2019),
https://arxiv.org/abs/1905.13002, Lemmas 7–8. Only the proposal uses this
approximation. Its coefficients are built at u, held fixed throughout CSMC.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class _InformationElement(NamedTuple):
    matrix: jax.Array
    mean: jax.Array
    covariance: jax.Array
    shift: jax.Array
    precision: jax.Array


class GaussianTwist(NamedTuple):
    """Unnormalized future information, with a unit twist at the final time."""

    precision: jax.Array
    shift: jax.Array

    def log_density(self, value: jax.Array, index: jax.Array | int) -> jax.Array:
        """Evaluate the same information factor in forward and backward weights."""
        return -0.5 * jnp.sum(
            value[..., :, None] * self.precision[index] * value[..., None, :], axis=(-2, -1)
        ) + jnp.sum(value * self.shift[index], axis=-1)


def symmetric(matrix: jax.Array) -> jax.Array:
    """Remove antisymmetric roundoff from a covariance or precision matrix."""
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _element(
    matrix: jax.Array,
    offset: jax.Array,
    covariance: jax.Array,
    precision: jax.Array,
    shift: jax.Array,
) -> _InformationElement:
    identity = jnp.eye(covariance.shape[-1], dtype=covariance.dtype)
    forward = identity + covariance @ precision
    # Covariance and precision are symmetric, so the reverse system is forward.T.
    # Factor one system and share its inverse across all five information fields.
    inverse = jnp.linalg.solve(forward, identity)
    conditioned = symmetric(inverse @ covariance)
    likelihood = symmetric(inverse.T @ precision)
    return _InformationElement(
        inverse @ matrix,
        inverse @ offset + conditioned @ shift,
        conditioned,
        matrix.T @ (inverse.T @ shift - likelihood @ offset),
        symmetric(matrix.T @ likelihood @ matrix),
    )


def _combine(earlier: _InformationElement, later: _InformationElement) -> _InformationElement:
    identity = jnp.eye(earlier.covariance.shape[-1], dtype=earlier.covariance.dtype)
    forward = identity + earlier.covariance @ later.precision
    inverse = jnp.linalg.solve(forward, identity)
    carried = later.matrix @ inverse
    returned = earlier.matrix.T @ inverse.T
    return _InformationElement(
        carried @ earlier.matrix,
        carried @ (earlier.mean + earlier.covariance @ later.shift) + later.mean,
        symmetric(carried @ earlier.covariance @ later.matrix.T + later.covariance),
        returned @ (later.shift - later.precision @ earlier.mean) + earlier.shift,
        symmetric(returned @ later.precision @ earlier.matrix + earlier.precision),
    )


def future_information(
    matrices: jax.Array,
    offsets: jax.Array,
    covariances: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
) -> GaussianTwist:
    """Integrate future pseudo-observations in logarithmic parallel depth."""
    steps, dimension = auxiliary.shape
    zero_precision = jnp.zeros((1, dimension, dimension), dtype=auxiliary.dtype)
    zero_shift = jnp.zeros((1, dimension), dtype=auxiliary.dtype)
    if steps == 1:
        return GaussianTwist(zero_precision, zero_shift)
    precision = jax.vmap(jnp.diag)(mask[1:] / variance[1:, None])
    shift = jnp.where(mask[1:], auxiliary[1:] / variance[1:, None], 0.0)
    elements = jax.vmap(_element)(matrices, offsets, covariances, precision, shift)
    reverse = jax.tree.map(lambda value: value[::-1], elements)
    suffix = jax.lax.associative_scan(
        jax.vmap(lambda later, earlier: _combine(earlier, later)), reverse
    )
    return GaussianTwist(
        jnp.concatenate((suffix.precision[::-1], zero_precision)),
        jnp.concatenate((suffix.shift[::-1], zero_shift)),
    )
