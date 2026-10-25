"""Gaussian information for exact twisted particle proposals.

The associative filtering operator is Särkkä & García-Fernández (2019),
https://arxiv.org/abs/1905.13002, Lemmas 7–8. Only the proposal uses this
approximation. Its coefficients are built at u, held fixed throughout CSMC.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.shapes import Array, Float, Int


class _InformationElement(NamedTuple):
    matrix: jax.Array
    mean: jax.Array
    covariance: jax.Array
    shift: jax.Array
    precision: jax.Array


class GaussianTwist(NamedTuple):
    """An unnormalized Gaussian factor in precision and information-vector form."""

    precision: jax.Array
    shift: jax.Array

    def log_density(
        self, value: Float[Array, "*batch {self.shift.shape[-1]}"], index: Int[Array, ""] | int
    ) -> Float[Array, " *batch"]:
        """Evaluate the same information factor in forward and backward weights."""
        return -0.5 * jnp.sum(
            value[..., :, None] * self.precision[index] * value[..., None, :], axis=(-2, -1)
        ) + jnp.sum(value * self.shift[index], axis=-1)


def symmetric(matrix: Float[Array, "*batch N N"]) -> Float[Array, "*batch N N"]:
    """Remove antisymmetric roundoff from a covariance or precision matrix."""
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _element(
    matrix: Float[Array, "N N"],
    offset: Float[Array, " N"],
    covariance: Float[Array, "N N"],
    precision: Float[Array, "N N"],
    shift: Float[Array, " N"],
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
    matrices: Float[Array, "{information.shift.shape[0]-1} N N"],
    offsets: Float[Array, "{information.shift.shape[0]-1} N"],
    covariances: Float[Array, "{information.shift.shape[0]-1} N N"],
    information: GaussianTwist,
) -> GaussianTwist:
    """Integrate future factors in logarithmic depth, including zero information."""
    steps, dimension = information.shift.shape
    zero_precision = jnp.zeros((1, dimension, dimension), dtype=information.shift.dtype)
    zero_shift = jnp.zeros((1, dimension), dtype=information.shift.dtype)
    if steps == 1:
        return GaussianTwist(zero_precision, zero_shift)
    elements = jax.vmap(_element)(
        matrices, offsets, covariances, information.precision[1:], information.shift[1:]
    )
    reverse = jax.tree.map(lambda value: value[::-1], elements)
    suffix = jax.lax.associative_scan(
        jax.vmap(lambda later, earlier: _combine(earlier, later)), reverse
    )
    return GaussianTwist(
        jnp.concatenate((suffix.precision[::-1], zero_precision)),
        jnp.concatenate((suffix.shift[::-1], zero_shift)),
    )
