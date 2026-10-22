"""Associative composition of exact finite-dimensional affine maps."""

import jax
import jax.numpy as jnp


def compose_affine_maps(
    left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Compose right after left, retaining any leading parallel batch axes."""
    left_matrix, left_offset = left
    right_matrix, right_offset = right
    return right_matrix @ left_matrix, (
        jnp.einsum("...ij,...j->...i", right_matrix, left_offset) + right_offset
    )
