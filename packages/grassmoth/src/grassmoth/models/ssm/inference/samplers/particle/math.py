"""Numerical primitives for conditional trajectory kernels."""

from typing import Annotated

import jax
import jax.numpy as jnp
from pydantic import Field

from grassmoth.models.ssm.shapes import Array, Bool, Float, Int

TRITON_NUM_WARPS: Annotated[
    int,
    Field(
        description=(
            "Warps per fused GPU kernel, at 32 threads each. "
            "Heuristic: no paper or recorded benchmark sets it."
        )
    ),
] = 4


def label_logsumexp(values: Float[Array, "*batch K"]) -> Float[Array, " *batch"]:
    """Sum tiny label mixtures without a separate parallel reduction kernel."""
    if values.shape[-1] == 1:
        return values[..., 0]
    if values.shape[-1] == 2:
        return jnp.logaddexp(values[..., 0], values[..., 1])
    return jax.scipy.special.logsumexp(values, axis=-1)


def _masked_normal_log_prob(
    values: Float[Array, "*#batch D"],
    mean: Float[Array, "*#batch D"],
    variance: Float[Array, "*#batch #D"],
    free_mask: Bool[Array, "*#batch D"],
) -> Float[Array, " *batch"]:
    """Density with respect to the free coordinates, including its normalization."""
    per_coordinate = -0.5 * (
        jnp.log(2.0 * jnp.pi * variance) + jnp.square(values - mean) / variance
    )
    return jnp.sum(jnp.where(free_mask, per_coordinate, 0.0), axis=-1)


def _masked_mean(values: jax.Array, mask: jax.Array, *, axis: int | None = None) -> jax.Array:
    """Average diagnostics over sampled coordinates; an empty selection contributes zero."""
    return jnp.sum(jnp.where(mask, values, 0.0), axis=axis) / jnp.maximum(
        jnp.sum(mask, axis=axis), 1
    )


def _normalize_log_probs(
    logits: Float[Array, " *shape"], *, axis: int = -1
) -> Float[Array, " *shape"]:
    # Nonfinite densities have zero numerical support, never NaN selection mass.
    logits = jnp.where(jnp.isfinite(logits), logits, -jnp.inf)
    normalizer = jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)
    # A zero-mass prefix has no normalized law; keep every entry at -inf.
    return logits - jnp.where(jnp.isfinite(normalizer), normalizer, 0.0)


def _select_pytree(ensemble, index: Int[Array, " ..."]):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], ensemble)
