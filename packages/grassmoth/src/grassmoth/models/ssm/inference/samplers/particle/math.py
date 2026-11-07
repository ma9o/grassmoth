"""Numerical primitives for conditional trajectory kernels."""

from collections.abc import Sequence
from typing import Annotated, cast

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
    """Fold small mixtures; reduce arrays above 32 labels.

    L40S, SV C10/P128/T3000/u8, 2026-11-07: at M50 the array
    reduction reduced paired ordinary sweeps 97.76 to 82.48 ms and
    compile time 20.88 to 14.68 s. Retain the fold through 32 labels,
    where the wider fused tail avoids this per-step reduction. Routes still
    unfused at 17 to 32 labels retain the fold; the array form is unmeasured
    on them.
    """
    if values.shape[-1] > 32:
        maximum = jnp.max(values, axis=-1, keepdims=True)
        shift = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0.0))
        return jnp.log(jnp.sum(jnp.exp(values - shift), axis=-1)) + shift[..., 0]
    if values.shape[-1] == 1:
        return values[..., 0]
    maximum = values[..., 0]
    for index in range(1, values.shape[-1]):
        maximum = jnp.maximum(maximum, values[..., index])
    shift = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0.0))
    total = jnp.exp(values[..., 0] - shift)
    for index in range(1, values.shape[-1]):
        total = total + jnp.exp(values[..., index] - shift)
    return jnp.log(total) + shift


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


@jax.custom_batching.custom_vmap
def _gather_particles(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather owned particle indices using one flattened address under batching."""
    return jnp.take(values, indices, axis=0, mode="clip")


def _gather_particles_batch(
    axis_size: int,
    batched: Sequence[bool],
    values: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, bool]:
    values_batched, indices_batched = batched
    if not values_batched:
        return gather_particles(values, indices), True
    if not indices_batched:
        indices = jnp.broadcast_to(indices, (axis_size, *indices.shape))
    count = values.shape[1]
    offsets = (
        jnp.arange(axis_size, dtype=indices.dtype).reshape(
            (axis_size, *((1,) * (indices.ndim - 1)))
        )
        * count
    )
    flattened = values.reshape((axis_size * count, *values.shape[2:]))
    return gather_particles(flattened, indices + offsets), True


_gather_particles.def_vmap(_gather_particles_batch)


def gather_particles(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather valid particle indices with a typed, flattened batching rule."""
    return cast("jax.Array", _gather_particles(values, indices))


def categorical_from_uniform(logits: jax.Array, uniform: jax.Array) -> jax.Array:
    """Invert the categorical law with one uniform instead of K Gumbels.

    Static component folds avoid a separate scan at small label counts. The
    cumulative mass and total use the same additions, so trailing zero-mass
    labels cannot be selected by rounding a normalized CDF up to its endpoint.
    A prefix with no support keeps the previous argmax convention of label zero.
    """
    maximum = logits[..., 0]
    for label in range(1, logits.shape[-1]):
        maximum = jnp.maximum(maximum, logits[..., label])
    shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    masses = tuple(jnp.exp(logits[..., label] - shift) for label in range(logits.shape[-1]))
    total = masses[0]
    for mass in masses[1:]:
        total = total + mass
    threshold = uniform * total
    cumulative = masses[0]
    selected = jnp.zeros_like(uniform, dtype=jnp.int32)
    for label, mass in enumerate(masses[1:], 1):
        selected = jnp.where(threshold >= cumulative, label, selected)
        cumulative = cumulative + mass
    return jnp.where(total > 0, selected, 0)


def categorical_running_mass(logits: jax.Array) -> jax.Array:
    """Prepare unnormalized cumulative masses for repeated categorical draws."""
    maximum = jnp.max(logits, axis=-1, keepdims=True)
    mass = jnp.exp(logits - jnp.where(jnp.isfinite(maximum), maximum, 0.0))
    return jnp.cumsum(mass, axis=-1)


def categorical_from_running(running: jax.Array, uniform: jax.Array) -> jax.Array:
    """Select from carried masses without division or repeated exponentials."""
    threshold = uniform * running[..., -1]
    selected = jnp.sum(threshold[..., None] >= running[..., :-1], axis=-1).astype(jnp.int32)
    return jnp.where(running[..., -1] > 0, selected, 0)
