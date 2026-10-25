"""Numerical primitives for conditional trajectory kernels."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from grassmoth.artifacts.sampling import LatentBlockSize

FloatScalar = Float[Array, ""]


def multinomial(key: jax.Array, logits: jax.Array, count: int) -> jax.Array:
    """Draw by inverse CDF, with linear storage instead of a count-by-category Gumbel array."""
    cumulative = jnp.cumsum(jax.nn.softmax(logits))
    uniforms = jax.random.uniform(key, (count,), dtype=cumulative.dtype)
    return jnp.minimum(jnp.searchsorted(cumulative, uniforms, side="right"), logits.size - 1)


def coordinate_proposal_mask(
    key: jax.Array, free_mask: jax.Array, block_coords: LatentBlockSize
) -> jax.Array:
    """Choose only coordinates with independent, unobserved values to propose."""
    dimension = free_mask.shape[-1]
    if block_coords == "all" or block_coords >= dimension:
        return free_mask
    scores = jnp.where(jnp.any(free_mask, axis=0), jax.random.uniform(key, (dimension,)), -jnp.inf)
    chosen = jnp.argsort(scores)[-block_coords:]
    return free_mask & jnp.zeros(dimension, dtype=bool).at[chosen].set(True)[None, :]


def _value_and_grad_by_param[Context](
    contexts: Context,
    particle: jax.Array,
    log_prob_fn: Callable[[Context, jax.Array], jax.Array],
    *,
    value_dtype: DTypeLike,
    grad_dtype: DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    """Differentiate the explicit particle argument and batch over parameter contexts."""
    evaluate: Callable[[Context, jax.Array], tuple[jax.Array, jax.Array]] = jax.value_and_grad(
        log_prob_fn, argnums=1
    )
    evaluate_by_param: Callable[[Context, jax.Array], tuple[jax.Array, jax.Array]] = jax.vmap(
        evaluate, in_axes=(0, None)
    )
    value, grad = evaluate_by_param(contexts, particle)
    return value.astype(value_dtype), grad.astype(grad_dtype)


def _masked_normal_log_prob(
    values: jax.Array, mean: jax.Array, variance: jax.Array, free_mask: jax.Array
) -> jax.Array:
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
    logits: Float[Array, "*shape"], *, axis: int = -1
) -> Float[Array, "*shape"]:
    # Nonfinite densities have zero numerical support, never NaN selection mass.
    logits = jnp.where(jnp.isfinite(logits), logits, -jnp.inf)
    normalizer = jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)
    # A zero-mass prefix has no normalized law; keep every entry at -inf.
    return logits - jnp.where(jnp.isfinite(normalizer), normalizer, 0.0)


def _observation_log_probs_by_param(
    contexts,
    particles_t: Float[Array, "P D"],
    time_idx: Int[Array, ""],
    runtime_observations: Float[Array, "T M"],
    obs_increment_fn,
) -> Float[Array, "P K"]:
    def _one_param(context):
        return jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                time_idx,
                runtime_observations,
            )
        )(particles_t)

    return jnp.swapaxes(jax.vmap(_one_param)(contexts), 0, 1)


def _select_pytree(ensemble, index: Int[Array, "..."]):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], ensemble)
