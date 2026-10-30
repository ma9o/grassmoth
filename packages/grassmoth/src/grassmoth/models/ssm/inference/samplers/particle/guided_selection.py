"""Exact backward, bounded-window replay and ancestral path selection."""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.models.ssm.inference.samplers.particle.math import (
    _masked_normal_log_prob,
    label_logsumexp,
)
from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int, PRNGKeyArray

type PathSelector = Literal["backward", "window_replay", "ancestral"]


def _cached_backward(
    weights: Float[Array, "T P"],
    links: Float[Array, "T-1 P P K"],
    scores: Float[Array, "T P K"],
    observations: Float[Array, "T P K"],
    gumbels: Float[Array, "T-1 P"],
    last: Int[Array, ""],
) -> tuple[Int[Array, " T"], Float[Array, " K"]]:
    """Keep the exact label suffix on device for the complete backward traversal."""
    from functools import partial

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    steps, count, labels = scores.shape
    if steps == 1:
        return last[None], observations[0, last]
    particle_block = pl.next_power_of_2(count)
    label_block = pl.next_power_of_2(labels)

    def kernel(
        links_ref,
        weights_ref,
        scores_ref,
        observations_ref,
        gumbels_ref,
        last_ref,
        picks_ref,
        suffix_ref,
    ):
        p = jnp.arange(particle_block)
        k = jnp.arange(label_block)
        last_pick = last_ref[()]
        initial_suffix = pltriton.load(
            observations_ref.at[steps - 1, last_pick, k], mask=k < labels, other=0.0
        )
        picks_ref[steps - 1] = last_pick

        def body(offset, carry):
            following, suffix = carry
            index = steps - 2 - offset
            valid = (p[:, None] < count) & (k[None, :] < labels)
            edge = pltriton.load(
                links_ref.at[index, following, p[:, None], k[None, :]], mask=valid, other=0.0
            )
            prefix = pltriton.load(
                scores_ref.at[index, p[:, None], k[None, :]], mask=valid, other=-jnp.inf
            )
            total = prefix + edge + suffix[None, :]
            maximum = jnp.max(total, axis=1)
            shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            mixed = jnp.log(jnp.sum(jnp.exp(total - shift[:, None]), axis=1)) + shift
            weight = pltriton.load(weights_ref.at[index, p], mask=p < count, other=-jnp.inf)
            noise = pltriton.load(gumbels_ref.at[index, p], mask=p < count, other=0.0)
            value = mixed + weight + noise
            pick = jnp.min(jnp.where(value == jnp.max(value), p, particle_block))
            chosen_edge = jnp.sum(jnp.where(p[:, None] == pick, edge, 0.0), axis=0)
            observed = pltriton.load(
                observations_ref.at[index, pick, k], mask=k < labels, other=0.0
            )
            picks_ref[index] = pick
            return pick, suffix + chosen_edge + observed

        _, suffix = jax.lax.fori_loop(0, steps - 1, body, (last_pick, initial_suffix))
        pltriton.store(suffix_ref.at[k], suffix, mask=k < labels)

    def run(*args, interpret):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((steps,), jnp.int32),
                jax.ShapeDtypeStruct((labels,), scores.dtype),
            ),
            interpret=interpret,
            compiler_params=pltriton.CompilerParams(num_warps=4),
            name="exact_cached_backward",
        )(*args)

    result = jax.lax.platform_dependent(
        links,
        weights,
        scores,
        observations,
        gumbels,
        last,
        cpu=partial(run, interpret=True),
        cuda=partial(run, interpret=False),
    )
    return cast("tuple[jax.Array, jax.Array]", result)


def _window_backward(
    memory: SequentialObservationRuntime,
    bank: Float[Array, "T P R"],
    weights: Float[Array, "T P"],
    scores: Float[Array, "T P K"],
    last: Int[Array, ""],
    key: PRNGKeyArray,
    auxiliary: Float[Array, "T D"],
    variance: Float[Array, "T D"],
    mask: Bool[Array, "T D"],
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    unroll: int,
) -> Int[Array, " T"]:
    """Pay all crossing windows and auxiliary factors before reusing an exact suffix.

    The compiler's window span bounds the replay. Beyond it, every label's future
    factors are independent of this prefix; their cached sums remain inside the
    label mixture. No approximation or finite-lag truncation enters these weights.
    """
    window = memory.windows
    assert window is not None
    steps, count, _ = bank.shape
    states = memory.state(bank)
    path = jnp.zeros_like(states[:, 0]).at[-1].set(states[-1, last])
    suffix = jnp.zeros((steps + 1, scores.shape[-1]), dtype=scores.dtype)

    def backward(carry, inputs):
        selected, suffix_scores = carry
        index, gumbels = inputs
        stop = window.replay_stops[index]

        def replay(carry, offset):
            previous, label_sum, auxiliary_sum = carry
            time = jnp.minimum(index + 1 + offset, steps - 1)

            def active():
                current = jax.vmap(memory.advance, in_axes=(0, None, None))(
                    previous, selected[time], time
                )
                observed, gradients = jax.vmap(observe, in_axes=(0, None))(current, time)
                density = _masked_normal_log_prob(
                    auxiliary[time],
                    selected[time] + variance[time] * gradients,
                    variance[time],
                    mask[time],
                )
                factors = observed + jax.vmap(transition, in_axes=(0, 0, None))(
                    previous, current, time
                )
                return (current, label_sum + factors, auxiliary_sum + density), factors

            return jax.lax.cond(
                index + 1 + offset <= stop,
                active,
                lambda: (carry, jnp.zeros_like(label_sum)),
            )

        (_, label_sum, auxiliary_sum), factors = jax.lax.scan(
            replay,
            (bank[index], jnp.zeros_like(scores[index]), jnp.zeros_like(weights[index])),
            jnp.arange(window.max_replay),
        )
        logits = (
            weights[index]
            + auxiliary_sum
            + label_logsumexp(scores[index] + label_sum + suffix_scores[stop + 1])
        )
        pick = jnp.argmax(logits + gumbels).astype(jnp.int32)

        def replace_suffix(carry, offset):
            total, stored = carry
            time = jnp.minimum(index + 1 + offset, steps - 1)

            def active():
                updated = total + factors[offset, pick]
                return updated, stored.at[time].set(updated)

            return jax.lax.cond(index + 1 + offset <= stop, active, lambda: carry), None

        (_, suffix_scores), _ = jax.lax.scan(
            replace_suffix,
            (suffix_scores[stop + 1], suffix_scores),
            jnp.arange(window.max_replay - 1, -1, -1),
        )
        return (selected.at[index].set(states[index, pick]), suffix_scores), pick

    _, picks = jax.lax.scan(
        backward,
        (path, suffix),
        (
            jnp.arange(steps - 2, -1, -1),
            jax.vmap(lambda k: jax.random.gumbel(k, (count,), dtype=states.dtype))(
                jax.random.split(key, steps - 1)
            ),
        ),
        unroll=unroll,
    )
    return jnp.concatenate((picks[::-1], last[None]))


def select_path(
    selector: PathSelector,
    memory: SequentialObservationRuntime,
    memory_bank: Float[Array, "T P R"],
    bank: Float[Array, "T P D"],
    backward_weights: Float[Array, "T P"],
    scores: Float[Array, "T P K"],
    observation_bank: Float[Array, "T P K"],
    ancestors: Int[Array, "T-1 P"],
    last: Int[Array, ""],
    earlier_key: PRNGKeyArray,
    auxiliary: Float[Array, "T D"],
    variance: Float[Array, "T D"],
    mask: Bool[Array, "T D"],
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    reference_dtype: DTypeLike,
    unroll: int,
) -> tuple[Float[Array, "T D"], Int[Array, " T"], Float[Array, " K"]]:
    """Recover a path with the selector authorized by the compiled memory support."""
    steps, count, _ = bank.shape
    if selector != "backward":
        if selector == "ancestral":
            # The exact conditional filter's genealogy is valid even when a deterministic
            # stage feeds nonlinear later drift. No suffix splices or quadratic replay.
            def ancestor(
                pick: Int[Array, ""], row: Int[Array, " P"]
            ) -> tuple[Int[Array, ""], Int[Array, ""]]:
                parent = row[pick]
                return parent, parent

            _, earlier = jax.lax.scan(ancestor, last, ancestors, reverse=True, unroll=unroll)
            picks = jnp.concatenate((earlier, last[None]))
        else:
            picks = _window_backward(
                memory,
                memory_bank,
                backward_weights,
                scores,
                last,
                earlier_key,
                auxiliary,
                variance,
                mask,
                observe,
                transition,
                unroll,
            )
        path = bank[jnp.arange(steps), picks]
        if memory.log_precisions is None:
            trajectory_scores = jax.vmap(
                lambda bound: memory.target.path_log_prob(bound, path, memory.target.observations)
            )(memory.contexts)
        else:
            trajectory_scores = memory.path_log_probs(path)
    else:
        # The point-reading predecessor bank is fixed before backward selection.
        # Evaluate its exact nonlinear means over all times in parallel; only the
        # successor choice and label suffix remain sequential. Window replay above
        # cannot reuse this table because its reconstructed predecessors change.
        means = jax.vmap(
            lambda values, index: jax.vmap(memory.transition_means, in_axes=(0, None))(
                values, index
            )
        )(memory_bank[:-1], jnp.arange(1, steps))

        # All candidate successors are known after filtering. Materialize exact
        # links over time and both particle axes before the suffix recursion.
        # This trades O(T P^2 K) storage for parallel transition scoring.
        def pairwise(
            previous: Float[Array, "P R"],
            current: Float[Array, "P R"],
            index: Int[Array, ""],
            transition_means: Float[Array, "P K D"],
        ) -> Float[Array, "P P K"]:
            return jax.vmap(
                lambda following: jax.vmap(
                    lambda value, mean: memory.transition_log_probs_with_means(
                        value, following, index, mean
                    )
                )(previous, transition_means)
            )(current)

        links_bank = jax.lax.optimization_barrier(
            jax.vmap(pairwise)(
                memory_bank[:-1], memory_bank[1:], jnp.arange(1, steps), means
            ).astype(scores.dtype)
        )

        gumbels = jax.vmap(lambda k: jax.random.gumbel(k, (count,), dtype=reference_dtype))(
            jax.random.split(earlier_key, steps - 1)
        )[::-1]
        picks, suffix = _cached_backward(
            backward_weights, links_bank, scores, observation_bank, gumbels, last
        )
        initial_selected = picks[0]
        # The backward suffix already contains every selected exact observation and
        # transition factor. Only the initial density remains; do not rescore the path.
        trajectory_scores = suffix + memory.initial_log_probs(memory_bank[0, initial_selected])
        path = bank[jnp.arange(steps), picks]
    return path, picks, cast("jax.Array", trajectory_scores)
