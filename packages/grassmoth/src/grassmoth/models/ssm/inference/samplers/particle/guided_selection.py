"""Exact backward, bounded-window replay and ancestral path selection."""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import DTypeLike

from grassmoth.models.ssm.inference.samplers.particle.math import _masked_normal_log_prob
from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int, PRNGKeyArray

type PathSelector = Literal["backward", "window_replay", "ancestral"]


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
            + logsumexp(scores[index] + label_sum + suffix_scores[stop + 1], axis=-1)
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

        def backward(
            carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            following, suffix = carry
            index, gumbels, transition_means = inputs
            links = jax.vmap(
                lambda value, mean: memory.transition_log_probs_with_means(
                    value, following, index + 1, mean
                )
            )(memory_bank[index], transition_means).astype(scores.dtype)
            logits = backward_weights[index] + logsumexp(scores[index] + links + suffix, axis=-1)
            pick = jnp.argmax(logits + gumbels).astype(jnp.int32)
            value = memory_bank[index, pick]
            return (value, suffix + links[pick] + observation_bank[index, pick]), pick

        (initial_selected, suffix), earlier = jax.lax.scan(
            backward,
            (memory_bank[-1, last], observation_bank[-1, last]),
            (
                jnp.arange(steps - 2, -1, -1),
                jax.vmap(lambda k: jax.random.gumbel(k, (count,), dtype=reference_dtype))(
                    jax.random.split(earlier_key, steps - 1)
                ),
                means[::-1],
            ),
            unroll=unroll,
        )
        picks = jnp.concatenate((earlier[::-1], last[None]))
        # The backward suffix already contains every selected exact observation and
        # transition factor. Only the initial density remains; do not rescore the path.
        trajectory_scores = suffix + memory.initial_log_probs(initial_selected)
        path = bank[jnp.arange(steps), picks]
    return path, picks, cast("jax.Array", trajectory_scores)
