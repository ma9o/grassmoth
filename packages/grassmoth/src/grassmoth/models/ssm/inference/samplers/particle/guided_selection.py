"""Exact backward, bounded-window replay and ancestral path selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from grassmoth.models.ssm.inference.samplers.particle.math import _masked_normal_log_prob

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import DTypeLike

    from grassmoth.models.ssm.inference.samplers.particle.contract import SmootherContext
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime

type PathSelector = Literal["backward", "window_replay", "ancestral"]


def _window_backward(
    memory: SequentialObservationRuntime,
    bank: jax.Array,
    weights: jax.Array,
    scores: jax.Array,
    last: jax.Array,
    key: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> jax.Array:
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
    )
    return jnp.concatenate((picks[::-1], last[None]))


def select_path(
    selector: PathSelector,
    target: ParticleTarget,
    ctx: SmootherContext,
    memory: SequentialObservationRuntime,
    memory_bank: jax.Array,
    bank: jax.Array,
    backward_weights: jax.Array,
    scores: jax.Array,
    observation_bank: jax.Array,
    ancestors: jax.Array,
    last: jax.Array,
    earlier_key: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    log_precisions: jax.Array | None,
    reference_dtype: DTypeLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Recover a path with the selector authorized by the compiled memory support."""
    steps, count, _ = bank.shape
    if selector != "backward":
        if selector == "ancestral":
            # The exact conditional filter's genealogy is valid even when a deterministic
            # stage feeds nonlinear later drift. No suffix splices or quadratic replay.
            def ancestor(pick: jax.Array, row: jax.Array) -> tuple[jax.Array, jax.Array]:
                parent = row[pick]
                return parent, parent

            _, earlier = jax.lax.scan(ancestor, last, ancestors, reverse=True)
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
            )
        path = bank[jnp.arange(steps), picks]
        if log_precisions is None:
            trajectory_scores = jax.vmap(
                lambda bound: target.path_log_prob(bound, path, target.observations)
            )(ctx.contexts)
        else:
            trajectory_scores = memory.path_log_probs(path)
    else:

        def backward(
            carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            following, suffix = carry
            index, gumbels = inputs
            links = jax.vmap(lambda value: transition(value, following, index + 1))(
                memory_bank[index]
            )
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
            ),
        )
        picks = jnp.concatenate((earlier[::-1], last[None]))
        # The backward suffix already contains every selected exact observation and
        # transition factor. Only the initial density remains; do not rescore the path.
        trajectory_scores = suffix + memory.initial_log_probs(initial_selected)
        path = bank[jnp.arange(steps), picks]
    return path, picks, cast("jax.Array", trajectory_scores)
