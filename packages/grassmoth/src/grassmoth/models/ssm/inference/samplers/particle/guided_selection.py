"""Exact backward, bounded-window replay and ancestral path selection."""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.models.ssm.inference.samplers.particle.backward_stream import streamed_backward
from grassmoth.models.ssm.inference.samplers.particle.guided_random import particle_gumbel
from grassmoth.models.ssm.inference.samplers.particle.math import (
    TRITON_NUM_WARPS,
    _masked_normal_log_prob,
    label_logsumexp,
)
from grassmoth.models.ssm.inference.samplers.particle.point_program import (
    CompiledPointKernels,
    PointKernelRoute,
)
from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int, PRNGKeyArray

type PathSelector = Literal["backward", "window_replay", "ancestral"]


def _backward_from_moments(
    weights: Float[Array, "T P"],
    means: Float[Array, "T-1 P K D"],
    points: Float[Array, "T-1 P D"],
    whiteners: jax.Array,
    normalizers: Float[Array, "T-1 K"],
    first_scores: Float[Array, "P K"],
    scores: Float[Array, "S P K"],
    first_observations: Float[Array, "P K"],
    observations: Float[Array, "S P K"],
    gumbels: Float[Array, "T-1 P"],
    last: Int[Array, ""],
) -> tuple[Int[Array, " T"], Float[Array, " K"]]:
    """Score only the selected successor, retaining every exact label suffix.

    Means contain the model's exact nonlinear drift. The prepared Gaussian
    transition factors supply the same density as the target scorer. Evaluating
    predecessor edges on demand replaces the quadratic particle-pair table with
    linear storage. Matrix, label and particle extents specialize at compilation.
    For vector states, whitening both banks about a shared origin moves the dense
    products outside the sequential traversal. The origin cancels in differences
    and keeps float32 products independent of a large absolute state location.
    """
    from functools import partial

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    steps, count = weights.shape
    labels = first_scores.shape[-1]
    if steps == 1:
        return last[None], first_observations[last]
    particle_block = pl.next_power_of_2(count)
    label_block = pl.next_power_of_2(labels)
    dimension = means.shape[-1]
    dimension_block = pl.next_power_of_2(dimension)
    diagonal = whiteners.ndim == 3
    if dimension > 1:
        origin = means[:, :1]
        if diagonal:
            means = (means - origin) * whiteners[:, None]
            points = (points[:, :, None] - origin) * whiteners[:, None]
        else:
            means = jnp.einsum(
                "tpkd,tked->tpke", means - origin, whiteners, precision=jax.lax.Precision.HIGHEST
            )
            points = jnp.einsum(
                "tpkd,tked->tpke",
                points[:, :, None] - origin,
                whiteners,
                precision=jax.lax.Precision.HIGHEST,
            )

    if dimension == 0:
        # Empty transition laws have zero quadratic form. Pallas's interpreter
        # requires nonempty operand buffers even when the kernel never reads them.
        means = points = whiteners = jnp.zeros((steps - 1, 1), dtype=scores.dtype)

    def kernel(
        means_ref,
        points_ref,
        whiteners_ref,
        normalizers_ref,
        weights_ref,
        first_scores_ref,
        scores_ref,
        first_observations_ref,
        observations_ref,
        gumbels_ref,
        last_ref,
        picks_ref,
        suffix_ref,
    ):
        chain = pl.program_id(0)
        p = jnp.arange(particle_block)
        k = jnp.arange(label_block)
        d = jnp.arange(dimension_block)
        last_pick = last_ref[chain]
        initial_suffix = pltriton.load(
            observations_ref.at[steps - 2, chain, last_pick, k], mask=k < labels, other=0.0
        )
        picks_ref[chain, steps - 1] = last_pick

        def body(offset, carry):
            following, suffix = carry
            index = steps - 2 - offset
            valid = (p[:, None] < count) & (k[None, :] < labels)
            normalizer = pltriton.load(
                normalizers_ref.at[index, chain, k], mask=k < labels, other=0.0
            )
            if dimension == 0:
                norm = jnp.zeros((particle_block, label_block), dtype=scores.dtype)
            else:
                centers = pltriton.load(
                    means_ref.at[
                        index, chain, p[:, None, None], k[None, :, None], d[None, None, :]
                    ],
                    mask=valid[:, :, None] & (d[None, None, :] < dimension),
                    other=0.0,
                )
                if dimension > 1:
                    following_point = pltriton.load(
                        points_ref.at[index, chain, following, k[:, None], d[None, :]],
                        mask=(k[:, None] < labels) & (d[None, :] < dimension),
                        other=0.0,
                    )
                else:
                    following_point = pltriton.load(
                        points_ref.at[index, chain, following, d[None, :]],
                        mask=d[None, :] < dimension,
                        other=0.0,
                    )
                residual = following_point[None, :, :] - centers

                def squared_norm(row, total):
                    address = (
                        whiteners_ref.at[index, chain, k[:, None], d[None, :]]
                        if diagonal
                        else whiteners_ref.at[index, chain, k[:, None], row, d[None, :]]
                    )
                    whitener = pltriton.load(
                        address,
                        mask=(k[:, None] < labels) & (d[None, :] < dimension),
                        other=0.0,
                    )
                    white = jnp.sum(whitener[None, :, :] * residual, axis=-1)
                    return total + white * white

                norm = (
                    jnp.sum(residual * residual, axis=-1)
                    if dimension > 1
                    else squared_norm(
                        0, jnp.zeros((particle_block, label_block), dtype=scores.dtype)
                    )
                )
            edge = normalizer[None, :] - 0.5 * norm
            first_prefix = pltriton.load(
                first_scores_ref.at[chain, p[:, None], k[None, :]],
                mask=valid & (index == 0),
                other=-jnp.inf,
            )
            tail_prefix = pltriton.load(
                scores_ref.at[jnp.maximum(index - 1, 0), chain, p[:, None], k[None, :]],
                mask=valid & (index > 0),
                other=-jnp.inf,
            )
            prefix = jnp.where(index == 0, first_prefix, tail_prefix)
            total = prefix + edge + suffix[None, :]
            maximum = jnp.max(total, axis=1)
            shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            mixed = jnp.log(jnp.sum(jnp.exp(total - shift[:, None]), axis=1)) + shift
            weight = pltriton.load(weights_ref.at[index, chain, p], mask=p < count, other=-jnp.inf)
            noise = pltriton.load(gumbels_ref.at[chain, offset, p], mask=p < count, other=0.0)
            value = mixed + weight + noise
            pick = jnp.min(jnp.where(value == jnp.max(value), p, particle_block))
            chosen_edge = jnp.sum(jnp.where(p[:, None] == pick, edge, 0.0), axis=0)
            first_observed = pltriton.load(
                first_observations_ref.at[chain, pick, k],
                mask=(k < labels) & (index == 0),
                other=0.0,
            )
            tail_observed = pltriton.load(
                observations_ref.at[jnp.maximum(index - 1, 0), chain, pick, k],
                mask=(k < labels) & (index > 0),
                other=0.0,
            )
            observed = jnp.where(index == 0, first_observed, tail_observed)
            picks_ref[chain, index] = pick
            return pick, suffix + chosen_edge + observed

        _, suffix = jax.lax.fori_loop(0, steps - 1, body, (last_pick, initial_suffix))
        pltriton.store(suffix_ref.at[chain, k], suffix, mask=k < labels)

    def run(*args, interpret):
        chains = args[-1].shape[0]
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((chains, steps), jnp.int32),
                jax.ShapeDtypeStruct((chains, labels), scores.dtype),
            ),
            grid=(chains,),
            interpret=interpret,
            compiler_params=pltriton.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="exact_backward_from_time_major_moments",
        )(*args)

    def dispatch(*args):
        return jax.lax.platform_dependent(
            *args, cpu=partial(run, interpret=True), cuda=partial(run, interpret=False)
        )

    timed = (True, True, True, True, True, False, True, False, True, False, False)

    @jax.custom_batching.custom_vmap
    def call(*args):
        values = tuple(
            jnp.expand_dims(value, 1 if time_axis else 0)
            for value, time_axis in zip(args, timed, strict=True)
        )
        return tuple(value[0] for value in dispatch(*values))

    @call.def_vmap
    def batch(chains, batched, *args):
        values = tuple(
            value if mapped else jnp.broadcast_to(value, (chains, *value.shape))
            for value, mapped in zip(args, batched, strict=True)
        )
        values = tuple(
            jnp.swapaxes(value, 0, 1) if time_axis else value
            for value, time_axis in zip(values, timed, strict=True)
        )
        return dispatch(*values), (True, True)

    result = call(
        means,
        points,
        whiteners,
        normalizers,
        weights,
        first_scores,
        scores,
        first_observations,
        observations,
        gumbels,
        last,
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
    """Pay all crossing windows and aGRAD's auxiliary factors before reusing a suffix.

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
            jax.vmap(lambda k: particle_gumbel(k, (count,), dtype=states.dtype))(
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
    first_scores: Float[Array, "P K"],
    scores: Float[Array, "S P K"],
    first_observations: Float[Array, "P K"],
    observation_bank: Float[Array, "S Q K"],
    ancestors: Int[Array, "T-1 A"],
    last: Int[Array, ""],
    earlier_key: PRNGKeyArray,
    auxiliary: Float[Array, "T D"],
    variance: Float[Array, "T D"],
    mask: Bool[Array, "T D"],
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    reference_dtype: DTypeLike,
    unroll: int,
    *,
    point_kernels: PointKernelRoute,
    stream_backward: bool,
) -> tuple[Float[Array, "T D"], Int[Array, " T"], Float[Array, " K"]]:
    """Recover a path with the selector authorized by the compiled memory support."""
    steps, count, _ = bank.shape
    if selector != "backward":
        scores = jnp.concatenate((first_scores[None], scores[: steps - 1]))
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
        if isinstance(point_kernels, CompiledPointKernels) and stream_backward:
            picks, suffix = streamed_backward(
                memory,
                point_kernels,
                memory_bank,
                backward_weights,
                first_scores,
                scores,
                last,
                earlier_key,
            )
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

            gumbels = jax.vmap(lambda k: particle_gumbel(k, (count,), dtype=reference_dtype))(
                jax.random.split(earlier_key, steps - 1)
            )
            coordinates = jnp.asarray(memory.target.state_space.transition, dtype=jnp.int32)

            # Transition coordinates are independent at t>=1, including those with
            # label-specific deterministic initial states. Static projections never
            # change transition coordinates; replay preserves these shared draws.
            points = memory.state(memory_bank[1:])[..., coordinates]
            normalizers = jnp.swapaxes(memory.noise.transitions.log_normalizer[:, 1:], 0, 1)
            picks, suffix = _backward_from_moments(
                backward_weights,
                means,
                points,
                jnp.swapaxes(memory.noise.transitions.whitener[:, 1:], 0, 1),
                normalizers,
                first_scores,
                scores,
                first_observations,
                observation_bank,
                gumbels,
                last,
            )
        initial_selected = picks[0]
        # The backward suffix already contains every selected exact observation and
        # transition factor. Only the initial density remains; do not rescore the path.
        trajectory_scores = suffix + memory.initial_log_probs(memory_bank[0, initial_selected])
        path = bank[jnp.arange(steps), picks]
    return path, picks, cast("jax.Array", trajectory_scores)
