"""Persistent exact backward selection with native expressions and on-demand RNG."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.samplers.particle.point_program import (
        CompiledPointKernels,
        CompiledPointProgram,
    )
    from grassmoth.models.ssm.inference.samplers.particle.scalar_program import ScalarTensor
    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.guided_random import (
    _gumbel,
    _threefry,
    _uniform_bits,
)
from grassmoth.models.ssm.inference.samplers.particle.persistent_launch import persistent_warps


def observe_point(
    program: CompiledPointProgram[jax.Array], state: jax.Array, values: tuple[ScalarTensor, ...]
) -> jax.Array:
    """Evaluate the original label-dependent observation expression."""
    return program.evaluate(state, values)


def streamed_backward(
    memory: SequentialObservationRuntime,
    programs: CompiledPointKernels,
    bank: jax.Array,
    weights: jax.Array,
    first_scores: jax.Array,
    scores: jax.Array,
    last: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Retain particles and prefix laws; recompute true means/emissions when read."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    steps, count, width = bank.shape
    labels = first_scores.shape[-1]
    dtype = bank.dtype
    observation = programs.observation
    transition = programs.transition
    obs_count = len(observation.buffers)
    coordinates = tuple(memory.target.state_space.transition)
    point_block = pl.next_power_of_2(count)
    label_block = pl.next_power_of_2(labels)
    width_block = pl.next_power_of_2(width)
    keys = jax.random.key_data(jax.random.split(key, max(steps - 1, 1)))
    whites = jnp.swapaxes(memory.noise.transitions.whitener, 0, 1)
    normalizers = jnp.swapaxes(memory.noise.transitions.log_normalizer, 0, 1)

    def kernel(
        bank_ref, weights_ref, first_ref, scores_ref, white_ref, norm_ref, keys_ref, last_ref, *rest
    ):
        invariant_refs, picks_ref, suffix_ref = rest[:-2], rest[-2], rest[-1]
        obs_refs, transition_refs = invariant_refs[:obs_count], invariant_refs[obs_count:]
        chain = pl.program_id(0)
        p, k, r = jnp.arange(point_block), jnp.arange(label_block), jnp.arange(width_block)
        following = last_ref[chain]
        selected_state = pt.load(
            bank_ref.at[steps - 1, chain, following, r], mask=r < width, other=0.0
        )
        suffix = observe_point(
            observation, selected_state, observation.load(obs_refs, steps - 1, chain)
        )
        picks_ref[chain, steps - 1] = following

        def body(offset, carry):
            following, suffix = carry
            index = steps - 2 - offset
            states = pt.load(
                bank_ref.at[index, chain, p[:, None], r[None, :]],
                mask=(p[:, None] < count) & (r[None, :] < width),
                other=0.0,
            )
            constants = transition.load(transition_refs, index + 1, chain)
            means = jax.vmap(lambda state: transition.evaluate(state, constants))(states)
            following_memory = pt.load(
                bank_ref.at[index + 1, chain, following, r], mask=r < width, other=0.0
            )
            following_state = programs.state.evaluate(following_memory, ())
            state_axis = jnp.arange(following_state.shape[0])
            normalizer = pt.load(norm_ref.at[index + 1, chain, k])
            norm = jnp.zeros((point_block, label_block), dtype=dtype)
            for noise_coordinate, state_coordinate in enumerate(coordinates):
                point = jnp.sum(jnp.where(state_axis == state_coordinate, following_state, 0.0))
                mean = jnp.sum(
                    jnp.where(state_axis[None, None, :] == state_coordinate, means, 0.0), axis=-1
                )
                white = pt.load(white_ref.at[index + 1, chain, k, noise_coordinate])
                norm = norm + ((point - mean) * white[None, :]) ** 2
            edges = normalizer[None] - 0.5 * norm
            first = pt.load(
                first_ref.at[chain, p[:, None], k[None, :]],
                mask=(p[:, None] < count) & (index == 0),
                other=-jnp.inf,
            )
            tail = pt.load(
                scores_ref.at[jnp.maximum(index - 1, 0), chain, p[:, None], k[None, :]],
                mask=(p[:, None] < count) & (index > 0),
                other=-jnp.inf,
            )
            total = jnp.where(index == 0, first, tail) + edges + suffix[None]
            maximum = jnp.max(total, axis=1)
            shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            mixed = jnp.log(jnp.sum(jnp.exp(total - shift[:, None]), axis=1)) + shift
            weight = pt.load(weights_ref.at[index, chain, p], mask=p < count, other=-jnp.inf)
            key0, key1 = keys_ref[chain, offset, 0], keys_ref[chain, offset, 1]
            high, low = _threefry(
                key0, key1, jnp.zeros_like(p, dtype=jnp.uint32), p.astype(jnp.uint32)
            )
            noise = _gumbel(_uniform_bits(high, low))
            logits = mixed + weight + noise
            pick = jnp.min(jnp.where(logits == jnp.max(logits), p, point_block))
            edge = jnp.sum(jnp.where(p[:, None] == pick, edges, 0.0), axis=0)
            selected_state = pt.load(bank_ref.at[index, chain, pick, r], mask=r < width, other=0.0)
            observed = observe_point(
                observation, selected_state, observation.load(obs_refs, index, chain)
            )
            picks_ref[chain, index] = pick
            return pick, suffix + edge + observed

        _, suffix = jax.lax.fori_loop(0, steps - 1, body, (following, suffix))
        pt.store(suffix_ref.at[chain, k], suffix)

    arguments = (
        bank,
        weights,
        first_scores,
        scores,
        whites,
        normalizers,
        keys,
        last,
        *observation.buffers,
        *transition.buffers,
    )
    time_axes = (
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        *((True,) * (len(arguments) - 8)),
    )

    def run(*args, interpret):
        chains = args[7].shape[0]
        # At T=1 the unused predecessor history has zero logical rows.
        args = tuple(
            value if value.size else jnp.zeros(tuple(max(1, n) for n in value.shape), value.dtype)
            for value in args
        )
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((chains, steps), jnp.int32),
                jax.ShapeDtypeStruct((chains, labels), dtype),
            ),
            grid=(chains,),
            interpret=interpret,
            compiler_params=pt.CompilerParams(
                num_warps=persistent_warps(
                    count, labels, memory.target.state_space.size, "backward"
                )
            ),
            name="streamed_exact_backward",
        )(*args)

    def dispatch(*args):
        return jax.lax.platform_dependent(
            *args, cpu=partial(run, interpret=True), cuda=partial(run, interpret=False)
        )

    @jax.custom_batching.custom_vmap
    def call(*args):
        expanded = tuple(
            jnp.expand_dims(a, 1 if timed else 0) for a, timed in zip(args, time_axes, strict=True)
        )
        return tuple(a[0] for a in dispatch(*expanded))

    @call.def_vmap
    def batch(chains, batched, *args):
        values = tuple(
            a if mapped else jnp.broadcast_to(a, (chains, *a.shape))
            for a, mapped in zip(args, batched, strict=True)
        )
        values = tuple(
            jnp.swapaxes(a, 0, 1) if timed else a
            for a, timed in zip(values, time_axes, strict=True)
        )
        return dispatch(*values), (True, True)

    return cast("tuple[jax.Array, jax.Array]", call(*arguments))
