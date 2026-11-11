"""Persistent exact diagonal CF24 filter over internally padded state coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
        _DiagonalProposalGeometry,
    )
    from grassmoth.models.ssm.inference.samplers.particle.point_program import CompiledPointKernels
    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.guided_random import (
    _threefry,
    _uniform_bits,
    conditional_rotation,
)
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import (
    _diagonal_integral_terms,
)
from grassmoth.models.ssm.inference.samplers.particle.persistent_launch import persistent_warps


def correct_label(prior: jax.Array, gathered: jax.Array, _ungathered: jax.Array) -> jax.Array:
    """Auxiliary-label importance correction; exposed to the fault gate."""
    return prior - gathered


def persistent_forward(
    memory: SequentialObservationRuntime,
    programs: CompiledPointKernels,
    geometry: _DiagonalProposalGeometry,
    reference: jax.Array,
    reference_observed: jax.Array,
    gradients: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    label_prior: jax.Array,
    reference_running: jax.Array,
    first: jax.Array,
    first_parent: jax.Array,
    first_ancestors: jax.Array,
    first_labels: jax.Array,
    first_correction: jax.Array,
    proposal_keys: jax.Array,
    resampling_keys: jax.Array,
    *,
    kappa: bool,
    record_trace: bool,
    keep_observations: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, None, jax.Array, jax.Array, jax.Array]:
    """Preserve conditional killing and the CF24 D.2 per-coordinate bank integral.

    The cyclic conditioning uses the shared rotation law. Gather the parent label
    law before correcting by log(prior)-log(proposal). Noise flattens true D, not
    padded D, to preserve the original Threefry stream. Inverse-erf constants are
    sqrt(2) and the nearest float32 to -1 from above; they prevent infinite draws.
    Paired L40S SV runs measured the scalar N-by-1 noise layout at 23.467 ms per
    sweep versus 24.120 ms for coordinate-vector assembly at N256. N2048 stayed
    at 79.185 versus 79.179 ms. Other dimensions draw only stochastic coordinates.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    steps, dimension = reference.shape
    count, labels = first.shape[0], first_parent.shape[-1]
    coordinate_block = pl.next_power_of_2(dimension)
    transition = programs.transition
    observation = programs.forward_observation
    split = len(transition.buffers)
    whites = jnp.swapaxes(memory.noise.transitions.whitener, 0, 1)
    normalizers = jnp.swapaxes(memory.noise.transitions.log_normalizer, 0, 1)
    dtype = first.dtype

    def kernel(
        first_ref,
        parent_ref,
        anc_ref,
        lab_ref,
        correction_ref,
        map_ref,
        shift_ref,
        chol_ref,
        white_ref,
        reference_ref,
        obs_ref,
        gradient_ref,
        variance_ref,
        mask_ref,
        prior_ref,
        running_ref,
        normal_keys_ref,
        keys_ref,
        transition_white_ref,
        transition_norm_ref,
        *rest,
        interpret,
    ):
        invariant_refs, output = rest[:-8], rest[-8:]
        bank, weights, scores, ancestors_history, centers, observations, cumulative, running = (
            output
        )
        tr_refs, ob_refs = invariant_refs[:split], invariant_refs[split:]
        chain = pl.program_id(0)
        p, k, r = jnp.arange(count), jnp.arange(labels), jnp.arange(coordinate_block)
        state_valid = r < dimension
        ancestor = pt.load(anc_ref.at[chain, p])
        label = pt.load(lab_ref.at[chain, p])
        parent = pt.load(parent_ref.at[chain, p[:, None], k[None, :]])
        correction = pt.load(correction_ref.at[chain, p])
        label_prior = pt.load(prior_ref.at[chain, k])
        reference_running = pt.load(running_ref.at[chain, k])

        def draw(index, component, counter):
            row = jnp.minimum(index, steps - 2)
            high, low = _threefry(
                keys_ref[chain, row, component, 0],
                keys_ref[chain, row, component, 1],
                jnp.zeros_like(counter, dtype=jnp.uint32),
                counter.astype(jnp.uint32),
            )
            return _uniform_bits(high, low)

        def body(offset, carry):
            ancestor, label, parent, correction = carry
            index = offset + 1
            initial = pt.load(
                first_ref.at[chain, ancestor[:, None], r[None, :]],
                mask=(offset == 0) & state_valid[None, :],
                other=0.0,
            )
            earlier = pt.load(
                bank.at[jnp.maximum(offset - 1, 0), chain, ancestor[:, None], r[None, :]],
                mask=(offset > 0) & state_valid[None, :],
                other=0.0,
            )
            previous = jnp.where(offset == 0, initial, earlier)
            if record_trace:
                pt.store(ancestors_history.at[offset, chain, p], ancestor)
            tr_values = transition.load(tr_refs, index, chain)
            means = jax.vmap(lambda state: transition.evaluate(state, tr_values))(previous)
            means = jnp.reshape(means, (count, labels, coordinate_block))
            free = pt.load(mask_ref.at[index, chain, r], mask=state_valid, other=False)
            mean_map = pt.load(
                map_ref.at[index, chain, k[:, None], r[None, :]],
                mask=state_valid[None, :],
                other=0.0,
            )
            shifts = pt.load(
                shift_ref.at[index, chain, k[:, None], r[None, :]],
                mask=state_valid[None, :],
                other=0.0,
            )
            component_centers = (
                jnp.where(free, means, 0.0) * mean_map[None, :, :] + shifts[None, :, :]
            )
            center = jnp.sum(
                jnp.where(k[None, :, None] == label[:, None, None], component_centers, 0.0), axis=1
            )
            ch = pt.load(
                chol_ref.at[index, chain, label[:, None], r[None, :]],
                mask=state_valid[None, :],
                other=1.0,
            )
            white = pt.load(
                white_ref.at[index, chain, label[:, None], r[None, :]],
                mask=state_valid[None, :],
                other=1.0,
            )
            selected_shift = pt.load(
                shift_ref.at[index, chain, label[:, None], r[None, :]],
                mask=state_valid[None, :],
                other=0.0,
            )
            if dimension == 1 and memory.target.state_space.transition == (0,):
                # Retain the measured scalar N-by-1 layout.
                high, low = _threefry(
                    normal_keys_ref[chain, offset, 0],
                    normal_keys_ref[chain, offset, 1],
                    jnp.zeros((count, coordinate_block), dtype=jnp.uint32),
                    (jnp.maximum(p - 1, 0)[:, None] * dimension + r[None, :]).astype(jnp.uint32),
                )
                uniform = _uniform_bits(high, low)
                normal = jnp.float32(1.4142135623730951) * jax.lax.erf_inv(
                    jnp.maximum(
                        jnp.float32(-0.9999999403953552),
                        uniform * 2.0 + jnp.float32(-0.9999999403953552),
                    )
                )
            else:
                # Preserve original counter strides but draw only stochastic coordinates.
                normal = jnp.zeros((count, coordinate_block), dtype=dtype)
                for coordinate in memory.target.state_space.transition:
                    high, low = _threefry(
                        normal_keys_ref[chain, offset, 0],
                        normal_keys_ref[chain, offset, 1],
                        jnp.zeros((count,), dtype=jnp.uint32),
                        (jnp.maximum(p - 1, 0) * dimension + coordinate).astype(jnp.uint32),
                    )
                    uniform = _uniform_bits(high, low)
                    noise = jnp.float32(1.4142135623730951) * jax.lax.erf_inv(
                        jnp.maximum(
                            jnp.float32(-0.9999999403953552),
                            uniform * 2.0 + jnp.float32(-0.9999999403953552),
                        )
                    )
                    normal = normal + jnp.where(r[None, :] == coordinate, noise[:, None], 0.0)
            reference_state = pt.load(
                reference_ref.at[index, chain, r], mask=state_valid, other=0.0
            )
            current = jnp.where(
                free[None, :] & (p[:, None] > 0), center + ch * normal, reference_state[None, :]
            )
            pt.store(
                bank.at[offset, chain, p[:, None], r[None, :]], current, mask=state_valid[None, :]
            )
            if record_trace:
                pt.store(
                    centers.at[offset, chain, p[:, None], r[None, :]],
                    center,
                    mask=state_valid[None, :],
                )
            obs_values = observation.load(ob_refs, index, chain)
            observed, derivatives = jax.vmap(lambda state: observation.evaluate(state, obs_values))(
                current
            )
            observed = jnp.where(
                p[:, None] == 0, pt.load(obs_ref.at[index, chain, k])[None, :], observed
            )
            if keep_observations:
                pt.store(observations.at[offset, chain, p[:, None], k[None, :]], observed)
            projected = jax.vmap(lambda state: programs.state.evaluate(state, ()))(current)
            norm = jnp.zeros((count, labels), dtype=dtype)
            for noise_coordinate, state_coordinate in enumerate(
                memory.target.state_space.transition
            ):
                point = jnp.sum(jnp.where(r[None, :] == state_coordinate, projected, 0.0), axis=-1)
                mean = jnp.sum(jnp.where(r[None, None, :] == state_coordinate, means, 0.0), axis=-1)
                transition_white = pt.load(
                    transition_white_ref.at[index, chain, k, noise_coordinate]
                )
                norm = norm + ((point[:, None] - mean) * transition_white[None, :]) ** 2
            transition_normalizer = pt.load(transition_norm_ref.at[index, chain, k])
            edges = transition_normalizer[None, :] - 0.5 * norm
            total = parent + edges + observed
            maximum = jnp.max(total, axis=1)
            stable = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            mass = jnp.exp(total - stable[:, None])
            increment = jnp.log(jnp.sum(mass, axis=1)) + stable
            normalized = total - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
            pt.store(scores.at[offset, chain, p[:, None], k[None, :]], normalized)
            pt.store(running.at[chain, p[:, None], k[None, :]], jnp.cumsum(mass, axis=1))
            if kappa:
                logits = observed + label_prior[None, :]
                stable = jnp.max(logits, axis=1)
                probability = jnp.exp(logits - stable[:, None])
                probability = probability / jnp.sum(probability, axis=1)[:, None]
                derivative = jnp.reshape(derivatives, (count, labels, coordinate_block))
                gradient = jnp.sum(probability[:, :, None] * derivative, axis=1)
                ref_gradient = pt.load(
                    gradient_ref.at[index, chain, r], mask=state_valid, other=0.0
                )
                gradient = jnp.where(p[:, None] == 0, ref_gradient[None, :], gradient)
            else:
                gradient = jnp.zeros_like(current)
            local_variance = pt.load(variance_ref.at[index, chain, r], mask=state_valid, other=1.0)
            integral = _diagonal_integral_terms(
                current,
                center - selected_shift,
                ch,
                white,
                gradient,
                local_variance,
                free,
                state_valid[None, :],
                count,
            )
            value = increment + correction + jnp.sum(integral, axis=-1)
            pt.store(weights.at[offset, chain, p], value)
            maximum = jnp.max(value)
            survival = jnp.exp(value - maximum)
            cdf = jnp.cumsum(survival / jnp.sum(survival))
            pt.store(cumulative.at[chain, p], cdf)
            if not interpret:
                pt.debug_barrier()
            selected = jnp.minimum(
                (draw(index, 2, jnp.asarray(0, jnp.uint32)) * count).astype(jnp.int32), count - 1
            )
            rotation = conditional_rotation(
                selected,
                jnp.exp(weights[offset, chain, selected] - maximum),
                draw(index, 2, jnp.asarray(1, jnp.uint32)),
            )
            source = (p + rotation) % count
            source_value = pt.load(weights.at[offset, chain, source])
            uniforms = draw(index, 1, source)

            def search(_, bounds):
                low, high = bounds
                mid = (low + high) // 2
                cv = pt.load(cumulative.at[chain, jnp.minimum(mid, count - 1)])
                right = (mid < count) & (cv <= uniforms)
                return jnp.where(right, mid + 1, low), jnp.where(right, high, mid)

            low, _ = jax.lax.fori_loop(
                0,
                count.bit_length(),
                search,
                (jnp.zeros((count,), jnp.int32), jnp.full((count,), count, jnp.int32)),
            )
            ancestor = jnp.where(
                draw(index, 0, source) < jnp.exp(source_value - maximum),
                source,
                jnp.minimum(low, count - 1),
            )
            ancestor = jnp.where(p == 0, 0, ancestor).astype(jnp.int32)
            masses = pt.load(running.at[chain, ancestor[:, None], k[None, :]])
            masses = jnp.where(p[:, None] == 0, reference_running[None, :], masses)
            total = jnp.sum(jnp.where(k[None, :] == labels - 1, masses, 0.0), axis=1)
            label = jnp.sum(
                (masses <= (draw(index, 3, p) * total)[:, None]) & (k[None, :] < labels - 1), axis=1
            )
            label = jnp.where(total > 0, label, 0).astype(jnp.int32)
            parent = pt.load(scores.at[offset, chain, ancestor[:, None], k[None, :]])
            selected = jnp.sum(jnp.where(k[None, :] == label[:, None], parent, 0.0), axis=1)
            ungathered = jnp.sum(jnp.where(k[None, :] == label[:, None], normalized, 0.0), axis=1)
            prior = pt.load(prior_ref.at[chain, label])
            correction = correct_label(prior, selected, ungathered)
            if not interpret:
                pt.debug_barrier()
            return ancestor, label, parent, correction

        jax.lax.fori_loop(0, steps - 1, body, (ancestor, label, parent, correction))

    arguments = (
        first,
        first_parent,
        first_ancestors,
        first_labels,
        first_correction,
        geometry.mean_map,
        geometry.shift,
        geometry.chol,
        geometry.whiten,
        reference,
        reference_observed,
        gradients,
        variance,
        mask,
        label_prior,
        reference_running,
        jax.random.key_data(proposal_keys),
        resampling_keys,
        whites,
        normalizers,
        *transition.buffers,
        *observation.buffers,
    )
    timed = (
        (False,) * 5 + (True,) * 9 + (False,) * 4 + (True, True) + (True,) * (len(arguments) - 20)
    )

    def run(*args, interpret):
        chains = args[0].shape[0]

        def shape(dims: tuple[int, ...], kind: DTypeLike = dtype) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(dims, kind)

        return pl.pallas_call(
            partial(kernel, interpret=interpret),
            grid=(chains,),
            interpret=interpret,
            compiler_params=pt.CompilerParams(
                num_warps=persistent_warps(count, labels, dimension, "forward")
            ),
            out_shape=(
                shape((steps - 1, chains, count, dimension)),
                shape((steps - 1, chains, count)),
                shape((steps - 1, chains, count, labels)),
                shape((steps - 1, chains, count if record_trace else 1), jnp.int32),
                shape((steps - 1, chains, count if record_trace else 1, dimension)),
                shape((steps - 1, chains, count if keep_observations else 1, labels)),
                shape((chains, count)),
                shape((chains, count, labels)),
            ),
            name="persistent_exact_forward",
        )(*args)[:6]

    def dispatch(*args):
        return jax.lax.platform_dependent(
            *args, cpu=partial(run, interpret=True), cuda=partial(run, interpret=False)
        )

    @jax.custom_batching.custom_vmap
    def call(*args):
        values = tuple(jnp.expand_dims(a, 1 if t else 0) for a, t in zip(args, timed, strict=True))
        return tuple(jnp.squeeze(a, axis=1) for a in dispatch(*values))

    @call.def_vmap
    def batch(chains, mapped, *args):
        values = tuple(
            a if m else jnp.broadcast_to(a, (chains, *a.shape))
            for a, m in zip(args, mapped, strict=True)
        )
        values = tuple(
            jnp.swapaxes(a, 0, 1) if t else a for a, t in zip(values, timed, strict=True)
        )
        return tuple(jnp.swapaxes(a, 0, 1) for a in dispatch(*values)), (True,) * 6

    tail, weights, scores, ancestors, centers, observed = call(*arguments)
    observed = (
        observed[..., None]
        if keep_observations
        else jnp.empty((steps - 1, 0, labels, 1), dtype=dtype)
    )
    if not record_trace:
        ancestors = jnp.empty((steps - 1, 0), dtype=jnp.int32)
        centers = jnp.empty((steps - 1, 0, dimension), dtype=dtype)
    return cast(
        "tuple[jax.Array, jax.Array, jax.Array, None, jax.Array, jax.Array, jax.Array]",
        (tail, weights, scores, None, ancestors, centers, observed),
    )
