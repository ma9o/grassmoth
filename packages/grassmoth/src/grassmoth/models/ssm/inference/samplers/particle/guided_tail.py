"""Fused diagonal mGRAD mixture normalization, gradient contraction and bank integral."""

from functools import partial
from typing import Annotated, cast

import jax
import jax.numpy as jnp
from pydantic import Field

from grassmoth.models.ssm.inference.samplers.particle.guided_random import (
    KillingNoise,
    TailRandomKeys,
    _gumbel,
    _threefry,
    _uniform_bits,
)
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import (
    _diagonal_integral_terms,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    TRITON_NUM_WARPS,
)

MIXTURE_TAIL_TILE_LIMIT: Annotated[
    int,
    Field(
        description="Heuristic: cap the fused particle-label tile at 2048 entries to bound register use; the measured L40S tile of 128 particles by 8 labels uses 64 registers per thread."
    ),
] = 2048


MIXTURE_TAIL_SMALL_STATE_FACTOR: Annotated[
    int,
    Field(
        description="Measured L40S C10/P128/u8 at M32: doubling the 2048-entry admission limit reduced ordinary sweeps 113.24 to 43.85 ms on SV and 75.64 to 57.39 ms on PARAM. Larger coordinate tiles retain the previous limit."
    ),
] = 2

MIXTURE_TAIL_SMALL_STATE_DIMENSIONS: Annotated[
    int,
    Field(
        description="Admit the wider tile only through the measured M32 PARAM path, D3 padded to four coordinates. Unmeasured above that: the D9 no-spill gates fail in the backward kernel at K16 and K32 with or without this limit, so they say nothing about the wider tile."
    ),
] = 4


MIXTURE_TAIL_DIMENSION_LIMIT: Annotated[
    int,
    Field(
        description="Measured L40S cutover: whole-coordinate fused tail improved D=3 and D=9 native sweeps; the tiled D=40 arm regressed 16.08 to 18.17 ms. Cap padded coordinate width at 16."
    ),
] = 16


TAIL_COORDINATE_TILE: Annotated[
    int,
    Field(
        description="Measured L40S HD D=9 tail: two eight-coordinate CTAs plus reduction/resampling improved 70.33 to 66.78 ms/sweep versus one sixteen-coordinate CTA. D<=8 keeps one fused resampling launch."
    ),
] = 8


TAIL_PARALLEL_COORDINATE_TILE: Annotated[
    int,
    Field(
        description="Paired L40S HD D=9, P=128, K=8, unroll=8: four-coordinate tiles reduced sweep time 58.66 to 54.37 ms at C=10 and 80.61 to 76.97 ms at C=40. Two-coordinate tiles gained slightly more at C=10 but less at C=40. D<=8 retains its single-CTA tile."
    ),
] = 4


def diagonal_mixture_tail(
    x: jax.Array,
    prior: jax.Array,
    chol: jax.Array,
    whitener: jax.Array,
    correction: jax.Array,
    observed: jax.Array,
    derivatives: jax.Array,
    parent_scores: jax.Array,
    transitions: jax.Array,
    label_prior: jax.Array,
    reference_observed: jax.Array,
    reference_gradient: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    killing_noise: KillingNoise | TailRandomKeys,
    index: jax.Array,
    reference_running: jax.Array,
    label_noise: jax.Array,
    *,
    compute_gradient: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fuse exact label sums, the CF24 D.2 bank integral and conditional resampling.

    Model densities and their per-label derivatives remain ordinary JAX code.
    Non-inline draws supply one unused key row: the Pallas CPU interpreter
    requires positive block extents even for an operand the kernel never reads.
    When kappa is zero, no gradient mixture is evaluated: a zero-mass candidate
    therefore keeps its zero score without introducing an undefined 0/0.
    Resampling uses the next time's already-generated independent noise, with
    the original cyclic conditioning on ancestor zero. Its particle payloads
    are gathered by XLA in the following step.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    generate_random = isinstance(killing_noise, TailRandomKeys)
    count, labels = observed.shape
    dimension = x.shape[-1]
    coordinate_block = (
        pl.next_power_of_2(dimension)
        if dimension <= TAIL_COORDINATE_TILE
        else TAIL_PARALLEL_COORDINATE_TILE
    )
    tiles = pl.cdiv(dimension, coordinate_block)
    block_dtype = observed.dtype
    block = pl.next_power_of_2(count)
    label_block = pl.next_power_of_2(labels)

    def finish(
        weights,
        scores_out,
        label_running_out,
        label_ref,
        reference_running_ref,
        survival_ref,
        replacement_ref,
        rotation_ref,
        index_ref,
        label_noise_ref,
        random_keys_ref,
        weights_out,
        ancestors_out,
        cumulative_out,
        labels_out,
        parent_scores_out,
        correction_out,
        *,
        interpret,
    ):
        index = index_ref[()]
        noise_index = jnp.minimum(index, survival_ref.shape[0] - 1)
        p = jnp.arange(block)
        k = jnp.arange(label_block)
        valid = (p[:, None] < count) & (k[None, :] < labels)

        def draw(component, counters):
            row = jnp.minimum(index, random_keys_ref.shape[0] - 1)
            key0 = random_keys_ref[row, component, 0]
            key1 = random_keys_ref[row, component, 1]
            counters = counters.astype(jnp.uint32)
            high, low = _threefry(key0, key1, jnp.zeros_like(counters), counters)
            return _uniform_bits(high, low)

        pt.store(weights_out.at[p], weights, mask=p < count)
        values = jnp.where(p < count, weights, -jnp.inf)
        maximum = jnp.max(values)
        survival = jnp.exp(values - maximum)
        cumulative = jnp.cumsum(survival / jnp.sum(survival))
        rotation_mass = jnp.where((p > 0) & (p < count), (1.0 - survival) / count, 0.0)
        rotation_mass = jnp.where(p == 0, 1.0 - jnp.sum(rotation_mass), rotation_mass)
        rotation_noise = (
            jnp.where(p < count, _gumbel(draw(2, p)), -jnp.inf)
            if generate_random and not interpret
            else pt.load(rotation_ref.at[noise_index, p], mask=p < count, other=-jnp.inf)
        )
        rotation_logits = jnp.log(rotation_mass) + rotation_noise
        rotation = jnp.min(jnp.where(rotation_logits == jnp.max(rotation_logits), p, block))
        source = (p + rotation) % count
        pt.store(cumulative_out.at[p], cumulative, mask=p < count)
        if not interpret:
            pt.debug_barrier()
        chosen_logits = pt.load(weights_out.at[source])
        survival_noise = (
            draw(0, source)
            if generate_random and not interpret
            else pt.load(survival_ref.at[noise_index, source])
        )
        replacement_noise = (
            draw(1, source)
            if generate_random and not interpret
            else pt.load(replacement_ref.at[noise_index, source])
        )

        def search_step(_, bounds):
            lower, upper = bounds
            middle = (lower + upper) // 2
            value = pt.load(cumulative_out.at[jnp.minimum(middle, count - 1)])
            right = (middle < count) & (value <= replacement_noise)
            return jnp.where(right, middle + 1, lower), jnp.where(right, upper, middle)

        lower, _ = jax.lax.fori_loop(
            0,
            count.bit_length(),
            search_step,
            (jnp.zeros((block,), dtype=jnp.int32), jnp.full((block,), count, dtype=jnp.int32)),
        )
        replacements = jnp.minimum(lower, count - 1)
        ancestors = jnp.where(
            survival_noise < jnp.exp(chosen_logits - maximum), source, replacements
        )
        ancestors = jnp.where(p == 0, 0, ancestors).astype(jnp.int32)
        pt.store(ancestors_out.at[p], ancestors, mask=p < count)
        running = pt.load(
            label_running_out.at[ancestors[:, None], k[None, :]], mask=valid, other=0.0
        )
        reference_running = pt.load(reference_running_ref.at[k], mask=k < labels, other=0.0)
        running = jnp.where(p[:, None] == 0, reference_running[None, :], running)
        total = jnp.sum(jnp.where(k[None, :] == labels - 1, running, 0.0), axis=1)
        uniform = (
            jnp.where(p < count, draw(3, p), 0.0)
            if generate_random and not interpret
            else pt.load(label_noise_ref.at[noise_index, p], mask=p < count, other=0.0)
        )
        label = jnp.sum((running <= (uniform * total)[:, None]) & (k[None, :] < labels - 1), axis=1)
        label = jnp.where(total > 0, label, 0).astype(jnp.int32)
        pt.store(labels_out.at[p], label, mask=p < count)
        parent_law = pt.load(scores_out.at[ancestors[:, None], k[None, :]], mask=valid, other=0.0)
        pt.store(parent_scores_out.at[p[:, None], k[None, :]], parent_law, mask=valid)
        selected_score = jnp.sum(jnp.where(k[None, :] == label[:, None], parent_law, 0.0), axis=1)
        prior_score = pt.load(label_ref.at[label], mask=p < count, other=0.0)
        pt.store(correction_out.at[p], prior_score - selected_score, mask=p < count)

    def kernel(
        x_ref,
        prior_ref,
        chol_ref,
        white_ref,
        corr_ref,
        observed_ref,
        derivative_ref,
        parent_ref,
        transition_ref,
        label_ref,
        reference_obs_ref,
        reference_grad_ref,
        variance_ref,
        mask_ref,
        survival_ref,
        replacement_ref,
        rotation_ref,
        index_ref,
        reference_running_ref,
        label_noise_ref,
        random_keys_ref,
        weights_out,
        scores_out,
        observations_out,
        ancestors_out,
        cumulative_out,
        label_running_out,
        labels_out,
        parent_scores_out,
        correction_out,
        *,
        interpret,
    ):
        index = index_ref[()]
        p = jnp.arange(block)
        k = jnp.arange(label_block)
        valid = (p[:, None] < count) & (k[None, :] < labels)
        obs = pt.load(observed_ref.at[p[:, None], k[None, :]], mask=valid, other=-jnp.inf)
        ref_obs = pt.load(reference_obs_ref.at[index, k], mask=k < labels, other=-jnp.inf)
        obs = jnp.where(p[:, None] == 0, ref_obs[None, :], obs)
        if compute_gradient:
            prior_labels = pt.load(label_ref.at[k], mask=k < labels, other=-jnp.inf)
            obs_logits = obs + prior_labels[None, :]
            maximum = jnp.max(obs_logits, axis=1)
            shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            mass = jnp.exp(obs_logits - shift[:, None])
            probability = mass / jnp.sum(mass, axis=1)[:, None]
        else:
            probability = jnp.zeros_like(obs)
        parent = pt.load(parent_ref.at[p[:, None], k[None, :]], mask=valid, other=-jnp.inf)
        transition = pt.load(transition_ref.at[p[:, None], k[None, :]], mask=valid, other=0.0)
        scores = parent + transition + obs
        maximum = jnp.max(scores, axis=1)
        shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        prefix_mass = jnp.exp(scores - shift[:, None])
        running_mass = jnp.cumsum(prefix_mass, axis=1)
        increment = jnp.log(jnp.sum(prefix_mass, axis=1)) + shift
        pt.store(
            label_running_out.at[p[:, None], k[None, :]],
            running_mass,
            mask=valid & (pl.program_id(0) == 0),
        )
        normalized = scores - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
        pt.store(
            scores_out.at[p[:, None], k[None, :]], normalized, mask=valid & (pl.program_id(0) == 0)
        )
        pt.store(
            observations_out.at[p[:, None], k[None, :]], obs, mask=valid & (pl.program_id(0) == 0)
        )

        def coordinate_step(tile, total):
            d = tile * coordinate_block + jnp.arange(coordinate_block)
            coordinate_valid = (p[:, None] < count) & (d[None, :] < dimension)
            if compute_gradient:
                derivative = pt.load(
                    derivative_ref.at[p[:, None, None], k[None, :, None], d[None, None, :]],
                    mask=valid[:, :, None] & (d[None, None, :] < dimension),
                    other=0.0,
                )
                gradient = jnp.sum(probability[:, :, None] * derivative, axis=1)
                reference_gradient = pt.load(
                    reference_grad_ref.at[index, d], mask=d < dimension, other=0.0
                )
                gradient = jnp.where(p[:, None] == 0, reference_gradient[None, :], gradient)
            else:
                gradient = jnp.zeros((block, coordinate_block), dtype=block_dtype)
            x = pt.load(x_ref.at[p[:, None], d[None, :]], mask=coordinate_valid, other=0.0)
            prior = pt.load(prior_ref.at[p[:, None], d[None, :]], mask=coordinate_valid, other=0.0)
            chol = pt.load(chol_ref.at[p[:, None], d[None, :]], mask=coordinate_valid, other=1.0)
            white = pt.load(white_ref.at[p[:, None], d[None, :]], mask=coordinate_valid, other=1.0)
            variance = pt.load(variance_ref.at[index, d], mask=d < dimension, other=1.0)
            free = pt.load(mask_ref.at[index, d], mask=d < dimension, other=False)
            return total + jnp.sum(
                _diagonal_integral_terms(
                    x, prior, chol, white, gradient, variance, free, coordinate_valid, count
                ),
                axis=1,
            )

        initial = jnp.zeros((block,), dtype=block_dtype)
        terms = coordinate_step(pl.program_id(0), initial)
        correction = pt.load(corr_ref.at[p], mask=p < count, other=0.0)
        if tiles == 1:
            integral = correction + terms
            finish(
                increment + integral,
                scores_out,
                label_running_out,
                label_ref,
                reference_running_ref,
                survival_ref,
                replacement_ref,
                rotation_ref,
                index_ref,
                label_noise_ref,
                random_keys_ref,
                weights_out,
                ancestors_out,
                cumulative_out,
                labels_out,
                parent_scores_out,
                correction_out,
                interpret=interpret,
            )

        else:
            partial = terms + jnp.where(pl.program_id(0) == 0, increment + correction, 0.0)
            pt.store(weights_out.at[pl.program_id(0), p], partial, mask=p < count)

    def coordinate_kernel(
        x_ref,
        prior_ref,
        chol_ref,
        white_ref,
        corr_ref,
        observed_ref,
        derivative_ref,
        parent_ref,
        transition_ref,
        label_ref,
        reference_obs_ref,
        reference_grad_ref,
        variance_ref,
        mask_ref,
        survival_ref,
        replacement_ref,
        rotation_ref,
        index_ref,
        reference_running_ref,
        label_noise_ref,
        random_keys_ref,
        weights_out,
        scores_out,
        observations_out,
        label_running_out,
        *,
        interpret,
    ):
        kernel(
            x_ref,
            prior_ref,
            chol_ref,
            white_ref,
            corr_ref,
            observed_ref,
            derivative_ref,
            parent_ref,
            transition_ref,
            label_ref,
            reference_obs_ref,
            reference_grad_ref,
            variance_ref,
            mask_ref,
            survival_ref,
            replacement_ref,
            rotation_ref,
            index_ref,
            reference_running_ref,
            label_noise_ref,
            random_keys_ref,
            weights_out,
            scores_out,
            observations_out,
            None,
            None,
            label_running_out,
            None,
            None,
            None,
            interpret=interpret,
        )

    def run(*args, interpret):
        if generate_random and not interpret:
            empty = jnp.zeros((1, count), dtype=x.dtype)
            args = (*args[:14], empty, empty, empty, *args[17:19], empty, args[20])
        weight_shape = (count,) if tiles == 1 else (tiles, count)
        common_outputs = (
            jax.ShapeDtypeStruct(weight_shape, x.dtype),
            jax.ShapeDtypeStruct((count, labels), x.dtype),
            jax.ShapeDtypeStruct((count, labels), x.dtype),
        )
        running_output = jax.ShapeDtypeStruct((count, labels), x.dtype)
        if tiles == 1:
            outputs = (
                *common_outputs,
                jax.ShapeDtypeStruct((count,), jnp.int32),
                jax.ShapeDtypeStruct((count,), x.dtype),
                running_output,
                jax.ShapeDtypeStruct((count,), jnp.int32),
                jax.ShapeDtypeStruct((count, labels), x.dtype),
                jax.ShapeDtypeStruct((count,), x.dtype),
            )
            body = partial(kernel, interpret=interpret)
        else:
            outputs = (*common_outputs, running_output)

            body = partial(coordinate_kernel, interpret=interpret)
        return pl.pallas_call(
            body,
            out_shape=outputs,
            grid=(tiles,),
            interpret=interpret,
            compiler_params=pt.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="diagonal_mgrad_mixture_tail_killing"
            if tiles == 1
            else "diagonal_mgrad_mixture_tail_coordinates",
        )(*args)

    noise_arrays = (
        killing_noise.reference if isinstance(killing_noise, TailRandomKeys) else killing_noise
    )
    label_noise = (
        killing_noise.label_reference if isinstance(killing_noise, TailRandomKeys) else label_noise
    )
    random_keys = (
        killing_noise.values
        if isinstance(killing_noise, TailRandomKeys)
        else jnp.zeros((1, 4, 2), dtype=jnp.uint32)
    )
    result = jax.lax.platform_dependent(
        x,
        prior,
        chol,
        whitener,
        correction,
        observed,
        derivatives,
        parent_scores,
        transitions,
        label_prior,
        reference_observed,
        reference_gradient,
        variance,
        mask,
        *noise_arrays,
        index,
        reference_running,
        label_noise,
        random_keys,
        cpu=partial(run, interpret=True),
        cuda=partial(run, interpret=False),
    )
    if tiles == 1:
        (
            weights,
            scores,
            observations,
            ancestors,
            _,
            running_mass,
            next_labels,
            next_parent_scores,
            next_correction,
        ) = result
    else:
        partials, scores, observations, running_mass = result

        def finish_kernel(
            partials_ref,
            scores_ref,
            running_ref,
            labels_ref,
            reference_running_ref,
            survival_ref,
            replacement_ref,
            rotation_ref,
            index_ref,
            label_noise_ref,
            random_keys_ref,
            weights_out,
            ancestors_out,
            cumulative_out,
            labels_out,
            parent_out,
            correction_out,
            *,
            interpret,
        ):
            q = jnp.arange(pl.next_power_of_2(tiles))
            p = jnp.arange(block)
            partial = pt.load(
                partials_ref.at[q[:, None], p[None, :]],
                mask=(q[:, None] < tiles) & (p[None, :] < count),
                other=0.0,
            )
            finish(
                jnp.sum(partial, axis=0),
                scores_ref,
                running_ref,
                labels_ref,
                reference_running_ref,
                survival_ref,
                replacement_ref,
                rotation_ref,
                index_ref,
                label_noise_ref,
                random_keys_ref,
                weights_out,
                ancestors_out,
                cumulative_out,
                labels_out,
                parent_out,
                correction_out,
                interpret=interpret,
            )

        def finish_run(*args, interpret):
            if generate_random and not interpret:
                empty = jnp.zeros((1, count), dtype=x.dtype)
                args = (*args[:5], empty, empty, empty, args[8], empty, args[10])
            return pl.pallas_call(
                partial(finish_kernel, interpret=interpret),
                out_shape=(
                    jax.ShapeDtypeStruct((count,), x.dtype),
                    jax.ShapeDtypeStruct((count,), jnp.int32),
                    jax.ShapeDtypeStruct((count,), x.dtype),
                    jax.ShapeDtypeStruct((count,), jnp.int32),
                    jax.ShapeDtypeStruct((count, labels), x.dtype),
                    jax.ShapeDtypeStruct((count,), x.dtype),
                ),
                grid=(1,),
                interpret=interpret,
                compiler_params=pt.CompilerParams(num_warps=TRITON_NUM_WARPS),
                name="diagonal_mgrad_mixture_finish",
            )(*args)

        weights, ancestors, _, next_labels, next_parent_scores, next_correction = (
            jax.lax.platform_dependent(
                partials,
                scores,
                running_mass,
                label_prior,
                reference_running,
                *noise_arrays,
                index,
                label_noise,
                random_keys,
                cpu=partial(finish_run, interpret=True),
                cuda=partial(finish_run, interpret=False),
            )
        )

    return cast(
        "tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]",
        (
            weights,
            scores,
            observations,
            ancestors,
            running_mass,
            next_labels,
            next_parent_scores,
            next_correction,
        ),
    )
