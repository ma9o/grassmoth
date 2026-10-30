"""Conditional divide-and-conquer SMC for the exact auxiliary trajectory target.

The conditional tree is Algorithm 4 of https://arxiv.org/abs/2202.02264.
Auxiliary factors are owned by the smallest node containing their pair (aMALA)
or triple (aMALA+); interval emissions follow their compiled ownership plan.
Parameter labels remain marginalized throughout the tree.

Every leaf first pays a positive Gaussian auxiliary guide. When a pair or triple
becomes available, its exact local auxiliary factor replaces that guide. These
ratios telescope, leaving the final augmented target unchanged while preventing
the inverse proposal density from dominating the intermediate leaf weights.

For nonlocal corrections, the root first conditionally resamples N trajectories
from its local target, retaining the reference. Importance selection then replaces
that target by the exact replayed scientific target. This is a final
conditional SMC stage: its conditional particle-generation law is unchanged,
and its terminal weights are the exact target/local-target ratio. Consequently
only N complete trajectories need replay, instead of N squared.
The terminal forced move is the modified Gibbs update in Particle-aMALA Algorithm
1 (https://arxiv.org/abs/2401.14868); its stationary law is the terminal weights.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
from grassmoth.models.ssm.inference.samplers.particle.auxiliary import build_auxiliary
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    _normalize_log_probs,
    _observation_log_probs_by_param,
    _value_and_grad_by_param,
    coordinate_proposal_mask,
    multinomial,
)
from grassmoth.models.ssm.inference.samplers.particle.weights import (
    requested_weight_statistics,
    weight_history_fields,
)
from grassmoth.models.ssm.inference.targets.intervals import (
    IntervalTreeLevelRuntime,
    gather_interval_statistics,
    stack_interval_tree_levels,
)
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    SmootherTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from grassmoth.models.ssm.compile.interval_tree import CompiledIntervalTreeLevel


# Bound each label-specific pair-score array to 4 GiB per chain in float32.
# Keep broad node concurrency when memory permits; the sweep planner verifies
# actual executable memory and can request smaller batches.
_PAIRWISE_BATCH_ELEMENTS = 1 << 30


class _Segment(NamedTuple):
    first: jax.Array
    last: jax.Array
    evidence: jax.Array
    prior_log_prob: jax.Array
    boundary: jax.Array
    weights: jax.Array
    statistics: jax.Array


type _CombinedSegment = tuple[
    _Segment, dict[ParticleWeightMetric, jax.Array], tuple[jax.Array, jax.Array]
]


def _forced_move(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Modified Gibbs selection starting at the retained reference (index zero)."""
    proposal_key, accept_key = jax.random.split(key)
    indices = jnp.arange(logits.size)
    other = jnp.where(indices == 0, -jnp.inf, logits)
    other_mass = jax.scipy.special.logsumexp(other)

    def move() -> jax.Array:
        proposed = multinomial(proposal_key, other, 1)[0]
        reverse_mass = jax.scipy.special.logsumexp(jnp.where(indices == proposed, -jnp.inf, logits))
        log_acceptance = jnp.minimum(0.0, other_mass - reverse_mass)
        accepted = jnp.log(jax.random.uniform(accept_key, dtype=logits.dtype)) < log_acceptance
        return jnp.where(accepted, proposed, 0).astype(jnp.int32)

    # A target supported only on the reference has no off-diagonal proposal.
    selected: jax.Array = jax.lax.cond(
        jnp.isneginf(other_mass), lambda: jnp.asarray(0, jnp.int32), move
    )
    return selected


def _step(
    ctx: SmootherContext,
    key: jax.Array,
    x_ref: jax.Array,
    root_correction: Callable[[jax.Array, jax.Array], jax.Array] | None,
    *,
    record_trace: bool = False,
    batch_size: int | None = None,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, SmootherTraceRecord]:
    # Reuse traced density and gradient bodies across leaf, seam, and root batches.
    # Inline their JAX graphs so sweep closure conversion can lift captured
    # observations and prior arrays too; the Python traces are still reused.
    ctx = ctx._replace(
        initial_value_grad_by_param=jax.jit(
            ctx.initial_value_grad_by_param, inline=jax.Inline.JAX_EARLY
        ),
        transition_current_value_grad_by_param=jax.jit(
            ctx.transition_current_value_grad_by_param, inline=jax.Inline.JAX_EARLY
        ),
        transition_next_value_grad_by_param=jax.jit(
            ctx.transition_next_value_grad_by_param, inline=jax.Inline.JAX_EARLY
        ),
        pairwise_transition_log_probs=jax.jit(
            ctx.pairwise_transition_log_probs, inline=jax.Inline.JAX_EARLY
        ),
        obs_increment_fn=jax.jit(ctx.obs_increment_fn, inline=jax.Inline.JAX_EARLY),
    )

    def measure_weights(logits: jax.Array) -> dict[ParticleWeightMetric, jax.Array]:
        return requested_weight_statistics(logits, enabled=ctx.diagnostic_weights)

    count = ctx.num_free_particles + 1
    steps = ctx.num_steps
    labels = ctx.num_parameter_particles
    depth = (steps - 1).bit_length()
    padded = 1 << depth
    logpi = ctx.initial_label_log_probs
    intervals = ctx.intervals
    method = ctx.drift
    aux_key, mask_key, leaf_key, tree_key, _root_key, select_key = jax.random.split(key, 6)
    mask = coordinate_proposal_mask(mask_key, ctx.latent_free_mask, ctx.latent_block_coords)
    auxiliary = build_auxiliary(ctx, aux_key, x_ref, mask)
    local_log_density: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array] = jax.jit(
        auxiliary.local_log_density, inline=jax.Inline.JAX_EARLY
    )
    with jax.named_scope("dsmc_leaves"):
        particles, proposal_log_prob = auxiliary.propose(leaf_key)
        # Every tree level gathers from this same bank. Materialize it once so
        # random draws and pilot-mixture arithmetic cannot be cloned into each
        # boundary gather and its differentiated transition graph.
        proposed: tuple[jax.Array, jax.Array] = jax.lax.optimization_barrier(
            (particles, proposal_log_prob)
        )
        particles, proposal_log_prob = proposed

    def observation_scores(index: jax.Array, bank: jax.Array) -> tuple[jax.Array, jax.Array]:
        if method == "none":
            values = _observation_log_probs_by_param(
                ctx.contexts, bank, index, ctx.runtime_observations, ctx.obs_increment_fn
            )
            return values, jnp.zeros((*values.shape, 0), dtype=ctx.latent_dtype)
        return jax.vmap(
            lambda state: _value_and_grad_by_param(
                ctx.contexts,
                state,
                lambda bound, value: ctx.obs_increment_fn(
                    bound, value, index, ctx.runtime_observations
                ),
                value_dtype=ctx.traj_dtype,
                grad_dtype=ctx.latent_dtype,
            )
        )(bank)

    observations, observation_gradients = jax.vmap(observation_scores)(jnp.arange(steps), particles)
    initial, initial_gradient = jax.vmap(ctx.initial_value_grad_by_param)(particles[0])
    leaf_values = observations.at[0].add(initial)
    leaf_gradients = (
        observation_gradients
        if method == "none"
        else observation_gradients.at[0].add(initial_gradient)
    )
    leaf_values, leaf_gradients = jax.lax.optimization_barrier((leaf_values, leaf_gradients))
    paid_auxiliary = jax.vmap(auxiliary.guide_log_density)(particles, jnp.arange(steps)).astype(
        ctx.traj_dtype
    )
    if ctx.drift != "none" and (method == "amala" or steps == 1):
        initial_auxiliary = jax.vmap(
            lambda value: local_log_density(value, value, value, jnp.asarray(0))
        )(particles[0]).astype(ctx.traj_dtype)
        paid_auxiliary = paid_auxiliary.at[0].set(initial_auxiliary)
    evidence = leaf_values - proposal_log_prob[..., None] + paid_auxiliary[..., None]
    # Carry only selected scientific prior factors; observations and auxiliary
    # densities remain in evidence. Targets without replay need no score storage.
    prior = jnp.zeros(
        (steps, count, labels if root_correction is not None else 0), dtype=ctx.traj_dtype
    )
    if root_correction is not None:
        prior = prior.at[0].set(initial)
    statistics = (
        jnp.zeros((padded, count, labels, 0, 0, 1), dtype=ctx.latent_dtype)
        if intervals is None
        else intervals.leaf_statistics(ctx.contexts, particles)
    )
    if intervals is not None and intervals.tree.levels[0].emission_slots.shape[1]:
        evidence = evidence + jax.vmap(
            lambda stats, node: intervals.emission_log_probs(
                ctx.contexts, stats, intervals.tree.levels[0], node
            )
        )(statistics[:steps], jnp.arange(steps))

    def seam_auxiliary(
        left: _Segment, right: _Segment, seam: jax.Array, width: jax.Array
    ) -> jax.Array:
        zeros = jnp.zeros((count, count), dtype=ctx.traj_dtype)
        if ctx.drift == "none":
            # The guide is the actual factor: all auxiliary density was paid at leaves.
            return zeros

        def real_seam() -> jax.Array:
            # Cache all leaf scores once and evaluate within-segment neighbors
            # once per retained boundary particle, before expanding the seam pair.
            right_values = leaf_values[seam, right.boundary[:, 0]]
            right_gradients = leaf_gradients[seam, right.boundary[:, 0]]
            cross_values, cross_current = jax.vmap(
                lambda before: jax.vmap(
                    lambda current: ctx.transition_current_value_grad_by_param(
                        before, current, seam
                    )
                )(right.first)
            )(left.last)
            cross_values, cross_current = jax.lax.optimization_barrier(
                (cross_values, cross_current)
            )

            def correction(
                current: jax.Array,
                values: jax.Array,
                gradients: jax.Array,
                index: jax.Array,
                guide: jax.Array,
            ) -> jax.Array:
                weights = jax.nn.softmax(ctx.initial_label_log_probs + values, axis=-1)
                gradient = jnp.sum(weights[..., None] * gradients, axis=-2)
                return auxiliary.log_density_from_gradient(current, gradient, index) - guide

            if method == "amala":
                return correction(
                    right.first[None],
                    cross_values + right_values[None],
                    cross_current + right_gradients[None],
                    seam,
                    paid_auxiliary[seam, right.boundary[:, 0]][None],
                ).astype(ctx.traj_dtype)

            def left_correction() -> jax.Array:
                previous = particles[jnp.maximum(seam - 2, 0), left.boundary[:, -2]]
                values, gradients = jax.lax.cond(
                    seam > 1,
                    lambda: jax.vmap(
                        ctx.transition_current_value_grad_by_param, in_axes=(0, 0, None)
                    )(previous, left.last, seam - 1),
                    lambda: (jnp.zeros_like(right_values), jnp.zeros_like(right_gradients)),
                )
                values = values + leaf_values[seam - 1, left.boundary[:, -1]]
                gradients = gradients + leaf_gradients[seam - 1, left.boundary[:, -1]]
                future, future_gradient = jax.vmap(
                    lambda current: jax.vmap(
                        lambda after: ctx.transition_next_value_grad_by_param(current, after, seam)
                    )(right.first)
                )(left.last)
                future, future_gradient = jax.lax.optimization_barrier((future, future_gradient))
                return correction(
                    left.last[:, None],
                    values[:, None] + future,
                    gradients[:, None] + future_gradient,
                    seam - 1,
                    paid_auxiliary[seam - 1, left.boundary[:, -1]][:, None],
                ).astype(ctx.traj_dtype)

            def right_correction() -> jax.Array:
                following = particles[jnp.minimum(seam + 1, steps - 1), right.boundary[:, 1]]
                future, future_gradient = jax.lax.cond(
                    seam + 1 < steps,
                    lambda: jax.vmap(ctx.transition_next_value_grad_by_param, in_axes=(0, 0, None))(
                        right.first, following, jnp.minimum(seam + 1, steps - 1)
                    ),
                    lambda: (jnp.zeros_like(right_values), jnp.zeros_like(right_gradients)),
                )
                return correction(
                    right.first[None],
                    cross_values + right_values[None] + future[None],
                    cross_current + right_gradients[None] + future_gradient[None],
                    seam,
                    paid_auxiliary[seam, right.boundary[:, 0]][None],
                ).astype(ctx.traj_dtype)

            left_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == 1), left_correction, lambda: zeros
            )
            right_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == steps - 1), right_correction, lambda: zeros
            )
            return left_factor + right_factor

        density: jax.Array = jax.lax.cond(seam < steps, real_seam, lambda: zeros)
        return density

    def join(
        left: _Segment,
        right: _Segment,
        seam: jax.Array,
        width: jax.Array,
        emission: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        transition = ctx.pairwise_transition_log_probs(left.last, right.first, seam)
        paid = seam_auxiliary(left, right, seam, width)
        # These exact factors are reused by weighting and retained evidence.
        # Materializing them prevents the GPU fusion pass from repeatedly
        # cloning the full auxiliary-gradient graph into both consumers.
        factors: tuple[jax.Array, jax.Array] = jax.lax.optimization_barrier((transition, paid))
        transition, paid = factors
        joint = (
            left.evidence[:, None]
            + right.evidence[None]
            + (transition + emission)
            + paid[..., None]
        )
        numerator = jax.scipy.special.logsumexp(logpi + joint, axis=-1)
        left_density = jax.scipy.special.logsumexp(logpi + left.evidence, axis=-1)
        right_density = jax.scipy.special.logsumexp(logpi + right.evidence, axis=-1)
        logits = (
            left.weights[:, None]
            + right.weights[None]
            + numerator
            - left_density[:, None]
            - right_density[None]
        )
        return logits.astype(ctx.traj_dtype), joint.astype(ctx.traj_dtype), transition

    def combine(
        left: _Segment,
        right: _Segment,
        seam: jax.Array,
        draw_key: jax.Array,
        plan: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime | None,
        node: jax.Array,
        width: jax.Array,
        emission: jax.Array,
    ) -> _CombinedSegment:
        boundaries: tuple[_Segment, _Segment] = jax.lax.optimization_barrier((left, right))
        left, right = boundaries
        logits, joint, transition = join(left, right, seam, width, emission)
        selected = jnp.concatenate(
            (jnp.zeros(1, dtype=jnp.int32), multinomial(draw_key, logits.reshape(-1), count - 1))
        )
        li, ri = selected // count, selected % count
        stats = left.statistics[li]
        if intervals is not None and plan is not None:
            stats = gather_interval_statistics(
                left.statistics[li], jnp.asarray(plan.left_slots)[node]
            ) + gather_interval_statistics(
                right.statistics[ri], jnp.asarray(plan.right_slots)[node]
            )
        segment = _Segment(
            left.first[li],
            right.last[ri],
            joint[li, ri],
            left.prior_log_prob[li] + right.prior_log_prob[ri] + transition[li, ri]
            if root_correction is not None
            else left.prior_log_prob,
            jnp.stack(
                (
                    left.boundary[li, 0],
                    jnp.where(width == 1, right.boundary[ri, 0], left.boundary[li, 1]),
                    jnp.where(width == 1, left.boundary[li, -1], right.boundary[ri, -2]),
                    right.boundary[ri, -1],
                ),
                axis=-1,
            ),
            jnp.full(count, -math.log(count), dtype=ctx.traj_dtype),
            stats,
        )
        metrics = measure_weights(logits.reshape(-1))
        return segment, metrics, (li, ri)

    def path_from_origins(origins: jax.Array) -> jax.Array:
        return particles[jnp.arange(steps), origins[:steps]]

    def terminal_correction(
        origins: jax.Array, local: jax.Array, local_prior: jax.Array
    ) -> jax.Array:
        if root_correction is None:
            # Every auxiliary pair/triple is already paid at its owning tree node.
            return jnp.asarray(0.0, dtype=ctx.traj_dtype)
        path = path_from_origins(origins)
        replay = root_correction(path, local_prior)
        ratio = jax.scipy.special.logsumexp(logpi + local + replay) - jax.scipy.special.logsumexp(
            logpi + local
        )
        return ratio.astype(ctx.traj_dtype)

    origins = jnp.broadcast_to(jnp.arange(count)[None, :, None], (padded, count, 1))
    leaf_logits = jax.scipy.special.logsumexp(logpi + evidence, axis=-1)
    weights = jax.nn.log_softmax(leaf_logits, axis=-1)
    leaf_metrics = measure_weights(leaf_logits)
    node_statistics = (leaf_metrics,)
    if steps == 1:
        root_logits = leaf_logits[0]
        root_logits = root_logits + jax.vmap(terminal_correction)(origins[0], evidence[0], prior[0])
        chosen = _forced_move(select_key, root_logits)
        selected_origins = origins[0, chosen]
    else:
        pad = padded - steps
        banks = jnp.pad(particles, ((0, pad), (0, 0), (0, 0)))
        segments = _Segment(
            banks,
            banks,
            jnp.pad(evidence, ((0, pad), (0, 0), (0, 0))),
            jnp.pad(prior, ((0, pad), (0, 0), (0, 0))),
            origins,
            jnp.pad(weights, ((0, pad), (0, 0)), constant_values=-math.log(count)),
            statistics,
        )
        max_nodes = padded // 2
        block = min(
            max(1, padded // 2),
            max(1, _PAIRWISE_BATCH_ELEMENTS // (count * count * labels)),
        )
        if batch_size is not None:
            block = min(block, batch_size)
        block = 1 << (block.bit_length() - 1)
        if intervals is not None:
            levels = intervals.tree.levels[1:]
            slots = max(
                segments.statistics.shape[-3], *(plan.left_slots.shape[1] for plan in levels)
            )
            statistics = jnp.pad(
                segments.statistics,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, slots - segments.statistics.shape[-3]),
                    (0, 0),
                    (0, 0),
                ),
            )
            segments = segments._replace(statistics=statistics)
        segments = segments._replace(boundary=jnp.repeat(segments.boundary, 4, axis=-1))

        # Preserve the existing per-node key stream while sharing the merge body.
        split_levels = min(4, depth - 1)
        level_keys = jax.random.split(tree_key, max(depth - 1, 1))
        upper_nodes = padded >> split_levels
        keys_by_level = tuple(
            jax.random.split(level_keys[level], padded >> (level + 1))
            if level < split_levels
            else jax.random.split(jax.random.fold_in(tree_key, level), upper_nodes)
            for level in range(depth)
        )
        keys = jnp.stack(
            tuple(
                level_keys[jnp.minimum(jnp.arange(max_nodes), level_keys.shape[0] - 1)]
                for level_keys in keys_by_level
            )
        )

        def level_step(
            current: _Segment, level: jax.Array, node_batch: int
        ) -> tuple[
            _Segment, tuple[tuple[jax.Array, jax.Array], dict[ParticleWeightMetric, jax.Array]]
        ]:
            width = 1 << level
            active_children = (steps + width - 1) // width
            pairs = (active_children + 1) // 2
            # Emitting-node groups must use this merge body's actual tile.
            # Upper levels share a smaller tile than the leaf-level merge.
            plan = (
                None
                if intervals is None
                else jax.tree.map(
                    lambda value: value[level],
                    stack_interval_tree_levels(
                        intervals.tree.levels[1:],
                        max_nodes,
                        current.statistics.shape[-3],
                        node_batch,
                    ),
                )
            )
            children = (jnp.zeros((max_nodes, count), jnp.int32),) * 2
            metrics = {
                metric: jnp.zeros(max_nodes, dtype=values.dtype)
                for metric, values in leaf_metrics.items()
            }

            def merge_block(index: jax.Array, carry: _CombinedSegment) -> _CombinedSegment:
                next_segments, metrics, children = carry
                offset = index * node_batch
                nodes = offset + jnp.arange(node_batch)

                def child(indices: jax.Array) -> _Segment:
                    selected: _Segment = jax.tree.map(lambda value: value[indices], current)
                    empty = jax.tree.map(jnp.zeros_like, selected)._replace(
                        weights=jnp.full_like(selected.weights, -math.log(count))
                    )
                    present = indices < active_children
                    segment: _Segment = jax.tree.map(
                        lambda value, neutral: jnp.where(
                            present.reshape((*present.shape, *((1,) * (value.ndim - 1)))),
                            value,
                            neutral,
                        ),
                        selected,
                        empty,
                    )
                    return segment

                left = child(2 * nodes)
                right = child(2 * nodes + 1)
                seams = (2 * nodes + 1) * width
                emissions = jnp.zeros((node_batch, count, count, labels), dtype=ctx.traj_dtype)
                if intervals is not None and plan is not None and plan.emitting_nodes.shape[1]:
                    emission_plan = plan
                    score_interval = intervals.pairwise_emission_log_probs

                    def emit(slot: jax.Array, values: jax.Array) -> jax.Array:
                        node = emission_plan.emitting_nodes[index, slot]
                        density = score_interval(
                            ctx.contexts,
                            current.statistics[2 * node],
                            current.statistics[2 * node + 1],
                            emission_plan,
                            node,
                        )
                        return values.at[node - offset].set(density)

                    emissions = jax.lax.fori_loop(0, plan.emission_counts[index], emit, emissions)
                moved, measured, parents = jax.vmap(
                    lambda first, second, seam, key, node, emission: combine(
                        first, second, seam, key, plan, node, width, emission
                    )
                )(left, right, seams, keys[level, nodes], nodes, emissions)

                def update(bank: jax.Array, values: jax.Array) -> jax.Array:
                    return jax.lax.dynamic_update_slice_in_dim(bank, values, offset, axis=0)

                updated: _CombinedSegment = (
                    jax.tree.map(update, next_segments, moved),
                    jax.tree.map(update, metrics, measured),
                    jax.tree.map(update, children, parents),
                )
                return updated

            combined: _CombinedSegment = jax.lax.fori_loop(
                0, (pairs + node_batch - 1) // node_batch, merge_block, (current, metrics, children)
            )
            moved, metrics, children = combined
            return moved, (children, metrics)

        # Two shared merge bodies bound GPU fusion-compiler work independently
        # of tree depth. One specialization per upper level duplicates the full
        # aMALA+ gradient graph and makes priority fusion prohibitively slow.
        # The smaller upper tile limits padded work without serializing nodes.
        # Both scans preserve the original per-node random keys.
        with jax.named_scope("dsmc_tree"):
            shared_depth = depth - (block.bit_length() - 1)
            segments, history = jax.lax.scan(
                lambda current, level: level_step(current, level, min(block, (steps + 1) // 2)),
                segments,
                jnp.arange(shared_depth),
            )
            histories = (history,)
            if shared_depth < depth:
                segments, history = jax.lax.scan(
                    lambda current, level: level_step(current, level, min(block, 128)),
                    segments,
                    jnp.arange(shared_depth, depth),
                )
                histories = (*histories, history)
            (left_history, right_history), history_metrics = jax.tree.map(
                lambda *parts: jnp.concatenate(parts), *histories
            )
        indices = jnp.broadcast_to(jnp.arange(count)[:, None], (count, padded))
        base_nodes = jnp.arange(padded)

        def descend(current: jax.Array, reverse_level: jax.Array) -> tuple[jax.Array, None]:
            level = depth - 1 - reverse_level
            parent = base_nodes >> (level + 1)
            right = (base_nodes >> level) & 1
            li = left_history[level, parent[None], current]
            ri = right_history[level, parent[None], current]
            return jnp.where(right[None], ri, li), None

        root_origins, _ = jax.lax.scan(descend, indices, jnp.arange(depth))
        root = jax.tree.map(lambda value: value[0], segments)
        node_statistics = (
            *node_statistics,
            *(
                {
                    metric: values[level, : (steps + (1 << (level + 1)) - 1) // (1 << (level + 1))]
                    for metric, values in history_metrics.items()
                }
                for level in range(depth)
            ),
        )
        # A conditional resampling stage makes full-path work linear in N.
        # No N x N x T candidate paths or replay-gradient arrays are formed.
        batch = count if batch_size is None else min(count, batch_size)
        with jax.named_scope("dsmc_root_correction"):
            root_logits = jax.lax.map(
                lambda inputs: terminal_correction(*inputs),
                (root_origins, root.evidence, root.prior_log_prob),
                batch_size=batch,
            )
        chosen = _forced_move(select_key, root_logits)
        selected_origins = root_origins[chosen, :steps]
    path = path_from_origins(selected_origins).astype(ctx.latent_dtype)
    weight_diagnostics = weight_history_fields(
        (
            *node_statistics,
            {metric: value[None] for metric, value in measure_weights(root_logits).items()},
        )
    )
    result = ParticleSmootherResult(
        latent_path=path,
        label_scores=ctx.score_labels(path),
        origin_path=selected_origins.astype(jnp.int32),
        proposal_mask=mask,
        diagnostics=MappingProxyType(
            {
                "amala_grad_norm_mean": jnp.mean(auxiliary.gradient_norms).astype(ctx.traj_dtype),
                "amala_grad_norm_max": jnp.max(auxiliary.gradient_norms).astype(ctx.traj_dtype),
                **weight_diagnostics,
            }
        ),
    )
    if record_trace:
        return TracedResult(
            result,
            SmootherTraceRecord(
                auxiliary=auxiliary.auxiliary,
                proposal_centers=auxiliary.auxiliary,
                particles=particles,
                root_selection=CategoricalTraceRecord(_normalize_log_probs(root_logits), chosen),
                selected_particles=selected_origins,
            ),
        )
    return result


def step(
    ctx: SmootherContext,
    key: jax.Array,
    x_ref: jax.Array,
    root_correction: Callable[[jax.Array, jax.Array], jax.Array] | None,
    *,
    batch_size: int | None = None,
) -> ParticleSmootherResult:
    """Advance conditional DSMC without retaining proposal history."""
    result = _step(ctx, key, x_ref, root_correction, batch_size=batch_size)
    assert isinstance(result, ParticleSmootherResult)
    return result


def trace_step(
    ctx: SmootherContext,
    key: jax.Array,
    x_ref: jax.Array,
    root_correction: Callable[[jax.Array, jax.Array], jax.Array] | None,
    *,
    batch_size: int | None = None,
) -> TracedResult[ParticleSmootherResult, SmootherTraceRecord]:
    """Record the same transition and random choices as the production smoother."""
    result = _step(ctx, key, x_ref, root_correction, record_trace=True, batch_size=batch_size)
    assert isinstance(result, TracedResult)
    return result
