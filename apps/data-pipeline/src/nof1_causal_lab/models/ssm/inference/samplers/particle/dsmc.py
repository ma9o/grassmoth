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

from nof1_causal_lab.models.ssm.inference.samplers.particle.auxiliary import build_auxiliary
from nof1_causal_lab.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.math import (
    _normalize_log_probs,
    _observation_log_probs_by_param,
    coordinate_proposal_mask,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.weights import (
    weight_history_fields,
    weight_statistics,
)
from nof1_causal_lab.models.ssm.inference.targets.intervals import (
    IntervalTreeLevelRuntime,
    gather_interval_statistics,
    stack_interval_tree_levels,
)
from nof1_causal_lab.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    SmootherTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
    from nof1_causal_lab.models.ssm.compile.interval_tree import CompiledIntervalTreeLevel


class _Segment(NamedTuple):
    first: jax.Array
    last: jax.Array
    evidence: jax.Array
    boundary: jax.Array
    weights: jax.Array
    statistics: jax.Array


type _CombinedSegment = tuple[
    _Segment, dict[ParticleWeightMetric, jax.Array], tuple[jax.Array, jax.Array]
]


def _multinomial(key: jax.Array, logits: jax.Array, count: int) -> jax.Array:
    cumulative = jnp.cumsum(jax.nn.softmax(logits))
    uniforms = jax.random.uniform(key, (count,), dtype=cumulative.dtype)
    return jnp.minimum(jnp.searchsorted(cumulative, uniforms, side="right"), logits.size - 1)


def _forced_move(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Modified Gibbs selection starting at the retained reference (index zero)."""
    proposal_key, accept_key = jax.random.split(key)
    indices = jnp.arange(logits.size)
    other = jnp.where(indices == 0, -jnp.inf, logits)
    other_mass = jax.scipy.special.logsumexp(other)

    def move() -> jax.Array:
        proposed = _multinomial(proposal_key, other, 1)[0]
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
    root_correction: Callable[[jax.Array], jax.Array] | None,
    *,
    record_trace: bool = False,
    batch_size: int | None = None,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, SmootherTraceRecord]:
    # Reuse traced density and gradient bodies across leaf, seam, and root batches.
    # These JIT boundaries preserve the tree shapes and the exact factors.
    ctx = ctx._replace(
        initial_value_grad_by_param=jax.jit(ctx.initial_value_grad_by_param),
        transition_current_value_grad_by_param=jax.jit(ctx.transition_current_value_grad_by_param),
        transition_next_value_grad_by_param=jax.jit(ctx.transition_next_value_grad_by_param),
        pairwise_transition_log_probs=jax.jit(ctx.pairwise_transition_log_probs),
        obs_increment_fn=jax.jit(ctx.obs_increment_fn),
    )
    count = ctx.num_free_particles + 1
    steps = ctx.num_steps
    labels = ctx.num_parameter_particles
    depth = (steps - 1).bit_length()
    padded = 1 << depth
    logpi = ctx.initial_label_log_probs
    intervals = ctx.intervals
    method = ctx.dsmc_leaf_proposal
    aux_key, mask_key, leaf_key, tree_key, _root_key, select_key = jax.random.split(key, 6)
    mask = coordinate_proposal_mask(mask_key, ctx.latent_free_mask, ctx.latent_block_coords)
    auxiliary = build_auxiliary(ctx, aux_key, x_ref, mask)
    local_log_density: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array] = jax.jit(
        auxiliary.local_log_density
    )
    with jax.named_scope("dsmc_leaves"):
        particles, proposal_log_prob = auxiliary.propose(leaf_key)

    observations = jax.vmap(
        lambda index, bank: _observation_log_probs_by_param(
            ctx.contexts, bank, index, ctx.runtime_observations, ctx.obs_increment_fn
        )
    )(jnp.arange(steps), particles)
    initial = jax.vmap(lambda value: ctx.initial_value_grad_by_param(value)[0])(particles[0])
    paid_auxiliary = jax.vmap(auxiliary.guide_log_density)(particles, jnp.arange(steps)).astype(
        ctx.traj_dtype
    )
    if ctx.amala_kappa != 0 and (method == "amala" or steps == 1):
        initial_auxiliary = jax.vmap(
            lambda value: local_log_density(value, value, value, jnp.asarray(0))
        )(particles[0]).astype(ctx.traj_dtype)
        paid_auxiliary = paid_auxiliary.at[0].set(initial_auxiliary)
    evidence = (
        observations.at[0].add(initial) - proposal_log_prob[..., None] + paid_auxiliary[..., None]
    )
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
        if ctx.amala_kappa == 0:
            # The guide is the actual factor: all auxiliary density was paid at leaves.
            return zeros

        def real_seam() -> jax.Array:
            if method == "amala":
                seam_density: jax.Array = jax.vmap(
                    lambda previous: jax.vmap(
                        lambda current: (
                            local_log_density(previous, current, current, seam)
                            - auxiliary.guide_log_density(current, seam)
                        )
                    )(right.first)
                )(left.last).astype(ctx.traj_dtype)
                return seam_density
            previous = jnp.where(
                width > 1,
                particles[jnp.maximum(seam - 2, 0), left.boundary[:, -2]],
                left.last,
            )
            following = jnp.where(
                width > 1,
                particles[jnp.minimum(seam + 1, steps - 1), right.boundary[:, 1]],
                right.first,
            )
            left_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == 1),
                lambda: jax.vmap(
                    lambda before, current: jax.vmap(
                        lambda after: (
                            local_log_density(before, current, after, seam - 1)
                            - auxiliary.guide_log_density(current, seam - 1)
                        )
                    )(right.first)
                )(previous, left.last).astype(ctx.traj_dtype),
                lambda: zeros,
            )
            right_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == steps - 1),
                lambda: jax.vmap(
                    lambda before: jax.vmap(
                        lambda current, after: (
                            local_log_density(before, current, after, seam)
                            - auxiliary.guide_log_density(current, seam)
                        )
                    )(right.first, following)
                )(left.last).astype(ctx.traj_dtype),
                lambda: zeros,
            )
            return left_factor + right_factor

        density: jax.Array = jax.lax.cond(seam < steps, real_seam, lambda: zeros)
        return density

    def join(
        left: _Segment,
        right: _Segment,
        seam: jax.Array,
        plan: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime | None,
        node: jax.Array,
        width: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        transition = ctx.pairwise_transition_log_probs(left.last, right.first, seam)
        paid = seam_auxiliary(left, right, seam, width)
        if intervals is not None and plan is not None and plan.emission_slots.shape[1]:
            transition = transition + intervals.pairwise_emission_log_probs(
                ctx.contexts, left.statistics, right.statistics, plan, node
            )
        joint = left.evidence[:, None] + right.evidence[None] + transition + paid[..., None]
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
        return logits.astype(ctx.traj_dtype), joint.astype(ctx.traj_dtype)

    def combine(
        left: _Segment,
        right: _Segment,
        seam: jax.Array,
        draw_key: jax.Array,
        plan: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime | None,
        node: jax.Array,
        width: jax.Array,
    ) -> _CombinedSegment:
        logits, joint = join(left, right, seam, plan, node, width)
        selected = jnp.concatenate(
            (jnp.zeros(1, dtype=jnp.int32), _multinomial(draw_key, logits.reshape(-1), count - 1))
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
        metrics = weight_statistics(logits.reshape(-1)) if ctx.diagnostic_weights else {}
        return segment, metrics, (li, ri)

    def path_from_origins(origins: jax.Array) -> jax.Array:
        return particles[jnp.arange(steps), origins[:steps]]

    def terminal_correction(origins: jax.Array, local: jax.Array) -> jax.Array:
        if root_correction is None:
            # Every auxiliary pair/triple is already paid at its owning tree node.
            return jnp.asarray(0.0, dtype=ctx.traj_dtype)
        path = path_from_origins(origins)
        replay = root_correction(path)
        ratio = jax.scipy.special.logsumexp(logpi + local + replay) - jax.scipy.special.logsumexp(
            logpi + local
        )
        return ratio.astype(ctx.traj_dtype)

    origins = jnp.broadcast_to(jnp.arange(count)[None, :, None], (padded, count, 1))
    leaf_logits = jax.scipy.special.logsumexp(logpi + evidence, axis=-1)
    weights = jax.nn.log_softmax(leaf_logits, axis=-1)
    leaf_metrics = weight_statistics(leaf_logits) if ctx.diagnostic_weights else {}
    node_statistics = (leaf_metrics,) if ctx.diagnostic_weights else ()
    if steps == 1:
        root_logits = leaf_logits[0]
        root_logits = root_logits + jax.vmap(terminal_correction)(
            origins[0], evidence[0]
        )
        chosen = _forced_move(select_key, root_logits)
        selected_origins = origins[0, chosen]
    else:
        pad = padded - steps
        banks = jnp.pad(particles, ((0, pad), (0, 0), (0, 0)))
        segments = _Segment(
            banks,
            banks,
            jnp.pad(evidence, ((0, pad), (0, 0), (0, 0))),
            origins,
            jnp.pad(weights, ((0, pad), (0, 0)), constant_values=-math.log(count)),
            statistics,
        )
        max_nodes = padded // 2
        block = max(1, padded // 16)
        if batch_size is not None:
            block = min(block, batch_size)
        block = 1 << (block.bit_length() - 1)
        if intervals is None:
            plans = None
        else:
            levels = intervals.tree.levels[1:]
            slots = max(
                segments.statistics.shape[-3], *(plan.left_slots.shape[1] for plan in levels)
            )
            plans = stack_interval_tree_levels(levels, max_nodes, slots)
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
            current: _Segment, level: jax.Array
        ) -> tuple[
            _Segment, tuple[tuple[jax.Array, jax.Array], dict[ParticleWeightMetric, jax.Array]]
        ]:
            pairs = padded >> (level + 1)
            width = 1 << level
            plan = None if plans is None else jax.tree.map(lambda value: value[level], plans)
            children = (jnp.zeros((max_nodes, count), jnp.int32),) * 2
            metrics = {
                metric: jnp.zeros(max_nodes, dtype=values.dtype)
                for metric, values in leaf_metrics.items()
            }

            def merge_block(index: jax.Array, carry: _CombinedSegment) -> _CombinedSegment:
                next_segments, metrics, children = carry
                offset = index * block
                nodes = offset + jnp.arange(block)
                left = jax.tree.map(lambda value: value[2 * nodes], current)
                right = jax.tree.map(lambda value: value[2 * nodes + 1], current)
                seams = (2 * nodes + 1) * width
                moved, measured, parents = jax.vmap(
                    lambda first, second, seam, key, node: combine(
                        first, second, seam, key, plan, node, width
                    )
                )(left, right, seams, keys[level, nodes], nodes)

                def update(bank: jax.Array, values: jax.Array) -> jax.Array:
                    return jax.lax.dynamic_update_slice_in_dim(bank, values, offset, axis=0)

                updated: _CombinedSegment = (
                    jax.tree.map(update, next_segments, moved),
                    jax.tree.map(update, metrics, measured),
                    jax.tree.map(update, children, parents),
                )
                return updated

            combined: _CombinedSegment = jax.lax.fori_loop(
                0, (pairs + block - 1) // block, merge_block, (current, metrics, children)
            )
            moved, metrics, children = combined
            return moved, (children, metrics)

        # At the default block width there are at most eight blocks at any level,
        # independent of T. Both the forward tree and reverse ancestry have O(log T) depth.
        with jax.named_scope("dsmc_tree"):
            segments, ((left_history, right_history), history_metrics) = jax.lax.scan(
                level_step, segments, jnp.arange(depth)
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
        if ctx.diagnostic_weights:
            node_statistics = (
                *node_statistics,
                *(
                    {
                        metric: values[
                            level, : (steps + (1 << (level + 1)) - 1) // (1 << (level + 1))
                        ]
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
                (root_origins, root.evidence),
                batch_size=batch,
            )
        chosen = _forced_move(select_key, root_logits)
        selected_origins = root_origins[chosen, :steps]
    path = path_from_origins(selected_origins).astype(ctx.latent_dtype)
    weight_diagnostics = (
        weight_history_fields(
            (
                *node_statistics,
                {metric: value[None] for metric, value in weight_statistics(root_logits).items()},
            )
        )
        if ctx.diagnostic_weights
        else {}
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
    root_correction: Callable[[jax.Array], jax.Array] | None,
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
    root_correction: Callable[[jax.Array], jax.Array] | None,
    *,
    batch_size: int | None = None,
) -> TracedResult[ParticleSmootherResult, SmootherTraceRecord]:
    """Record the same transition and random choices as the production smoother."""
    result = _step(ctx, key, x_ref, root_correction, record_trace=True, batch_size=batch_size)
    assert isinstance(result, TracedResult)
    return result
