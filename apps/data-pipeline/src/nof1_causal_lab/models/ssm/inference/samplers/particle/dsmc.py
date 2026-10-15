"""Conditional divide-and-conquer SMC for the exact auxiliary trajectory target.

The conditional tree is Algorithm 4 of https://arxiv.org/abs/2202.02264.
Auxiliary factors are owned by the smallest node containing their pair (aMALA)
or triple (aMALA+); interval emissions follow their compiled ownership plan.
Parameter labels remain marginalized throughout the tree.

For nonlocal corrections, the root first conditionally resamples N trajectories
from its local target, retaining the reference. Importance selection then replaces
that target by the exact replayed, auxiliary-augmented target. This is a final
conditional SMC stage: its conditional particle-generation law is unchanged,
and its terminal weights are the exact target/local-target ratio. Consequently
only N complete trajectories need replay and exact gradients, instead of N squared.
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
from nof1_causal_lab.models.ssm.inference.targets.intervals import gather_interval_statistics
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
    origin: jax.Array
    weights: jax.Array
    statistics: jax.Array
    auxiliary_log_density: jax.Array


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
    count = ctx.num_free_particles + 1
    steps = ctx.num_steps
    labels = ctx.num_parameter_particles
    depth = (steps - 1).bit_length()
    padded = 1 << depth
    logpi = ctx.initial_label_log_probs
    intervals = ctx.intervals
    method = ctx.dsmc_leaf_proposal
    aux_key, mask_key, leaf_key, tree_key, root_key, select_key = jax.random.split(key, 6)
    mask = coordinate_proposal_mask(mask_key, ctx.latent_free_mask, ctx.latent_block_coords)
    auxiliary = build_auxiliary(ctx, aux_key, x_ref, mask)
    with jax.named_scope("dsmc_leaves"):
        particles, proposal_log_prob = auxiliary.propose(leaf_key)

    def leaf(index: jax.Array, bank: jax.Array, proposal: jax.Array) -> tuple[jax.Array, jax.Array]:
        observations = _observation_log_probs_by_param(
            ctx.contexts, bank, index, ctx.runtime_observations, ctx.obs_increment_fn
        )
        prior: jax.Array = jax.lax.cond(
            index == 0,
            lambda: jax.vmap(lambda value: ctx.initial_value_grad_by_param(value)[0])(bank),
            lambda: jnp.zeros((count, labels), dtype=ctx.traj_dtype),
        )
        owns_auxiliary = (index == 0) & ((method == "amala") | (steps == 1))
        paid: jax.Array = jax.lax.cond(
            owns_auxiliary,
            lambda: jax.vmap(lambda value: auxiliary.local_log_density(value, value, value, index))(
                bank
            ).astype(ctx.traj_dtype),
            lambda: jnp.zeros(count, dtype=ctx.traj_dtype),
        )
        return observations + prior - proposal[:, None] + paid[:, None], paid

    evidence, paid_auxiliary = jax.vmap(leaf)(jnp.arange(steps), particles, proposal_log_prob)
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

    def seam_auxiliary(left: _Segment, right: _Segment, seam: jax.Array) -> jax.Array:
        zeros = jnp.zeros((count, count), dtype=ctx.traj_dtype)
        width = left.origin.shape[-1]

        def real_seam() -> jax.Array:
            if method == "amala":
                return jax.vmap(
                    lambda previous: jax.vmap(
                        lambda current: auxiliary.local_log_density(
                            previous, current, current, seam
                        )
                    )(right.first)
                )(left.last).astype(ctx.traj_dtype)
            previous = (
                particles[jnp.maximum(seam - 2, 0), left.origin[:, -2]] if width > 1 else left.last
            )
            following = (
                particles[jnp.minimum(seam + 1, steps - 1), right.origin[:, 1]]
                if width > 1
                else right.first
            )
            left_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == 1),
                lambda: jax.vmap(
                    lambda before, current: jax.vmap(
                        lambda after: auxiliary.local_log_density(before, current, after, seam - 1)
                    )(right.first)
                )(previous, left.last).astype(ctx.traj_dtype),
                lambda: zeros,
            )
            right_factor: jax.Array = jax.lax.cond(
                (width > 1) | (seam == steps - 1),
                lambda: jax.vmap(
                    lambda before: jax.vmap(
                        lambda current, after: auxiliary.local_log_density(
                            before, current, after, seam
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
        plan: CompiledIntervalTreeLevel | None,
        node: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        transition = ctx.pairwise_transition_log_probs(left.last, right.first, seam)
        paid = seam_auxiliary(left, right, seam)
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
        return logits.astype(ctx.traj_dtype), joint.astype(ctx.traj_dtype), paid

    def combine(
        left: _Segment,
        right: _Segment,
        seam: jax.Array,
        draw_key: jax.Array,
        plan: CompiledIntervalTreeLevel | None,
        node: jax.Array,
    ) -> tuple[_Segment, dict[ParticleWeightMetric, jax.Array]]:
        logits, joint, paid = join(left, right, seam, plan, node)
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
            jnp.concatenate((left.origin[li], right.origin[ri]), axis=-1),
            jnp.full(count, -math.log(count), dtype=ctx.traj_dtype),
            stats,
            left.auxiliary_log_density[li] + right.auxiliary_log_density[ri] + paid[li, ri],
        )
        return segment, weight_statistics(logits.reshape(-1)) if ctx.diagnostic_weights else {}

    def path_from_origins(origins: jax.Array) -> jax.Array:
        return particles[jnp.arange(steps), origins[:steps]]

    def terminal_correction(origins: jax.Array, local: jax.Array, paid: jax.Array) -> jax.Array:
        if ctx.local_auxiliary_exact and root_correction is None:
            # Every auxiliary pair/triple is already paid at its owning tree node.
            return jnp.asarray(0.0, dtype=ctx.traj_dtype)
        path = path_from_origins(origins)
        replay = (
            jnp.zeros(labels, dtype=ctx.traj_dtype)
            if root_correction is None
            else root_correction(path)
        )
        ratio = jax.scipy.special.logsumexp(logpi + local + replay) - jax.scipy.special.logsumexp(
            logpi + local
        )
        ratio = ratio + auxiliary.path_log_density(path) - paid
        return ratio.astype(ctx.traj_dtype)

    origins = jnp.broadcast_to(jnp.arange(count)[None, :, None], (padded, count, 1))
    leaf_logits = jax.scipy.special.logsumexp(logpi + evidence, axis=-1)
    weights = jax.nn.log_softmax(leaf_logits, axis=-1)
    node_statistics = (weight_statistics(leaf_logits),) if ctx.diagnostic_weights else ()
    if steps == 1:
        root_logits = leaf_logits[0]
        root_logits = root_logits + jax.vmap(terminal_correction)(
            origins[0], evidence[0], paid_auxiliary[0]
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
            jnp.pad(paid_auxiliary, ((0, pad), (0, 0))),
        )
        level_keys = jax.random.split(tree_key, max(depth - 1, 1))
        with jax.named_scope("dsmc_tree"):
            for level in range(depth - 1):
                pairs = segments.first.shape[0] // 2
                left = jax.tree.map(lambda value: value[::2], segments)
                right = jax.tree.map(lambda value: value[1::2], segments)
                seams = (1 << level) + jnp.arange(pairs) * (1 << (level + 1))
                keys = jax.random.split(level_keys[level], pairs)
                plan = None if intervals is None else intervals.tree.levels[level + 1]
                # Bound simultaneous pair matrices. The exact N^2 multinomial law
                # and each node's random key are independent of this scheduling.
                batch = pairs if batch_size is None else min(pairs, batch_size)
                batch = 1 << (batch.bit_length() - 1)
                with jax.named_scope("dsmc_combine"):
                    inputs = jax.tree.map(
                        lambda value, pairs=pairs, batch=batch: value.reshape(
                            pairs // batch, batch, *value.shape[1:]
                        ),
                        (left, right, seams, keys, jnp.arange(pairs)),
                    )
                    grouped = jax.lax.map(
                        lambda group, plan=plan: jax.vmap(
                            lambda first, second, seam, key, node: combine(
                                first, second, seam, key, plan, node
                            )
                        )(*group),
                        inputs,
                    )
                    segments, level_statistics = jax.tree.map(
                        lambda value, pairs=pairs: value.reshape(pairs, *value.shape[2:]), grouped
                    )
                    if ctx.diagnostic_weights:
                        width = 1 << (level + 1)
                        real_nodes = (steps + width - 1) // width
                        node_statistics = (
                            *node_statistics,
                            {
                                metric: values[:real_nodes]
                                for metric, values in level_statistics.items()
                            },
                        )
        left_root = jax.tree.map(lambda value: value[0], segments)
        right_root = jax.tree.map(lambda value: value[1], segments)
        plan = None if intervals is None else intervals.tree.levels[depth]
        seam = jnp.asarray(padded // 2)
        root, root_statistics = combine(left_root, right_root, seam, root_key, plan, jnp.asarray(0))
        if ctx.diagnostic_weights:
            node_statistics = (
                *node_statistics,
                {metric: value[None] for metric, value in root_statistics.items()},
            )
        # A conditional resampling stage makes full-path work linear in N.
        # No N x N x T candidate paths or replay-gradient arrays are formed.
        batch = count if batch_size is None else min(count, batch_size)
        with jax.named_scope("dsmc_root_correction"):
            root_logits = jax.lax.map(
                lambda inputs: terminal_correction(*inputs),
                (root.origin, root.evidence, root.auxiliary_log_density),
                batch_size=batch,
            )
        chosen = _forced_move(select_key, root_logits)
        selected_origins = root.origin[chosen, :steps]
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
