"""De-sequentialized conditional-SMC latent smoother (parallel-in-time)."""

# References:
#   https://arxiv.org/abs/2202.02264 — Corenflos, Chopin & Särkkä (2022),
#     arXiv:2202.02264: de-Sequentialized Monte Carlo. The smoothing distribution
#     is built by a divide-and-conquer binary tree over time (Prop. 2.2 stitching,
#     eq. 10-11), instead of a sequential forward filter + backward sampling.
#     We implement the conditional formulation c-dSMC (Section 3,
#     Algorithm 4): the reference trajectory is preserved through every block
#     combination, making this a valid parallel-in-time conditional-SMC Particle
#     Gibbs kernel. No backward-sampling step is needed — the balanced tree resamples
#     every time step at most O(log T) times. Mixing still depends on the target.
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), arXiv:2505.04611:
#     conditional SMC against the posterior mixture over the parameter ensemble. The
#     seam weights and trajectory evidence are marginalized over the parameter labels
#     exactly as in the sibling smoothers (logsumexp over the parameter axis).
#
# Leaf proposals are gradient-informed auxiliary ("amala_exact") proposals — or the
# paid mixture ("paid_mix") that wraps the same z-anchored component together with a
# fixed LAPS-pilot component and a wide tail in one paid density. Both are exactly
# corrected: the auxiliary pseudo-observation potential restores invariance for the
# reference-dependent component, and the mixture density is subtracted in full.
#
# Implementation notes (math-preserving):
#   * The tree is evaluated level-by-level — the time axis is padded to a power of two
#     with evidence-free phantom leaves (their seams are masked out) so every depth is
#     a single batched combine, giving O(log T) sequential depth.
#   * Prior-predictive marginals come from a parallel-prefix (associative) scan over
#     the affine-Gaussian transition operator, not a sequential scan.
#   * The tree carries only segment endpoints, per-label evidence, and integer
#     ancestry. The selected latent path is reconstructed from the leaf particle
#     bank after the root draw, so intermediate levels do not concatenate full
#     trajectory tensors.
#   * Pairwise seam Mahalanobis distances use a whiten-then-GEMM expansion rather than
#     an N x N batch of triangular solves.
#   * Resampling is inverse-CDF multinomial (a single cumulative sum + searchsorted),
#     avoiding the O(N^3) Gumbel-noise materialization of a batched categorical draw.
#   * An interval factor belongs to its smallest enclosing tree node. Leaves
#     carry weighted response moments; a merge adds the children's moments and
#     pays only the factors owned by that node. Thus each per-label segment
#     evidence contains precisely its internal transitions and completed factors.
#     The same target/product-of-children ratio below remains valid for these
#     nonlocal factors, without adding singular accumulator states to the SDE.
#   * For replayed state coordinates, intermediate targets retain only Gaussian
#     initial/transition factors and the paid auxiliary potentials. Observations
#     and the recurrence correction are paid at the root, so a hard likelihood
#     at an artificial recurrence value cannot exclude a valid complete path.

from __future__ import annotations

import math
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _masked_normal_log_prob,
    _normalize_log_probs,
    _observation_log_probs_by_param,
    _value_and_grad_by_param,
    coordinate_proposal_mask,
)
from nof1_causal_lab.models.ssm.inference.targets.intervals import gather_interval_statistics

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.compile.interval_tree import CompiledIntervalTreeLevel


class _Segment(NamedTuple):
    first: jax.Array
    last: jax.Array
    evidence: jax.Array
    origin: jax.Array
    weights: jax.Array
    statistics: jax.Array


def step(
    ctx: SmootherContext,
    key: jax.Array,
    x_ref: jax.Array,
    root_correction: Callable[[jax.Array], jax.Array],
) -> MPGibbsLatentSmootherResult:
    """Conditional de-sequentialized SMC sweep over the posterior parameter mixture."""
    contexts = ctx.contexts
    logpi = ctx.initial_label_log_probs
    runtime_observations = ctx.runtime_observations
    obs_increment_fn = ctx.obs_increment_fn
    trajectory_label_log_probs = ctx.trajectory_label_log_probs
    num_steps = ctx.num_steps
    num_parameter_particles = ctx.num_parameter_particles
    num_particles = ctx.num_free_particles + 1
    num_free_particles = ctx.num_free_particles
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    latent_dim = int(x_ref.shape[-1])
    intervals = ctx.intervals
    # Both leaves (amala_exact, paid_mix) draw the auxiliary trajectory z and pay its
    # pseudo-observation potential; paid_mix additionally mixes in the fixed pilot
    # and wide components (whose reference-independence needs no payment).
    amala_delta = jnp.asarray(ctx.amala_delta, dtype=latent_dtype)
    proposal_var_by_t = jnp.asarray(0.5, dtype=latent_dtype) * amala_delta
    proposal_scale_by_t = jnp.sqrt(proposal_var_by_t)
    proposal_kappa = jnp.asarray(ctx.amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(ctx.amala_grad_clip, dtype=latent_dtype)

    # Coordinate-block restriction: propose only `latent_block_coords` randomly
    # chosen coordinates this sweep (the same block at every time step — a whole-path
    # coordinate-conditional update); all other coordinates are copied from the
    # reference into every particle. The seams still evaluate the full transition
    # density, so cross-coordinate coupling is priced exactly and the sweep is
    # conditional SMC on the coordinate-block conditional of the same target.
    # Sampling a key and splitting that same key into descendants reuses its
    # random stream. Allocate disjoint branches before any draws.
    aux_key, mask_key, key_leaves, key_tree, key_root = random.split(key, 5)
    proposal_mask = coordinate_proposal_mask(
        mask_key, ctx.latent_free_mask, ctx.latent_block_coords
    )

    def _clip_gradient(grad: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        norm: jnp.ndarray = jnp.linalg.norm(grad)
        multiplier = jnp.minimum(
            jnp.asarray(1.0, dtype=latent_dtype),
            grad_clip / jnp.maximum(norm, jnp.asarray(1e-12, dtype=latent_dtype)),
        )
        return (grad * multiplier).astype(latent_dtype), norm.astype(traj_dtype)

    def _obs_value_grad_by_param(
        particle_t: jnp.ndarray, time_idx: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _value_and_grad_by_param(
            contexts,
            particle_t,
            lambda context, particle: obs_increment_fn(
                context, particle, time_idx, runtime_observations
            ),
            value_dtype=traj_dtype,
            grad_dtype=latent_dtype,
        )

    _initial_prior_value_grad_by_param = ctx.initial_value_grad_by_param
    _transition_current_value_grad_by_param = ctx.transition_current_value_grad_by_param
    _transition_next_value_grad_by_param = ctx.transition_next_value_grad_by_param

    # Linearisation points for the gradient leaf: draw an auxiliary trajectory
    # z ~ N(x_ref, (delta/2) I) and linearise there; the matching N(z_t; x_t,
    # (delta/2) I) pseudo-observation potential is added to the leaf weight in
    # _leaf, recovering the exact gradient-informed cSMC proposal of section
    # 3.3.1 in the auxiliary Kalman sampler construction. Centring on x_ref
    # directly (the historical amala/amala_plus variants) is reference-dependent
    # without an auxiliary correction and does not leave the target invariant.
    lin_pts = x_ref + proposal_scale_by_t[:, None] * random.normal(
        aux_key, x_ref.shape, dtype=latent_dtype
    )
    # Auxiliary values obey the same time-varying coordinate restriction as the
    # particles. Both normal densities below are paid only on that subspace.
    lin_pts = jnp.where(proposal_mask, lin_pts, x_ref)

    def _gradient_leaf_proposal_stats(time_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        particle_t = lin_pts[time_idx]
        obs_lp, obs_grad = _obs_value_grad_by_param(particle_t, time_idx)
        init_prior_lp, init_prior_grad = _initial_prior_value_grad_by_param(particle_t)
        prev_idx = jnp.maximum(time_idx - 1, 0)
        transition_lp, transition_grad = _transition_current_value_grad_by_param(
            lin_pts[prev_idx],
            particle_t,
            time_idx,
        )
        is_initial = time_idx == 0
        logits = jnp.where(
            is_initial,
            logpi + init_prior_lp + obs_lp,
            logpi + transition_lp + obs_lp,
        )
        component_grad = jnp.where(
            is_initial,
            init_prior_grad + obs_grad,
            transition_grad + obs_grad,
        )
        next_idx = jnp.minimum(time_idx + 1, num_steps - 1)
        future_lp, future_grad = _transition_next_value_grad_by_param(
            particle_t,
            lin_pts[next_idx],
            next_idx,
        )
        has_future = time_idx < (num_steps - 1)
        logits = jnp.where(has_future, logits + future_lp, logits)
        component_grad = jnp.where(
            has_future,
            component_grad + future_grad,
            component_grad,
        )
        label_log_probs = _normalize_log_probs(logits)
        label_probs = jnp.exp(label_log_probs).astype(latent_dtype)
        grad = jnp.einsum("p,pd->d", label_probs, component_grad)
        grad = jnp.where(proposal_mask[time_idx], grad, 0.0)
        clipped_grad, grad_norm = _clip_gradient(grad)
        drift = (proposal_kappa * proposal_var_by_t[time_idx] * clipped_grad).astype(latent_dtype)
        return (particle_t + drift).astype(latent_dtype), grad_norm

    proposal_centers, proposal_grad_norms = jax.vmap(_gradient_leaf_proposal_stats)(
        jnp.arange(num_steps, dtype=jnp.int32)
    )

    pilot = (
        None
        if ctx.pilot_moments is None
        else (
            *ctx.pilot_moments,
            jnp.sqrt(ctx.pilot_moments[1]),
            jnp.sqrt(ctx.pilot_moments[2]),
        )
    )
    mix_weights = jnp.asarray(
        [
            ctx.paid_mix_z_weight,
            ctx.paid_mix_pilot_weight,
            1.0 - ctx.paid_mix_z_weight - ctx.paid_mix_pilot_weight,
        ],
        dtype=latent_dtype,
    )
    mix_log_weights = jnp.log(mix_weights)

    def _leaf(
        time_idx: jnp.ndarray, leaf_key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        component_key, sample_key = random.split(leaf_key, 2)
        noise = random.normal(
            sample_key,
            (num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        z_component = proposal_centers[time_idx] + proposal_scale_by_t[time_idx] * noise
        if pilot is not None:
            pilot_means, _, _, pilot_scales, pilot_wide_scales = pilot
            component = random.categorical(
                component_key, mix_log_weights, shape=(num_free_particles,)
            )
            pilot_component = pilot_means[time_idx] + pilot_scales[time_idx] * noise
            wide_component = pilot_means[time_idx] + pilot_wide_scales[time_idx] * noise
            free_particles = jnp.where(
                (component == 0)[:, None],
                z_component,
                jnp.where((component == 1)[:, None], pilot_component, wide_component),
            )
        else:
            free_particles = z_component
        free_particles = jnp.where(
            proposal_mask[time_idx][None, :], free_particles, x_ref[time_idx][None, :]
        )
        particles = jnp.concatenate([x_ref[time_idx][None, :], free_particles], axis=0)
        obs_lp = _observation_log_probs_by_param(
            contexts,
            particles,
            jnp.asarray(time_idx, dtype=jnp.int32),
            runtime_observations,
            obs_increment_fn,
        )
        z_proposal_lp = _masked_normal_log_prob(
            particles,
            proposal_centers[time_idx],
            proposal_var_by_t[time_idx],
            proposal_mask[time_idx],
        )
        if pilot is not None:
            pilot_means, pilot_vars, pilot_wide_vars, _, _ = pilot
            pilot_lp = _masked_normal_log_prob(
                particles, pilot_means[time_idx], pilot_vars[time_idx], proposal_mask[time_idx]
            )
            wide_lp = _masked_normal_log_prob(
                particles, pilot_means[time_idx], pilot_wide_vars[time_idx], proposal_mask[time_idx]
            )
            proposal_lp = jax.scipy.special.logsumexp(
                jnp.stack(
                    [
                        mix_log_weights[0] + z_proposal_lp,
                        mix_log_weights[1] + pilot_lp,
                        mix_log_weights[2] + wide_lp,
                    ]
                ),
                axis=0,
            )
        else:
            proposal_lp = z_proposal_lp
        init_prior_lp = jax.vmap(lambda particle: ctx.initial_value_grad_by_param(particle)[0])(
            particles
        )
        # Pseudo-observation potential N(z_t; x_t, (delta/2) I) of the auxiliary
        # extended target (label-independent, so it factors through the parameter
        # logsumexp). With the q(x_t | z_t) proposal density already subtracted
        # above, this is the exact target/proposal leaf weight; the pilot/wide
        # mixture components are reference-independent so the same payment covers
        # paid_mix.
        aux_lp = _masked_normal_log_prob(
            particles, lin_pts[time_idx], proposal_var_by_t[time_idx], proposal_mask[time_idx]
        )
        tail_psi = obs_lp - proposal_lp[:, None] + aux_lp[:, None]
        initial_psi = init_prior_lp + tail_psi
        psi = jnp.where(time_idx == 0, initial_psi, tail_psi).astype(traj_dtype)
        evidence = jax.scipy.special.logsumexp(logpi[None, :] + psi, axis=1)
        log_weights = _normalize_log_probs(evidence).astype(traj_dtype)
        origin = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32)[:, None], (num_particles, 1)
        )
        return particles.astype(latent_dtype), psi, origin, log_weights

    def _phantom_leaf() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evidence-free padding leaf: psi = 0, uniform weights, sliced off the output."""
        particles = jnp.zeros((num_particles, latent_dim), dtype=latent_dtype)
        psi = jnp.zeros((num_particles, num_parameter_particles), dtype=traj_dtype)
        origin = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32)[:, None], (num_particles, 1)
        )
        log_weights = jnp.full((num_particles,), -math.log(num_particles), dtype=traj_dtype)
        return particles, psi, origin, log_weights

    def _multinomial(draw_key: jax.Array, logits: jnp.ndarray, num_draws: int) -> jnp.ndarray:
        probabilities = jax.nn.softmax(logits)
        cumulative = jnp.cumsum(probabilities)
        uniforms = random.uniform(draw_key, (num_draws,), dtype=cumulative.dtype)
        indices = jnp.searchsorted(cumulative, uniforms, side="right")
        return jnp.minimum(indices, logits.shape[0] - 1).astype(jnp.int32)

    def _selected_transition_log_probs(
        prev_particles: jnp.ndarray, next_particles: jnp.ndarray, seam: jnp.ndarray | int
    ) -> jnp.ndarray:
        return ctx.selected_transition_log_probs(prev_particles, next_particles, seam)

    def _stitch_logits(
        left: _Segment,
        right: _Segment,
        seam: jnp.ndarray | int,
        plan: CompiledIntervalTreeLevel | None,
        node: jax.Array,
        correction: jax.Array | float = 0.0,
    ) -> jnp.ndarray:
        transition_lp = ctx.pairwise_transition_log_probs(left.last, right.first, seam)
        if intervals is not None and plan is not None and plan.emission_slots.shape[1]:
            emission_log_probs = intervals.emission_log_probs
            left_stats = gather_interval_statistics(
                left.statistics, jnp.asarray(plan.left_slots)[node]
            )
            right_stats = gather_interval_statistics(
                right.statistics, jnp.asarray(plan.right_slots)[node]
            )
            transition_lp = transition_lp + jax.vmap(
                lambda stats: emission_log_probs(contexts, stats[None] + right_stats, plan, node)
            )(left_stats)
        log_joint = jax.scipy.special.logsumexp(
            logpi[None, None, :]
            + left.evidence[:, None, :]
            + right.evidence[None, :, :]
            + transition_lp
            + correction,
            # The full-path correction is paid only at the root. Intermediate
            # targets retain local factors; the root target is the exact model.
            axis=-1,
        )
        log_left = jax.scipy.special.logsumexp(logpi[None, :] + left.evidence, axis=1)
        log_right = jax.scipy.special.logsumexp(logpi[None, :] + right.evidence, axis=1)
        seam_coupling = log_joint - log_left[:, None] - log_right[None, :]
        pair_logits = left.weights[:, None] + right.weights[None, :] + seam_coupling
        return pair_logits.astype(traj_dtype)

    def _combine(
        left: _Segment,
        right: _Segment,
        seam: jnp.ndarray,
        combine_key: jax.Array,
        plan: CompiledIntervalTreeLevel | None,
        node: jax.Array,
    ) -> _Segment:
        pair_logits = _stitch_logits(left, right, seam, plan, node)
        free_pairs = _multinomial(combine_key, pair_logits.reshape(-1), num_free_particles)
        selected = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), free_pairs], axis=0)
        left_idx = selected // num_particles
        right_idx = selected % num_particles
        first = left.first[left_idx]
        last = right.last[right_idx]
        origin = jnp.concatenate([left.origin[left_idx], right.origin[right_idx]], axis=1)
        transition_lp = _selected_transition_log_probs(
            left.last[left_idx],
            right.first[right_idx],
            seam,
        )
        statistics = left.statistics
        if intervals is not None and plan is not None:
            statistics = gather_interval_statistics(
                left.statistics[left_idx], jnp.asarray(plan.left_slots)[node]
            ) + gather_interval_statistics(
                right.statistics[right_idx], jnp.asarray(plan.right_slots)[node]
            )
            if plan.emission_slots.shape[1]:
                transition_lp = transition_lp + intervals.emission_log_probs(
                    contexts, statistics, plan, node
                )
        psi = (left.evidence[left_idx] + right.evidence[right_idx] + transition_lp).astype(
            traj_dtype
        )
        log_weights = jnp.full((num_particles,), -math.log(num_particles), dtype=traj_dtype)
        return _Segment(first, last, psi, origin, log_weights, statistics)

    with jax.named_scope("dsmc_tree"):
        depth = max((num_steps - 1).bit_length(), 0)
        padded_steps = 1 << depth
        leaf_keys = random.split(key_leaves, num_steps)
        with jax.named_scope("dsmc_leaves"):
            leaf_particles, leaf_psi, leaf_origin, leaf_weights = jax.vmap(_leaf)(
                jnp.arange(num_steps, dtype=jnp.int32),
                leaf_keys,
            )
        statistics = (
            jnp.zeros(
                (padded_steps, num_particles, num_parameter_particles, 0, 0, 1), dtype=latent_dtype
            )
            if intervals is None
            else intervals.leaf_statistics(contexts, leaf_particles)
        )
        if intervals is not None and intervals.tree.levels[0].emission_slots.shape[1]:
            leaf_psi = leaf_psi + jax.vmap(
                lambda stats, node: intervals.emission_log_probs(
                    contexts, stats, intervals.tree.levels[0], node
                )
            )(statistics[:num_steps], jnp.arange(num_steps))
            evidence = jax.scipy.special.logsumexp(logpi[None, None, :] + leaf_psi, axis=-1)
            leaf_weights = jax.vmap(_normalize_log_probs)(evidence).astype(traj_dtype)
        if num_steps == 1:
            correction = jax.vmap(lambda state: root_correction(state[None]))(leaf_particles[0])
            evidence = jax.scipy.special.logsumexp(
                logpi[None, :] + leaf_psi[0] + correction, axis=1
            )
            chosen = _multinomial(key_root, evidence, 1)[0]
            latent_path = leaf_particles[0, chosen][None, :]
            origin_path = leaf_origin[0, chosen]
        else:
            phantom_particles, phantom_psi, phantom_origin, phantom_weights = _phantom_leaf()
            num_phantom = padded_steps - num_steps
            phantom_particles = jnp.broadcast_to(
                phantom_particles,
                (num_phantom, num_particles, latent_dim),
            )
            phantom_psi = jnp.broadcast_to(
                phantom_psi,
                (num_phantom, num_particles, num_parameter_particles),
            )
            phantom_origin = jnp.broadcast_to(
                phantom_origin,
                (num_phantom, num_particles, 1),
            )
            phantom_weights = jnp.broadcast_to(
                phantom_weights,
                (num_phantom, num_particles),
            )
            first = jnp.concatenate([leaf_particles, phantom_particles], axis=0)
            last = first
            psi = jnp.concatenate([leaf_psi, phantom_psi], axis=0)
            origin = jnp.concatenate([leaf_origin, phantom_origin], axis=0)
            weights = jnp.concatenate([leaf_weights, phantom_weights], axis=0)
            level_keys = random.split(key_tree, max(depth - 1, 1))
            segments = padded_steps
            for level in range(depth - 1):
                num_pairs = segments // 2
                seams = (1 << level) + jnp.arange(num_pairs, dtype=jnp.int32) * (1 << (level + 1))
                pair_keys = random.split(level_keys[level], num_pairs)
                left = _Segment(
                    first[0::2],
                    last[0::2],
                    psi[0::2],
                    origin[0::2],
                    weights[0::2],
                    statistics[0::2],
                )
                right = _Segment(
                    first[1::2],
                    last[1::2],
                    psi[1::2],
                    origin[1::2],
                    weights[1::2],
                    statistics[1::2],
                )
                plan = None if intervals is None else intervals.tree.levels[level + 1]
                with jax.named_scope("dsmc_combine"):
                    # All merges at this level share one vmap, including their N x N
                    # particle-pair workspaces. Future memory-budgeted batching should
                    # split independent merges while preserving each merge's full
                    # resampling law and its existing random key.
                    first, last, psi, origin, weights, statistics = jax.vmap(
                        partial(_combine, plan=plan)
                    )(left, right, seams, pair_keys, node=jnp.arange(num_pairs))
                segments = num_pairs
            left_root = _Segment(first[0], last[0], psi[0], origin[0], weights[0], statistics[0])
            right_root = _Segment(first[1], last[1], psi[1], origin[1], weights[1], statistics[1])
            plan = None if intervals is None else intervals.tree.levels[depth]

            def correct_pair(left_origins: jax.Array, right_origins: jax.Array) -> jax.Array:
                origins = jnp.concatenate((left_origins, right_origins))[:num_steps]
                path = leaf_particles[jnp.arange(num_steps), origins]
                return root_correction(path)

            # Keep all N x N root weights, but evaluate eight left choices at a
            # time instead of materializing N x N complete trajectories. Each
            # batch still reconstructs time with an affine prefix and reduction;
            # its depth is O(log T), with a batch count independent of T.
            correction = jax.lax.map(
                lambda left_origins: jax.vmap(correct_pair, in_axes=(None, 0))(
                    left_origins, right_root.origin
                ),
                left_root.origin,
                batch_size=min(8, num_particles),
            )
            with jax.named_scope("dsmc_combine"):
                pair_logits = _stitch_logits(
                    left_root, right_root, padded_steps // 2, plan, jnp.asarray(0), correction
                )
            chosen = _multinomial(key_root, pair_logits.reshape(-1), 1)[0]
            left_idx = chosen // num_particles
            right_idx = chosen % num_particles
            origin_path = jnp.concatenate([origin[0][left_idx], origin[1][right_idx]], axis=0)[
                :num_steps
            ]
            latent_path = leaf_particles[jnp.arange(num_steps, dtype=jnp.int32), origin_path]

    latent_path = latent_path.astype(latent_dtype)
    final_label_log_probs = trajectory_label_log_probs(latent_path)
    diagnostics = {
        "amala_grad_norm_mean": jnp.mean(proposal_grad_norms).astype(traj_dtype),
        "amala_grad_norm_max": jnp.max(proposal_grad_norms).astype(traj_dtype),
    }
    return MPGibbsLatentSmootherResult(
        latent_path=latent_path,
        final_label_log_probs=final_label_log_probs,
        origin_path=origin_path.astype(jnp.int32),
        proposal_mask=proposal_mask,
        diagnostics=MappingProxyType(diagnostics),
    )
