"""Auxiliary Kalman MH paths inside the unchanged MPGibbs parameter ensemble.

Corenflos & Särkkä (2025), https://arxiv.org/abs/2303.00301, section 2.2 and
their Lorenz experiment. Draw u ~ N(x, delta/2), linearize at x, draw a whole
Gaussian path, rebuild at the proposal and pay the normalized reverse density.
The proposal uses the symmetric parameter-ensemble guide. The target remains
the sum of complete nonlinear path densities over the same corrected parameter
labels as every other MPGibbs smoother. The auxiliary is label-independent, so
the final parameter-label law remains unchanged, including on path rejection.

Deterministic recurrences and window memory are never proposed. Each direction
linearizes at the guide's exact replay of its own independent coordinates, the
Gaussian carries those coordinates as noise-free affine rows, and its density
reads only the free coordinates. Every label then replays them exactly.

StudentT readings condition on the sweep's transient precisions: the Gaussian
carries their exact conditional curvature, and every density here is the
conditional target. MPGibbs discards the precisions before the final label draw.
Ambiguous outliers switch only through precision updates.

Gaussian approximations to nonlinear dynamics, non-Gaussian observations, long
memory or separated label modes can cause rejection. They never replace the target.
"""

from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from grassmoth.models.ssm.inference.samplers.particle.blocked_kalman import blocked_kalman_step
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
    TrajectoryLabelScores,
)
from grassmoth.models.ssm.inference.samplers.particle.kalman import (
    GaussianPathProposal,
    GaussianPathTerms,
    gaussian_path_proposal,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist
from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from grassmoth.models.ssm.inference.tracing import AuxiliaryKalmanTraceRecord, TracedResult
from grassmoth.models.ssm.inference.warmup.path_curvature import linearize_path


class AuxiliaryKalmanMove(NamedTuple):
    """One proposed path and the exact conditional-target MH calculation."""

    proposed: jax.Array
    proposal_scores: TrajectoryLabelScores
    reference_scores: TrajectoryLabelScores
    log_acceptance: jax.Array


def _proposal(
    target: ParticleTarget,
    guide: ParticleContext,
    reference: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    free: jax.Array,
    log_precisions: jax.Array | None = None,
) -> GaussianPathProposal:
    path = target.replay_path(guide, reference)
    curvature = linearize_path(target, guide, path, log_precisions)
    terms = curvature.proposal_terms(target, guide, path)
    local = terms.information
    free = curvature.augment(free, fill=False)
    variance = curvature.augment(variance, fill=1.0)
    information = GaussianTwist(
        local.precision + jax.vmap(jnp.diag)(free / variance),
        local.shift + jnp.where(free, curvature.augment(auxiliary, fill=0.0) / variance, 0.0),
    )
    return gaussian_path_proposal(
        GaussianPathTerms(terms.matrices, terms.offsets, terms.covariances, information), free
    )


def auxiliary_kalman_move(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    reference: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    sampling_key: jax.Array,
    log_precisions: jax.Array | None = None,
) -> AuxiliaryKalmanMove:
    """Propose and score both directions at the same auxiliary and label ensemble."""
    free = ctx.latent_free_mask
    with jax.named_scope("kalman_forward_proposal"):
        forward = _proposal(target, guide, reference, auxiliary, variance, free, log_precisions)
        draw, forward_density = forward.sample_with_log_density(sampling_key)
        proposed = draw[:, : reference.shape[-1]]
    with jax.named_scope("kalman_reverse_proposal"):
        reverse = _proposal(target, guide, proposed, auxiliary, variance, free, log_precisions)
        # Memory rows are replayed by the reverse proposal itself; their entries are unread.
        reverse_density = reverse.log_density(
            jnp.concatenate((reference, jnp.zeros_like(draw[:, reference.shape[-1] :])), axis=-1)
        )
    with jax.named_scope("kalman_exact_correction"):
        proposal_scores = ctx.score_labels(proposed)
        reference_scores = ctx.score_labels(reference)
        target_ratio = logsumexp(
            ctx.initial_label_log_probs + proposal_scores.trajectory_log_probs
        ) - logsumexp(ctx.initial_label_log_probs + reference_scores.trajectory_log_probs)
        auxiliary_ratio = -0.5 * jnp.sum(
            jnp.where(
                free, ((proposed - auxiliary) ** 2 - (reference - auxiliary) ** 2) / variance, 0.0
            )
        )
        log_acceptance = jnp.minimum(
            0.0, target_ratio + auxiliary_ratio + reverse_density - forward_density
        )
    return AuxiliaryKalmanMove(proposed, proposal_scores, reference_scores, log_acceptance)


def _auxiliary_kalman_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    key: jax.Array,
    reference: jax.Array,
    *,
    record_trace: bool,
    block_length: int | None,
    log_precisions: jax.Array | None = None,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, AuxiliaryKalmanTraceRecord]:
    """Return an exact path update with the existing marginal parameter-label scores."""
    if block_length is not None:
        blocked = blocked_kalman_step(
            target, ctx, key, reference, block_length=block_length, log_precisions=log_precisions
        )
        return blocked if record_trace else blocked.result
    auxiliary_key, sampling_key, acceptance_key = jax.random.split(key, 3)
    free = ctx.latent_free_mask
    variance = jnp.broadcast_to(ctx.amala_delta.reshape((ctx.num_steps, -1)), reference.shape) / 2
    auxiliary = reference + jnp.where(
        free,
        jnp.sqrt(variance) * jax.random.normal(auxiliary_key, reference.shape, reference.dtype),
        0.0,
    )
    move = auxiliary_kalman_move(
        target, ctx, guide, reference, auxiliary, variance, sampling_key, log_precisions
    )
    accepted = (
        jnp.log(jax.random.uniform(acceptance_key, dtype=reference.dtype)) < move.log_acceptance
    )
    result = ParticleSmootherResult(
        jnp.where(accepted, move.proposed, reference),
        jax.tree.map(
            lambda proposed, old: jnp.where(accepted, proposed, old),
            move.proposal_scores,
            move.reference_scores,
        ),
        jnp.full((ctx.num_steps,), accepted, dtype=jnp.int32),
        free,
        MappingProxyType(
            {
                "amala_grad_norm_mean": jnp.asarray(0.0, dtype=ctx.traj_dtype),
                "amala_grad_norm_max": jnp.asarray(0.0, dtype=ctx.traj_dtype),
                "path_block_acceptance": accepted.astype(ctx.traj_dtype),
                "path_update_rms": jnp.sqrt(
                    jnp.mean(jnp.where(accepted & free, (move.proposed - reference) ** 2, 0.0))
                ),
            }
        ),
    )
    if record_trace:
        return TracedResult(
            result,
            AuxiliaryKalmanTraceRecord(auxiliary, move.proposed, move.log_acceptance, accepted),
        )
    return result


def auxiliary_kalman_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    key: jax.Array,
    reference: jax.Array,
    *,
    block_length: int | None = None,
    log_precisions: jax.Array | None = None,
) -> ParticleSmootherResult:
    """Perform the Gaussian path update without retaining proposal internals."""
    result = _auxiliary_kalman_step(
        target,
        ctx,
        guide,
        key,
        reference,
        record_trace=False,
        block_length=block_length,
        log_precisions=log_precisions,
    )
    assert not isinstance(result, TracedResult)
    return result


def trace_auxiliary_kalman_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    key: jax.Array,
    reference: jax.Array,
    *,
    block_length: int | None = None,
    log_precisions: jax.Array | None = None,
) -> TracedResult[ParticleSmootherResult, AuxiliaryKalmanTraceRecord]:
    """Expose the proposal and MH decision from the same numerical transition."""
    result = _auxiliary_kalman_step(
        target,
        ctx,
        guide,
        key,
        reference,
        record_trace=True,
        block_length=block_length,
        log_precisions=log_precisions,
    )
    assert isinstance(result, TracedResult)
    return result
