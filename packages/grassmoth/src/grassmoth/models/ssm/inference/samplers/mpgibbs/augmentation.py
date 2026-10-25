"""The ordered, partially collapsed precision-augmentation boundary of one sweep.

The caller proposes parameter labels before entering this boundary, independently
of lambda. It draws precisions, conditions the path update, then discards lambda
and rescores marginally before the final label draw and cache update. Conditional
path scores cannot be passed as marginal label/cache scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from grassmoth.artifacts.sampling import DsmcSpec, FixedLatentStepSpec
from grassmoth.models.ssm.inference.samplers.particle.adaptation import DELTA_MAX
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.math import _normalize_log_probs
from grassmoth.models.ssm.inference.targets.student_precision import draw_student_precisions

if TYPE_CHECKING:
    from grassmoth.artifacts.sampling import PathUpdateSpec
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCState
    from grassmoth.models.ssm.inference.samplers.particle.contract import (
        SmootherContext,
        SmootherSetup,
        TrajectoryLabelScores,
    )
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.inference.targets.student_precision import StudentTPrecisions


@dataclass(frozen=True)
class MarginalPathUpdate:
    """A path update with no precision augmentation (including dSMC)."""

    context: SmootherContext
    key: jax.Array


@dataclass(frozen=True)
class PrecisionPathUpdate:
    """A path update conditioned on freshly drawn reading precisions."""

    context: SmootherContext
    key: jax.Array
    precisions: StudentTPrecisions


type PathUpdateBoundary = MarginalPathUpdate | PrecisionPathUpdate


@dataclass(frozen=True)
class ConditionalPathScores:
    """Scores paid by the path update, possibly conditional on precisions."""

    values: TrajectoryLabelScores


class MarginalLabelScores(NamedTuple):
    """Marginal target scores admitted to the final label draw and state caches."""

    log_probs: jax.Array
    trajectory_log_probs: jax.Array
    log_posteriors: jax.Array


def begin_path_update(
    target: ParticleTarget,
    static: SmootherSetup,
    path_update: PathUpdateSpec,
    state: TrajectoryMCMCState,
    parameter_particles: jax.Array,
    label_correction: jax.Array,
    block_key: jax.Array,
) -> PathUpdateBoundary:
    """Draw lambda after the parameter proposal and construct its conditional target.

    The transient cap is min(controller * frozen_metric, precision_limit); the
    caller has already formed the product. Fixed, window and replay geometry
    keep their existing behavior. No augmentation enters the persisted state.
    """
    precisions = None
    if not isinstance(path_update, DsmcSpec):
        precision_key, conditional_block_key = random.split(block_key)
        precisions = draw_student_precisions(
            target, state.latent_context, state.latent_trajectory, precision_key
        )
        if precisions is not None:
            block_key = conditional_block_key
    conditional_static = (
        static
        if precisions is None
        else static._replace(
            trajectory_log_prob_fn=lambda context, path, _observations: precisions.path_log_prob(
                target, context, path
            )
        )
    )
    ctx = build_smoother_context(
        conditional_static,
        parameter_particles,
        label_correction,
        reference=state.latent_trajectory,
        delta=state.latent_delta,
        trajectory_dtype=state.trajectory_log_prob.dtype,
        label_factor=None
        if precisions is None
        else lambda context: precisions.log_prior(target, context),
    )
    if (
        precisions is not None
        and not isinstance(path_update, DsmcSpec)
        and path_update.kappa == 1
        and not isinstance(path_update.latent_step, FixedLatentStepSpec)
    ):
        limit = precisions.step_limit(target, ctx.contexts, static.latent_free_mask, DELTA_MAX)
        ctx = ctx._replace(
            amala_delta=jnp.minimum(
                state.latent_delta.reshape((len(state.latent_trajectory), -1)), limit
            )
        )
    if precisions is None:
        return MarginalPathUpdate(ctx, block_key)
    return PrecisionPathUpdate(ctx, block_key, precisions)


def finish_path_update(
    boundary: PathUpdateBoundary,
    conditional: ConditionalPathScores,
    target: ParticleTarget,
    parameter_particles: jax.Array,
    label_correction: jax.Array,
    latent_path: jax.Array,
) -> MarginalLabelScores:
    """Discard lambda before selecting a label; only marginal scores may be cached."""
    if isinstance(boundary, MarginalPathUpdate):
        return MarginalLabelScores(*conditional.values)
    trajectory = jax.vmap(
        lambda context: target.path_log_prob(context, latent_path, target.observations)
    )(boundary.context.contexts)
    posterior = jax.vmap(target.log_prior)(parameter_particles) + trajectory
    return MarginalLabelScores(
        _normalize_log_probs(posterior + label_correction), trajectory, posterior
    )
