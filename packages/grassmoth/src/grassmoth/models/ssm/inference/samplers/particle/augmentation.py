"""Transient Student-t precision conditioning shared by all exact path kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from grassmoth.artifacts.sampling import (
    FixedLatentStepSpec,
    GuidedMethodSpec,
    PathUpdateSpec,
)
from grassmoth.models.ssm.inference.samplers.particle.adaptation import DELTA_MAX
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.targets.student_precision import draw_student_precisions

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.samplers.particle.contract import (
        SmootherContext,
        SmootherSetup,
    )
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
    from grassmoth.models.ssm.inference.targets.student_precision import StudentTPrecisions


@dataclass(frozen=True)
class MarginalPathUpdate:
    """A path update with no precision augmentation."""

    context: SmootherContext
    key: jax.Array


@dataclass(frozen=True)
class PrecisionPathUpdate:
    """A path update conditioned on freshly drawn reading precisions."""

    context: SmootherContext
    key: jax.Array
    precisions: StudentTPrecisions


type PathUpdateBoundary = MarginalPathUpdate | PrecisionPathUpdate


def begin_path_update(
    target: ParticleTarget,
    static: SmootherSetup,
    path_update: PathUpdateSpec,
    context: ParticleContext,
    reference: jax.Array,
    delta: jax.Array,
    parameter_particles: jax.Array,
    label_correction: jax.Array,
    block_key: jax.Array,
) -> PathUpdateBoundary:
    """Draw lambda after the parameter proposal and construct its conditional target.

    The transient cap is min(controller * frozen_metric, precision_limit); the
    caller has already formed the product. Fixed, window and replay geometry
    keep their existing behavior. No augmentation enters the persisted state.
    """
    precision_key, conditional_block_key = random.split(block_key)
    precisions = draw_student_precisions(target, context, reference, precision_key)
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
        reference=reference,
        delta=delta,
        trajectory_dtype=reference.dtype,
        label_factor=None
        if precisions is None
        else lambda context: precisions.log_prior(target, context),
    )
    if (
        precisions is not None
        and isinstance(path_update.method, GuidedMethodSpec)
        and path_update.method.kappa == 1
        and not isinstance(path_update.latent_step, FixedLatentStepSpec)
    ):
        limit = precisions.step_limit(target, ctx.contexts, static.latent_free_mask, DELTA_MAX)
        ctx = ctx._replace(amala_delta=jnp.minimum(delta.reshape((len(reference), -1)), limit))
    if precisions is None:
        return MarginalPathUpdate(ctx, block_key)
    return PrecisionPathUpdate(ctx, block_key, precisions)
