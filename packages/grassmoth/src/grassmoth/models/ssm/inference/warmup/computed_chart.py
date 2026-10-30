"""Computed path translations from the shared positive Gauss--Newton curvature.

Rebuilding belongs only to warmup boundaries. A local slope can mix poorly under
strong nonlinearity or multiple modes; every frozen chart keeps the exact target.
"""

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.ieks import ieks_path
from grassmoth.models.ssm.inference.warmup.path_curvature import (
    interval_translation,
    linearize_path,
)
from grassmoth.models.ssm.shapes import Array, Float


def computed_translation_slopes(
    target: ParticleTarget, position: Float[Array, " U"], path: Float[Array, "T D"]
) -> Float[Array, "T D U"]:
    """Solve the mean cross curvature, including deterministic stages and intervals."""
    context = target.context(position, target.times)
    curvature = linearize_path(target, context, path)
    slopes = jax.jacfwd(
        lambda value: curvature.mean(target, target.context(value, target.times), path)
    )(position)
    if target.intervals is not None:
        slopes = interval_translation(target, position, path, curvature, slopes)
    return jnp.asarray(slopes)


def computed_chart(
    target: ParticleTarget, positions: Float[Array, "B U"], paths: Float[Array, "B T D"]
) -> TranslationChart:
    """Build at mean cold parameters and their IEKS path, retaining only free shifts."""
    reference = jnp.mean(positions, axis=0)
    context = target.context(reference, target.times)
    path = ieks_path(target, context, target.replay_path(context, jnp.mean(paths, axis=0)))
    slopes = computed_translation_slopes(target, reference, path)
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    return TranslationChart(jnp.where(free[..., None], slopes, 0.0), reference)
