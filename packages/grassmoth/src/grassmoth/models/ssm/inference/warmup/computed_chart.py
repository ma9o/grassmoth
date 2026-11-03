"""Unified charts from one frozen Gaussian surrogate at an IEKS reference.

Only warmup boundaries rebuild the chart. Frozen linearizations shape mixing;
all sampling densities remain exact physical densities with chart Jacobians.
"""

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.targets.coordinates import PathChart
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.ieks import ieks_path
from grassmoth.models.ssm.inference.warmup.path_curvature import PathCurvature, linearize_path
from grassmoth.models.ssm.shapes import Array, Float


def _mean_response(
    target: ParticleTarget, curvature: PathCurvature, reference: jax.Array, path: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Solve m and H⁻¹(b′−H′m), with all parameter directions batched together.

    JAX differentiates the Gaussian linear solves implicitly: their reference
    factorizations serve the batched tangent right-hand sides. The closed-over
    geometry was constructed before differentiation, including PSD clipping.
    """

    def mean(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        context = target.context(value, target.times)
        result = curvature.with_transition_noise(target, context, path).mean(target, context, path)
        return result, result

    slopes, center = jax.jacfwd(mean, has_aux=True)(reference)
    return jnp.asarray(center), jnp.asarray(slopes)


def computed_chart(
    target: ParticleTarget, positions: Float[Array, "B U"], paths: Float[Array, "B T D"]
) -> PathChart:
    """Build c, B and structural A together at mean parameters and their IEKS path."""
    reference = jnp.mean(positions, axis=0)
    if reference.size == 0:
        coefficients = jnp.zeros((*paths.shape[1:], 0), dtype=paths.dtype)
        return PathChart(coefficients, reference, coefficients, jnp.zeros_like(paths[0]))
    context = target.context(reference, target.times)
    path = ieks_path(target, context, target.replay_path(context, jnp.mean(paths, axis=0)))
    curvature = linearize_path(target, context, path)
    center, slopes = _mean_response(target, curvature, reference, path)
    exponents = jnp.broadcast_to(target.diffusion_scale_exponents(reference), slopes.shape)
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    return PathChart(
        jnp.where(free[..., None], slopes, 0.0),
        reference,
        jnp.where(free[..., None], exponents, 0.0),
        jnp.where(free, center, 0.0),
    )
