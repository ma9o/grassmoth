"""Sparse, time-constant translations learned from within-chain warmup moments.

The least-squares slope for a common translation at every time is determined by
the covariance of parameters with the time-averaged physical path. Centering
each chain separately excludes between-chain offsets. Five observations of the
initial marginal parameter scale shrink slopes toward zero. The graph mask is
applied to each regression's normal equations, not to a dense solution afterward.
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int


class TranslationMoments(NamedTuple):
    """Window moments in diagonally standardized parameter coordinates."""

    parameter_scale: Float[Array, " U"]
    sample_count: Int[Array, ""]
    parameter_mean: Float[Array, "C U"]
    path_mean: Float[Array, "C D"]
    parameter_m2: Float[Array, "C U U"]
    cross_m2: Float[Array, "C D U"]


def initialize_translation_moments(
    positions: Float[Array, "C U"],
    paths: Float[Array, "C T D"],
    parameter_chol: Float[Array, "U U"],
) -> TranslationMoments:
    """Own one accumulator per cold chain and fixed marginal parameter scales."""
    chains, parameters = positions.shape
    latents = paths.shape[-1]
    return TranslationMoments(
        parameter_scale=jnp.sqrt(jnp.sum(parameter_chol**2, axis=1)),
        sample_count=jnp.asarray(0),
        parameter_mean=jnp.zeros_like(positions),
        path_mean=jnp.zeros((chains, latents), dtype=paths.dtype),
        parameter_m2=jnp.zeros((chains, parameters, parameters), dtype=positions.dtype),
        cross_m2=jnp.zeros((chains, latents, parameters), dtype=positions.dtype),
    )


def adapt_translation(
    moments: TranslationMoments,
    chart: TranslationChart,
    positions: Float[Array, "C U"],
    paths: Float[Array, "C T D"],
    active_pairs: Bool[Array, "D U"],
    iteration: Int[Array, ""],
    windows: Int[Array, "W 2"],
) -> tuple[TranslationMoments, TranslationChart]:
    """Accumulate selected physical states; change the chart only at window ends."""
    active = jnp.any((iteration >= windows[:, 0]) & (iteration <= windows[:, 1]))
    boundary = jnp.any(iteration == windows[:, 1])

    def accumulate() -> tuple[TranslationMoments, TranslationChart]:
        parameters = (positions - chart.reference) / moments.parameter_scale
        path = jnp.mean(paths, axis=1)
        count = moments.sample_count + 1
        parameter_delta = parameters - moments.parameter_mean
        path_delta = path - moments.path_mean
        parameter_mean = moments.parameter_mean + parameter_delta / count
        path_mean = moments.path_mean + path_delta / count
        parameter_m2 = moments.parameter_m2 + jnp.einsum(
            "ci,cj->cij", parameter_delta, parameters - parameter_mean
        )
        cross_m2 = moments.cross_m2 + jnp.einsum(
            "cd,cu->cdu", path_delta, parameters - parameter_mean
        )
        updated = moments._replace(
            sample_count=count,
            parameter_mean=parameter_mean,
            path_mean=path_mean,
            parameter_m2=parameter_m2,
            cross_m2=cross_m2,
        )

        def finish() -> tuple[TranslationMoments, TranslationChart]:
            covariance = jnp.mean(parameter_m2, axis=0)
            covariance = (covariance + covariance.T) / 2
            cross = jnp.mean(cross_m2, axis=0)
            mask = active_pairs.astype(covariance.dtype)
            normal = covariance[None] * mask[:, :, None] * mask[:, None, :]
            normal = normal + 5.0 * jnp.eye(positions.shape[-1], dtype=positions.dtype)[None]
            slopes = jnp.linalg.solve(normal, (cross * mask)[..., None])[..., 0]
            slopes = slopes / moments.parameter_scale
            return (
                updated._replace(
                    sample_count=jnp.zeros_like(count),
                    parameter_mean=jnp.zeros_like(parameter_mean),
                    path_mean=jnp.zeros_like(path_mean),
                    parameter_m2=jnp.zeros_like(parameter_m2),
                    cross_m2=jnp.zeros_like(cross_m2),
                ),
                TranslationChart(slopes, chart.reference),
            )

        return cast(
            "tuple[TranslationMoments, TranslationChart]",
            jax.lax.cond(boundary, finish, lambda: (updated, chart)),
        )

    return cast(
        "tuple[TranslationMoments, TranslationChart]",
        jax.lax.cond(active, accumulate, lambda: (moments, chart)),
    )
