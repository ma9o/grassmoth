"""Center paths of scaled-deviation charts, learned from warmup windows.

Each window's mean translated cold-chain path becomes the frozen center at the
window's end. Scaling deviations from this center leaves the path's data-pinned
shape in place, so a diffusion-scale label rescales mainly the diffusion-driven
deviations. Any frozen center gives the exact target; a poor one costs mixing.
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.targets.coordinates import ScaleChart
from grassmoth.models.ssm.shapes import Array, Float, Int


class CenterMoments(NamedTuple):
    """Running sum of chain-averaged translated paths within the current window."""

    sample_count: Int[Array, ""]
    path_sum: Float[Array, "T D"]
    position_sum: Float[Array, " U"]


def initialize_center_moments(
    positions: Float[Array, "C U"], paths: Float[Array, "C T D"]
) -> CenterMoments:
    """Start an empty accumulator shaped like one path."""
    return CenterMoments(
        jnp.asarray(0),
        jnp.zeros(paths.shape[1:], dtype=paths.dtype),
        jnp.zeros(positions.shape[1:], dtype=positions.dtype),
    )


def adapt_center(
    moments: CenterMoments,
    chart: ScaleChart,
    positions: Float[Array, "C U"],
    paths: Float[Array, "C T D"],
    iteration: Int[Array, ""],
    windows: Int[Array, "W 2"],
) -> tuple[CenterMoments, ScaleChart]:
    """Accumulate translated physical paths; replace the center only at window ends."""
    active = jnp.any((iteration >= windows[:, 0]) & (iteration <= windows[:, 1]))
    boundary = jnp.any(iteration == windows[:, 1])

    def accumulate() -> tuple[CenterMoments, ScaleChart]:
        updated = CenterMoments(
            moments.sample_count + 1,
            moments.path_sum + jnp.mean(paths, axis=0),
            moments.position_sum + jnp.mean(positions, axis=0),
        )

        def finish() -> tuple[CenterMoments, ScaleChart]:
            return (
                CenterMoments(
                    jnp.zeros_like(updated.sample_count),
                    jnp.zeros_like(updated.path_sum),
                    jnp.zeros_like(updated.position_sum),
                ),
                ScaleChart(
                    chart.translation,
                    chart.exponents,
                    chart.translation.encode(
                        updated.position_sum / updated.sample_count,
                        updated.path_sum / updated.sample_count,
                    ),
                ),
            )

        return cast(
            "tuple[CenterMoments, ScaleChart]",
            jax.lax.cond(boundary, finish, lambda: (updated, chart)),
        )

    return cast(
        "tuple[CenterMoments, ScaleChart]",
        jax.lax.cond(active, accumulate, lambda: (moments, chart)),
    )
