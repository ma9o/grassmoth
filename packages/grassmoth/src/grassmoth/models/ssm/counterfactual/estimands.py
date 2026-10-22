"""Runtime estimand summaries over posterior effect draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from grassmoth.artifacts.reports.effects import EffectSummary

if TYPE_CHECKING:
    from jax import Array


def summarize_draws(draws: Array) -> EffectSummary:
    """Posterior summary statistics for a 1-D array of effect draws."""
    if draws.ndim != 1 or draws.size == 0:
        raise ValueError("An effect summary requires a nonempty vector of posterior draws")
    return EffectSummary(
        mean=float(jnp.mean(draws)),
        median=float(jnp.median(draws)),
        lower_95=float(jnp.quantile(draws, 0.025)),
        upper_95=float(jnp.quantile(draws, 0.975)),
        prob_positive=float(jnp.mean(draws > 0)),
    )
