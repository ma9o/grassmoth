"""Shared categorical weight measurements and their persisted DSMC projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.reports.posterior_diagnostics import (
    DsmcNodeWeightDiagnostics,
    DsmcWeightDiagnostics,
    ParticleWeightDiagnostics,
    ParticleWeightMetric,
    ParticleWeightSummary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.artifacts.reports.posterior_diagnostics import DiagnosticPhase


def weight_statistics(logits: jax.Array) -> dict[ParticleWeightMetric, jax.Array]:
    """Reduce the last candidate axis, preserving all leading node/batch axes.

    Negative infinity is a zero-weight candidate. Finite log-weight moments are
    centered before reduction so additive offsets cannot inflate their variance.
    Undefined distributions produce nonfinite measurements, never uniform weights.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    ess = jnp.exp(-jax.scipy.special.logsumexp(2 * log_probs, axis=-1))
    finite = jnp.isfinite(logits)
    finite_count = jnp.sum(finite, axis=-1)
    maximum = jnp.max(jnp.where(finite, logits, -jnp.inf), axis=-1)
    minimum = jnp.min(jnp.where(finite, logits, jnp.inf), axis=-1)
    centered = jnp.where(finite, logits - maximum[..., None], 0.0)
    mean = jnp.sum(centered, axis=-1) / finite_count
    variance = (
        jnp.sum(jnp.where(finite, (centered - mean[..., None]) ** 2, 0.0), axis=-1) / finite_count
    )
    return {
        ParticleWeightMetric.ESS: ess,
        ParticleWeightMetric.ESS_FRACTION: ess / logits.shape[-1],
        ParticleWeightMetric.ENTROPY: -jnp.sum(
            jnp.where(jnp.isneginf(log_probs), 0.0, probabilities * log_probs), axis=-1
        ),
        ParticleWeightMetric.MAX_PROBABILITY: jnp.max(probabilities, axis=-1),
        ParticleWeightMetric.REFERENCE_PROBABILITY: probabilities[..., 0],
        ParticleWeightMetric.FINITE_LOG_WEIGHT_RANGE: maximum - minimum,
        ParticleWeightMetric.FINITE_LOG_WEIGHT_VARIANCE: variance,
        ParticleWeightMetric.FINITE_FRACTION: finite_count / logits.shape[-1],
    }


def weight_history_fields(
    stages: tuple[Mapping[ParticleWeightMetric, jax.Array], ...],
) -> dict[str, jax.Array]:
    """Flatten level-ordered nodes and terminal selection into shared history fields."""
    return {
        metric.history_key: jnp.concatenate(tuple(stage[metric] for stage in stages))
        for metric in ParticleWeightMetric
    }


def summarize_dsmc_weights(
    history: Mapping[str, jax.Array],
    *,
    num_steps: int,
    num_particles: int,
    phase: DiagnosticPhase,
) -> DsmcWeightDiagnostics:
    """Project native chain/sweep/node histories once for both particle sampler families."""

    def finite_scalar(value: np.floating) -> float | None:
        return float(value) if np.isfinite(value) else None

    def summarize(values: np.ndarray) -> tuple[ParticleWeightSummary, ...]:
        means = values.mean(axis=(0, 1))
        minima = values.min(axis=(0, 1))
        maxima = values.max(axis=(0, 1))
        nonfinite = (~np.isfinite(values)).mean(axis=(0, 1))
        return tuple(
            ParticleWeightSummary(
                mean=finite_scalar(mean),
                minimum=finite_scalar(minimum),
                maximum=finite_scalar(maximum),
                nonfinite_fraction=float(missing),
            )
            for mean, minimum, maximum, missing in zip(
                means, minima, maxima, nonfinite, strict=True
            )
        )

    summaries = {
        metric: summarize(np.asarray(history[metric.history_key]))
        for metric in ParticleWeightMetric
    }

    def weights(index: int, candidates: int) -> ParticleWeightDiagnostics:
        return ParticleWeightDiagnostics(
            candidates=candidates,
            metrics={metric: values[index] for metric, values in summaries.items()},
        )

    nodes = tuple(
        (level, start, min(start + (1 << level), num_steps))
        for level in range((num_steps - 1).bit_length() + 1)
        for start in range(0, num_steps, 1 << level)
    )
    return DsmcWeightDiagnostics(
        phase=phase,
        tree=tuple(
            DsmcNodeWeightDiagnostics(
                level=level,
                start=start,
                stop=stop,
                weights=weights(index, num_particles if level == 0 else num_particles**2),
            )
            for index, (level, start, stop) in enumerate(nodes)
        ),
        terminal=weights(len(nodes), num_particles),
    )
