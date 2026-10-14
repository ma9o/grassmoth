"""Pure read projections of retained simulation evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.reports.display_frames import central_frame
from nof1_causal_lab.artifacts.simulation import CategoryProbabilitySummary, SimulationSummary

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.simulation import SimulationEvidence


def category_probabilities(
    values: np.ndarray, mask: np.ndarray, levels: tuple[str, ...]
) -> CategoryProbabilitySummary:
    """Keep the full categorical distribution and per-time effective draw count."""
    present = mask & np.isfinite(values)
    selected = tuple(samples[keep] for samples, keep in zip(values.T, present.T, strict=True))
    return CategoryProbabilitySummary(
        probabilities={
            level: tuple(
                float(np.mean(samples == index)) if samples.size else None for samples in selected
            )
            for index, level in enumerate(levels)
        },
        n_draws=tuple(int(count) for count in present.sum(axis=0)),
    )


def simulation_summary(
    evidence: SimulationEvidence,
    latent: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray | None,
    reference_observed: np.ndarray | None,
) -> SimulationSummary:
    """Reduce the full saved draws once; a chart's page never changes its summaries."""
    variables = evidence.observation_layout.variables
    category_levels = {
        index: ("0", "1")
        if variable.measurement_dtype == "binary"
        else variable.ordinal_levels or variable.categorical_levels
        for index, variable in enumerate(variables)
    }
    states = np.concatenate((latent, reference)) if reference is not None else latent
    observations = (
        np.concatenate((observed, reference_observed))
        if reference_observed is not None
        else observed
    )
    observation_mask = np.concatenate((mask, mask)) if reference_observed is not None else mask
    observations = np.where(observation_mask, observations, np.nan)
    return SimulationSummary(
        state_frames={
            identity: frame
            for index, identity in enumerate(evidence.state_ids)
            if (frame := central_frame(states[:, :, index])) is not None
        },
        indicator_frames={
            variable.id: frame
            for index, variable in enumerate(variables)
            if (frame := central_frame(observations[:, :, index])) is not None
        },
        action_category_probabilities={
            variables[index].id: category_probabilities(
                observed[:, :, index], mask[:, :, index], levels
            )
            for index, levels in category_levels.items()
            if levels is not None
        },
        reference_category_probabilities={
            variables[index].id: category_probabilities(
                reference_observed[:, :, index], mask[:, :, index], levels
            )
            for index, levels in category_levels.items()
            if levels is not None
        }
        if reference_observed is not None
        else {},
    )
