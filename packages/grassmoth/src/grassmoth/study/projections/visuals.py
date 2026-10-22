"""Lossless reads of recorded data and draws for the model workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from grassmoth.artifacts.arrays import ArrayVector
from grassmoth.artifacts.observations.history import ObservationHistory
from grassmoth.artifacts.observations.time import ObservationInstant
from grassmoth.artifacts.reports.histograms import empirical_points

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.artifacts.observations.history import ObservationData
    from grassmoth.artifacts.simulation import SimulationEvidence


def finite_values(values: np.ndarray) -> tuple[float | None, ...]:
    """Convert numerical values to JSON-safe scalars, representing non-finite entries as ``None``."""
    return tuple(float(value) if np.isfinite(value) else None for value in values)


def observation_history(time_origin: datetime, panel: pl.DataFrame) -> ObservationHistory:
    """Project recorded rows into model-day coordinates with their support intervals and empirical CDF."""
    origin = ObservationInstant(time_origin)

    def days(column: str) -> tuple[float | None, ...]:
        return tuple(
            ObservationInstant(value).relative_to(origin).days if value is not None else None
            for value in panel[column]
        )

    values = panel["value"].cast(pl.Float64).to_numpy()
    return ObservationHistory(
        times=tuple(
            ObservationInstant(value).relative_to(origin).days for value in panel["anchor_time"]
        ),
        values=finite_values(values),
        support_start=days("support_start"),
        support_end=days("support_end"),
        empirical=empirical_points(values),
    )


def simulation_observation_histories(
    evidence: SimulationEvidence,
    observations: np.ndarray,
    mask: np.ndarray,
) -> tuple[ObservationData, ...]:
    """Project every saved replicate into the same history type as prepared user data."""
    return tuple(
        {
            variable.id: ObservationHistory(
                times=evidence.times,
                values=ArrayVector(
                    array=evidence.arms.action.observations,
                    indices=(replicate, None, column),
                    mask=ArrayVector(
                        array=evidence.observation_layout.mask,
                        indices=(replicate, None, column),
                    ),
                ),
                support_start=ArrayVector(
                    array=evidence.observation_layout.support_start_times,
                    indices=(None, column),
                ),
                support_end=ArrayVector(
                    array=evidence.observation_layout.support_end_times,
                    indices=(None, column),
                ),
                empirical=empirical_points(values),
            )
            for column, variable in enumerate(evidence.observation_layout.variables)
            for values in (
                np.where(mask[replicate, :, column], observations[replicate, :, column], np.nan),
            )
        }
        for replicate in range(evidence.draws)
    )
