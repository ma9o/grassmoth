"""Numerical observation grids and calendar boundaries for measurement windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.observations.duration import CalendarDuration
from grassmoth.artifacts.observations.time import merge_model_times, model_time_indices

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from grassmoth.artifacts.observations.dataset import SelectedObservations
    from grassmoth.artifacts.observations.spec import (
        ObservationWindow,
    )


def observation_window_boundaries(
    window: ObservationWindow, start: datetime, end: datetime
) -> tuple[datetime, ...]:
    """Calendar boundaries that a simulation grid must retain inside its requested span."""
    if not isinstance(window, CalendarDuration):
        return ()
    first = pl.Series([start]).dt.truncate(window.polars_interval)[0]
    instants: list[datetime] = pl.datetime_range(
        first, end, interval=window.polars_interval, eager=True
    ).to_list()
    return tuple(instant for instant in instants if instant >= start)


def observation_grid(
    observations: SelectedObservations, extra_times: Sequence[float] = ()
) -> np.ndarray:
    """Retain anchors, interval boundaries, initial day zero and supplied input events."""
    anchors = [
        float(time) for series in observations.series for time in series.history.times.values
    ]
    boundaries = [
        float(time)
        for series in observations.series
        if series.observation.support_kind == "interval"
        for bounds in (series.history.support_start, series.history.support_end)
        for time in vector_values(bounds)
        if np.isfinite(time)
    ]
    initial = [0.0] if anchors and min(anchors) > 0 else []
    return merge_model_times([*anchors, *boundaries, *initial, *extra_times])


def pivot_to_wide(observations: SelectedObservations, grid: np.ndarray) -> pl.DataFrame:
    """Place admitted readings on their shared model-day grid without timestamp parsing."""
    columns = {}
    for series in observations.series:
        values = np.full(len(grid), np.nan)
        indices = model_time_indices(grid, series.history.times.values)
        values[indices] = vector_values(series.history.values)
        columns[str(series.observation.id)] = pl.Series(values).fill_nan(None)
    return pl.DataFrame({"time": grid, **columns})
