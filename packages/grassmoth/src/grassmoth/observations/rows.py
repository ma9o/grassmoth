"""Numerical observation grids and calendar boundaries for measurement windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.observations.time import model_time_indices

if TYPE_CHECKING:
    from grassmoth.artifacts.observations.dataset import SelectedObservations


def pivot_to_wide(observations: SelectedObservations, grid: np.ndarray) -> pl.DataFrame:
    """Place admitted readings on their shared model-day grid without timestamp parsing."""
    columns = {}
    for series in observations.series:
        values = np.full(len(grid), np.nan)
        indices = model_time_indices(grid, series.history.times.values)
        values[indices] = vector_values(series.history.values)
        columns[str(series.observation.id)] = pl.Series(values).fill_nan(None)
    return pl.DataFrame({"time": grid, **columns})
