"""Backend binning for reported sample distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.reports.effects import HistogramBin
from grassmoth.artifacts.reports.empirical import EmpiricalDistribution

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def histogram_draws(draws: ArrayLike, *, max_bins: int = 25) -> list[HistogramBin]:
    """Bin finite scalar samples into equal-width, inclusive-final-edge bins."""
    values = np.asarray(draws)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Histogram draws must be a nonempty finite vector")
    counts, edges = np.histogram(values, bins=min(max_bins, int(np.ceil(np.sqrt(values.size)))))
    return [
        HistogramBin(
            bin_start=float(left),
            bin_end=float(right),
            bin_center=float((left + right) / 2),
            count=int(count),
        )
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]


def empirical_distribution(values: np.ndarray) -> EmpiricalDistribution:
    """Compute the empirical CDF at distinct finite values, retaining duplicate frequencies."""
    unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum() if counts.size else np.empty(0)
    return EmpiricalDistribution(
        values=NumericalArray.from_numpy(unique),
        probabilities=NumericalArray.from_numpy(cumulative),
    )
