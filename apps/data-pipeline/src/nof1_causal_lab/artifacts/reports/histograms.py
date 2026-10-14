"""Backend binning for reported sample distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.reports.effects import HistogramBin
from nof1_causal_lab.artifacts.reports.empirical import EmpiricalPoint

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


def empirical_points(values: np.ndarray) -> tuple[EmpiricalPoint, ...]:
    """Compute the empirical CDF at distinct finite values, retaining duplicate frequencies."""
    unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum() if counts.size else []
    return tuple(
        EmpiricalPoint(
            value=value,
            probability=float(probability),
        )
        for value, probability in zip(unique, cumulative, strict=True)
    )
