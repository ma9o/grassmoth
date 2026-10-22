"""The value range a chart of draws shows, reduced on the server like every other summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def central_frame(
    rows: Iterable[Sequence[float | None]], cover: Iterable[float | None] = ()
) -> tuple[float, float] | None:
    """Widest per-position central 95% of aligned rows, widened to cover the given values.

    Draws beyond the frame leave the chart, which says how many. Positions with no finite
    value are ignored; nothing finite at all frames nothing.
    """
    bounds: list[float] = []
    matrix = np.asarray(
        [[np.nan if value is None else value for value in row] for row in rows], dtype=float
    )
    if matrix.ndim == 2 and matrix.size:
        columns = matrix[:, np.isfinite(matrix).any(axis=0)]
        if columns.size:
            lows, highs = np.nanpercentile(columns, [2.5, 97.5], axis=0)
            bounds += [float(np.min(lows)), float(np.max(highs))]
    bounds += [float(value) for value in cover if value is not None and np.isfinite(value)]
    return (min(bounds), max(bounds)) if bounds else None
