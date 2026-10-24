"""Bounded-memory descriptive movement of retained exact nonlinear trajectories."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import dct

from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.reports.scorecard import (
    FrequencyRefresh,
    FrequencyRefreshBand,
    MovementQuantiles,
    MovementRegion,
    MovementWindow,
    ReadingMovement,
    RegionalMovement,
    WorstTimeMovement,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from grassmoth.artifacts.identity import ConstructId
    from grassmoth.artifacts.probability.joint_layout import JointLawLayout


_DRAW_BLOCK = 128
_BAND_EDGES = (0, 16, 64, 128, 256, 512, 1024, 2049)


def _refresh_ratio(paths: NDArray, *, frequency: bool) -> NDArray | None:
    """Pool variances with parallel Welford merges; never join separate chains.

    Each block includes its predecessor only for adjacent-draw jumps. DCT-II
    evaluates saved paths exactly as a display transform, without changing a
    model density. Working arrays scale with 129 draws of one construct.
    """
    if paths.shape[1] < 2:
        return None
    mean = np.zeros(paths.shape[-1], dtype=np.float64)
    squared = np.zeros_like(mean)
    jumps = np.zeros_like(mean)
    count = pairs = 0
    for chain in paths:
        for start in range(0, len(chain), _DRAW_BLOCK):
            block = np.asarray(chain[max(0, start - 1) : start + _DRAW_BLOCK], dtype=np.float64)
            values = (
                np.asarray(dct(block, type=2, norm="ortho", axis=-1), dtype=np.float64)
                if frequency
                else block
            )
            draws = values[1:] if start else values
            size = len(draws)
            block_mean = draws.mean(axis=0)
            difference = block_mean - mean
            squared = (
                squared
                + np.sum((draws - block_mean) ** 2, axis=0)
                + difference**2 * (count * size / (count + size))
            )
            mean = mean + difference * (size / (count + size))
            count += size
            jumps = jumps + np.sum(np.diff(values, axis=0) ** 2, axis=0)
            pairs += len(values) - 1
    variance = squared / count
    return np.divide(jumps / pairs, 2 * variance, out=np.zeros_like(jumps), where=variance > 0)


def frequency_refresh(
    atoms: NDArray,
    layout: JointLawLayout,
    labels: Mapping[ConstructId, str],
    chains: int,
) -> tuple[FrequencyRefresh, ...]:
    """Reduce every retained trajectory's modes in the scorecard's declared bands."""
    steps = len(layout.time_points)
    edges = (*tuple(edge for edge in _BAND_EDGES if edge < steps), steps)
    result = []
    for identity, span in layout.trajectory_slices.items():
        ratio = _refresh_ratio(atoms[:, span].reshape(chains, -1, steps), frequency=True)
        result.append(
            FrequencyRefresh(
                subject=ConstructRef(id=identity),
                label=labels[identity],
                bands=tuple(
                    FrequencyRefreshBand(
                        start_mode=start,
                        stop_mode=stop,
                        median_refresh=None
                        if ratio is None
                        else float(np.median(ratio[start:stop])),
                    )
                    for start, stop in pairwise(edges)
                ),
            )
        )
    return tuple(result)


def _quantiles(values: NDArray) -> MovementQuantiles | None:
    if not values.size:
        return None
    minimum, median, maximum = np.quantile(values, (0, 0.5, 1))
    return MovementQuantiles(minimum=float(minimum), median=float(median), maximum=float(maximum))


def regional_movement(
    atoms: NDArray,
    layout: JointLawLayout,
    labels: Mapping[ConstructId, str],
    chains: int,
    *,
    free_mask: NDArray,
    final_delta: NDArray | None,
    windows: tuple[MovementWindow, ...],
    observed_rows: NDArray,
) -> tuple[RegionalMovement, ...]:
    """Summarize actual retained movement and final steps without rebuilding paths."""
    times = np.asarray(layout.time_points)
    result = []
    for dimension, (identity, span) in enumerate(layout.trajectory_slices.items()):
        ratio = _refresh_ratio(atoms[:, span].reshape(chains, -1, len(times)), frequency=False)
        free = free_mask[:, dimension]
        indices = np.flatnonzero(free)
        worst = (
            ()
            if ratio is None
            else tuple(
                WorstTimeMovement(
                    time_index=int(index), time_day=float(times[index]), refresh=float(ratio[index])
                )
                for index in indices[np.argsort(ratio[indices], kind="stable")[:10]]
            )
        )
        regions = []
        for window in windows:
            selected = free & (times >= window.start_day) & (times < window.end_day)
            regions.append(
                MovementRegion(
                    window=window,
                    coordinates=int(selected.sum()),
                    refresh_quantiles=None if ratio is None else _quantiles(ratio[selected]),
                    delta_quantiles=None
                    if final_delta is None
                    else _quantiles(final_delta[:, selected]),
                )
            )
        coverage = tuple(
            ReadingMovement(
                has_readings=has_readings,
                coordinates=int(selected.sum()),
                stretches=int(np.count_nonzero(selected & ~np.r_[False, selected[:-1]])),
                refresh_quantiles=None if ratio is None else _quantiles(ratio[selected]),
                delta_quantiles=None
                if final_delta is None
                else _quantiles(final_delta[:, selected]),
            )
            for has_readings in (True, False)
            for selected in (free & (observed_rows == has_readings),)
        )
        result.append(
            RegionalMovement(
                subject=ConstructRef(id=identity),
                label=labels[identity],
                regions=tuple(regions),
                reading_coverage=coverage,
                worst_times=worst,
            )
        )
    return tuple(result)
