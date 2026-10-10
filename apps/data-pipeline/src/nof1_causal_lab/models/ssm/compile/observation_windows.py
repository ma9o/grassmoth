"""Sparse measurement quadrature, independent of scan slots and particle trees."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class CompiledObservationWindow:
    """One measurement's sparse weights on the endpoints of intersecting grid segments.

    Each row of ``coefficients`` weights the responses at ``step_indices - 1``
    and ``step_indices``. Repeated interior endpoints add, so the stencil
    defines a linear functional of the response without choosing an executor.
    """

    anchor_index: int
    manifest_index: int
    step_indices: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        """Detach the sparse stencil's buffers before publishing it."""
        freeze_fields(self)

    @property
    def start_index(self) -> int:
        """First path coordinate needed by this measurement, including empty windows."""
        return int(self.step_indices[0] - 1) if self.step_indices.size else self.anchor_index

    @property
    def weights(self) -> np.ndarray:
        """Quadrature measure contributed by each intersecting segment."""
        return self.coefficients.sum(axis=1)


@dataclass(frozen=True)
class CompiledWindowPlan:
    """Canonical measurement stencils bound to one path grid and indicator axis."""

    times: np.ndarray
    n_manifest: int
    windows: tuple[CompiledObservationWindow, ...]

    def __post_init__(self) -> None:
        """Own the grid and immutable sequence of sparse measurement stencils."""
        freeze_fields(self)

    def with_held_channels(self, channels: tuple[int, ...]) -> CompiledWindowPlan:
        """Use each segment's left endpoint for channels declaring held responses."""
        if not channels:
            return self
        return replace(
            self,
            windows=tuple(
                replace(
                    window,
                    coefficients=np.column_stack((window.weights, np.zeros_like(window.weights))),
                )
                if window.manifest_index in channels
                else window
                for window in self.windows
            ),
        )


@dataclass(frozen=True)
class CompiledObservationScan:
    """Reusable accumulator slots derived from a sparse window plan."""

    prev_coeffs: np.ndarray
    curr_coeffs: np.ndarray
    weights: np.ndarray
    emission_slots: np.ndarray

    def __post_init__(self) -> None:
        """Own the scan's fixed-shape execution buffers."""
        freeze_fields(self)

    @property
    def max_active_windows(self) -> int:
        """Accumulator width, including the empty slot used for point-only plans."""
        return int(self.prev_coeffs.shape[2])


def compile_window_plan(
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    support_kinds: Sequence[str | None],
    manifest_names: Sequence[str],
) -> CompiledWindowPlan | ObservationPreflightFailure:
    """Resolve window invariants and trapezoidal stencils before choosing an executor."""
    tolerance = 1e-8
    times = np.asarray(times, dtype=np.float64)
    windows = []
    for manifest, (kind, name) in enumerate(zip(support_kinds, manifest_names, strict=True)):
        if kind != "interval":
            continue
        rows = np.flatnonzero(np.isfinite(starts[:, manifest]) & np.isfinite(ends[:, manifest]))
        for row in rows:
            start, end = float(starts[row, manifest]), float(ends[row, manifest])
            anchor = float(times[row])
            if end + tolerance < start:
                return ObservationPreflightFailure.rejected(
                    f"Indicator '{name}' has support_end before support_start "
                    f"at row {row}: {start} -> {end}"
                )
            if abs(anchor - end) > tolerance:
                return ObservationPreflightFailure.rejected(
                    f"Indicator '{name}' has support_end={end} that does not match "
                    f"its anchored observation time {anchor} at row {row}."
                )
            if start < times[0] - tolerance:
                return ObservationPreflightFailure.rejected(
                    f"Indicator '{name}' has support starting before the first model time. "
                    "Add earlier model-clock rows or shift the observation anchor."
                )

            steps = np.arange(
                max(int(np.searchsorted(times, start, side="right")), 1), int(row) + 1
            )
            left, right = times[steps - 1], times[steps]
            overlap_start, overlap_end = np.maximum(left, start), np.minimum(right, end)
            active = (right - left > tolerance) & (overlap_end - overlap_start > tolerance)
            steps, left, right = steps[active], left[active], right[active]
            overlap_start, overlap_end = overlap_start[active], overlap_end[active]
            width = overlap_end - overlap_start
            fraction = ((overlap_start - left) + (overlap_end - left)) / (2 * (right - left))
            coefficients = np.column_stack((width * (1 - fraction), width * fraction))
            windows.append(CompiledObservationWindow(int(row), manifest, steps, coefficients))
    return CompiledWindowPlan(times, len(manifest_names), tuple(windows))


def compile_observation_scan(plan: CompiledWindowPlan) -> CompiledObservationScan:
    """Allocate reusable scan slots without changing any measurement's quadrature."""
    assignments = []
    max_slots = 0
    for manifest in range(plan.n_manifest):
        windows = sorted(
            (window for window in plan.windows if window.manifest_index == manifest),
            key=lambda window: (window.start_index, window.anchor_index),
        )
        active: list[tuple[int, int]] = []
        free: list[int] = []
        count = 0
        for window in windows:
            while active and active[0][0] <= window.start_index:
                _, slot = heapq.heappop(active)
                heapq.heappush(free, slot)
            if free:
                slot = heapq.heappop(free)
            else:
                slot = count
                count += 1
            assignments.append((window, slot))
            heapq.heappush(active, (window.anchor_index, slot))
        max_slots = max(max_slots, count)

    shape = (len(plan.times), plan.n_manifest, max(max_slots, 1))
    previous, current, weights = (np.zeros(shape) for _ in range(3))
    emissions = np.full(shape[:2], -1, dtype=np.int64)
    for window, slot in assignments:
        channel, steps = window.manifest_index, window.step_indices
        previous[steps, channel, slot] = window.coefficients[:, 0]
        current[steps, channel, slot] = window.coefficients[:, 1]
        weights[steps, channel, slot] = window.weights
        emissions[window.anchor_index, channel] = slot
    return CompiledObservationScan(previous, current, weights, emissions)
