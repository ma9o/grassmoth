"""Place interval likelihood factors on the smallest enclosing particle-tree node.

The tree carries response sums, not additional stochastic states. Each recorded
row is paid once, when all of its interval channels are available. Grouping by
row preserves the joint Gaussian density of simultaneously recorded channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.models.ssm.compile.observation_windows import CompiledWindowPlan


@dataclass(frozen=True)
class CompiledIntervalTreeLevel:
    """Sparse child-slot gathers and factors emitted at one balanced-tree depth."""

    left_slots: np.ndarray
    right_slots: np.ndarray
    emission_slots: np.ndarray
    observations: np.ndarray
    masks: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Detach the compiler's arrays before exposing the plan."""
        freeze_fields(self)


@dataclass(frozen=True)
class CompiledIntervalTree:
    """Bounded frontier of unfinished measurement rows at each tree node."""

    leaf_weights: np.ndarray
    levels: tuple[CompiledIntervalTreeLevel, ...]
    statistic_count: int

    def __post_init__(self) -> None:
        """Own immutable arrays and the depth-ordered level plans."""
        freeze_fields(self)


def compile_interval_tree(
    plan: CompiledWindowPlan,
    summary_operators: Sequence[str | None],
    observations: np.ndarray,
) -> CompiledIntervalTree:
    """Lower sparse window stencils to tree frontiers independently of scan slots."""
    steps, channels = observations.shape
    depth = (steps - 1).bit_length()
    padded = 1 << depth
    observed = np.zeros_like(observations, dtype=bool)
    for window in plan.windows:
        row, channel = window.anchor_index, window.manifest_index
        observed[row, channel] = np.isfinite(observations[row, channel])
    rows = tuple(int(row) for row in np.flatnonzero(observed.any(axis=1)))
    windows_by_row = {row: [] for row in rows}
    for window in plan.windows:
        if observed[window.anchor_index, window.manifest_index]:
            windows_by_row[window.anchor_index].append(window)
    # Each row owns just its nonzero time span, and only lives up to its LCA.
    row_weights: dict[int, tuple[int, np.ndarray]] = {}
    owners: dict[int, int] = {}
    frontiers: list[list[list[int]]] = [[[] for _ in range(padded >> d)] for d in range(depth + 1)]
    for row in rows:
        windows = windows_by_row[row]
        first = min(window.start_index for window in windows)
        coefficients = np.zeros((row - first + 1, channels))
        for window in windows:
            indices = window.step_indices - first
            coefficients[indices - 1, window.manifest_index] += window.coefficients[:, 0]
            coefficients[indices, window.manifest_index] += window.coefficients[:, 1]
        row_weights[row] = first, coefficients
        owner = (first ^ row).bit_length()
        owners[row] = owner
        for d in range(owner + 1):
            for node in range(first >> d, (row >> d) + 1):
                frontiers[d][node].append(row)

    leaf_width = max((len(node) for node in frontiers[0]), default=0)
    leaf_weights = np.zeros((padded, leaf_width, channels))
    for time, frontier in enumerate(frontiers[0]):
        for slot, row in enumerate(frontier):
            first, coefficients = row_weights[row]
            leaf_weights[time, slot] = coefficients[time - first]

    levels = []
    for d, nodes in enumerate(frontiers):
        width = max((len(node) for node in nodes), default=0)
        emitted = tuple(tuple(row for row in node if owners[row] == d) for node in nodes)
        emission_width = max((len(node) for node in emitted), default=0)
        left = np.full((len(nodes), width), -1, dtype=np.int32)
        right = left.copy()
        slots = np.zeros((len(nodes), emission_width), dtype=np.int32)
        values = np.zeros((len(nodes), emission_width, channels))
        masks = np.zeros_like(values, dtype=bool)
        weights = np.ones_like(values)
        for node, frontier in enumerate(nodes):
            if d:
                for child_offset, destination in ((0, left), (1, right)):
                    lookup = {
                        row: slot
                        for slot, row in enumerate(frontiers[d - 1][2 * node + child_offset])
                    }
                    for slot, row in enumerate(frontier):
                        destination[node, slot] = lookup.get(row, -1)
            for slot, row in enumerate(emitted[node]):
                slots[node, slot] = frontier.index(row)
                values[node, slot] = np.nan_to_num(observations[row], nan=0.0)
                masks[node, slot] = observed[row]
                weights[node, slot] = row_weights[row][1].sum(axis=0)
        levels.append(CompiledIntervalTreeLevel(left, right, slots, values, masks, weights))
    statistic_count = (
        2 if any(summary_operators[w.manifest_index] == "std" for w in plan.windows) else 1
    )
    return CompiledIntervalTree(leaf_weights, tuple(levels), statistic_count)
