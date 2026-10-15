"""Separate tree-local likelihood factors from factors requiring state replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.model.likelihood import NormalLawSpec
from nof1_causal_lab.models.ssm.execution.observation_groups import compile_law_groups

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.state_space import StateSpace
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.structure.blocks import SparseBlockSpec


@dataclass(frozen=True)
class CompiledObservationPartition:
    """Disjoint, exhaustive channel groups preserving every joint density factor."""

    local: tuple[int, ...]
    replayed: tuple[int, ...]


def compile_observation_partition(
    loading: SparseBlockSpec[tuple[int, int]],
    laws: ObservationLaws,
    state_space: StateSpace,
) -> CompiledObservationPartition:
    """Use structural loading support, including free coefficients, to place factors.

    Fixed initial coordinates need replay even when their later transitions
    are stochastic. Inputs and global static coordinates are already exact locally.
    A Gaussian block is indivisible: masking it into two marginal densities would
    lose its cross-channel covariance. Other laws factor over their channels.
    """
    loading_support = (np.asarray(loading.template) != 0) | loading.free_support
    dependent = {
        int(index)
        for index in np.flatnonzero(
            loading_support[:, state_space.replayed_coordinates].any(axis=1)
        )
    }
    for group in compile_law_groups(laws):
        if isinstance(group.laws[0], NormalLawSpec) and dependent.intersection(group.indices):
            dependent.update(group.indices)
    return CompiledObservationPartition(
        local=tuple(index for index in range(len(laws)) if index not in dependent),
        replayed=tuple(index for index in range(len(laws)) if index in dependent),
    )
