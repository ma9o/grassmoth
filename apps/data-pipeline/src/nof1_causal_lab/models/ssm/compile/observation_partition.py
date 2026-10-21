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
    from nof1_causal_lab.models.ssm.structure.blocks import ManifestCholBlockSpec, SparseBlockSpec


@dataclass(frozen=True)
class CompiledObservationPartition:
    """Disjoint, exhaustive channel groups preserving every joint density factor."""

    local: tuple[int, ...]
    replayed: tuple[int, ...]


def compile_observation_partition(
    loading: SparseBlockSpec[tuple[int, int]],
    noise: ManifestCholBlockSpec,
    laws: ObservationLaws,
    state_space: StateSpace,
) -> CompiledObservationPartition:
    """Use structural loading support, including free coefficients, to place factors.

    Fixed initial coordinates need replay even when their later transitions
    are stochastic. Inputs and global static coordinates are already exact locally.
    Gaussian channels connected by possible noise covariance stay together.
    Independent channels can remain local even when their law family is shared
    with a replayed observation. Structural Cholesky support includes free scales,
    so the partition remains valid for every parameter draw.
    """
    loading_support = (np.asarray(loading.template) != 0) | loading.free_support
    dependent = frozenset(
        int(index)
        for index in np.flatnonzero(
            loading_support[:, state_space.replayed_coordinates].any(axis=1)
        )
    )
    noise_support = (np.asarray(noise.template) != 0) | np.diag(noise.diag_support)
    covariance_support = noise_support @ noise_support.T
    for group in compile_law_groups(laws):
        if isinstance(group.laws[0], NormalLawSpec):
            frontier = dependent.intersection(group.indices)
            while frontier:
                connected = frozenset(
                    index
                    for index in group.indices
                    if any(covariance_support[index, other] for other in frontier)
                )
                frontier = connected - dependent
                dependent = dependent | frontier
    return CompiledObservationPartition(
        local=tuple(index for index in range(len(laws)) if index not in dependent),
        replayed=tuple(index for index in range(len(laws)) if index in dependent),
    )
