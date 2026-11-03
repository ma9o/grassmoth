"""Prove which deterministic outputs cannot couple the fitted path factors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel


def compile_output_coordinates(
    model: CompiledDynamicalModel, observations: np.ndarray
) -> tuple[int, ...]:
    """Find deterministic coordinates with no downstream density or later reading.

    Every stochastic coordinate is a root, even when it is unobserved. Initial
    random coordinates are roots too. An origin-only reading of a deterministic
    output does not couple later blocks. The same dependency proof applies to
    exact factors and their Gaussian proposal derivatives; it never relies on
    a derivative happening to vanish at a numerical linearization point.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(model.state_space.size))
    graph.add_edges_from(
        (source, component.target)
        for component in model.dynamics.spec.components
        for source in component.sources
    )
    loadings = np.asarray(model.loading_block.template) != 0
    loadings = loadings.copy()
    for row, column in model.loading_block.free_positions:
        loadings[row, column] = True
    observed = np.any(np.isfinite(observations[1:]), axis=0)
    roots = frozenset(
        (
            *model.state_space.initial,
            *model.state_space.transition,
            *(int(index) for index in np.flatnonzero(np.any(loadings[observed], axis=0))),
        )
    )
    needed = roots | frozenset(source for root in roots for source in nx.ancestors(graph, root))
    return tuple(
        index
        for stage in model.state_space.recurrence
        for index in stage.indices
        if index not in needed
    )
