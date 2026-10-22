"""Graph and observation helpers for canonical operation input projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grassmoth.artifacts.identity import ConstructId
    from grassmoth.artifacts.model.construct import CausalEdgeSpec, ConstructSpec
    from grassmoth.artifacts.model.indicator import IndicatorSpec
    from grassmoth.artifacts.observations.dtypes import MeasurementDtype


# How strongly a pinned reference loading anchors the latent scale, by dtype:
# continuous channels standardize (data units pin scale and location), ordinal
# channels pin scale through the fixed logistic link, binary/count pin scale
# through their link but carry less information, and a pinned categorical
# loading anchors nothing because the free class slopes absorb it.
_REFERENCE_DTYPE_TIERS: dict[MeasurementDtype, int] = {
    "continuous": 0,
    "ordinal": 1,
    "binary": 2,
    "count": 2,
    "categorical": 3,
}


def choose_reference_indicator(
    indicators: Sequence[IndicatorSpec],
) -> IndicatorSpec:
    """Choose a deterministic marker indicator for one construct.

    Prefer the dtype whose fixed loading anchors the latent scale most strongly
    (see ``_REFERENCE_DTYPE_TIERS``); within a tier prefer positive polarity so
    the latent orientation matches the construct name, then declaration order.
    """

    def _rank(item: tuple[int, IndicatorSpec]) -> tuple[int, int, int]:
        declaration_index, indicator = item
        tier = _REFERENCE_DTYPE_TIERS[indicator.observation.measurement_dtype]
        polarity_rank = 0 if indicator.construct_polarity == "positive" else 1
        return (tier, polarity_rank, declaration_index)

    return min(enumerate(indicators), key=_rank)[1]


def get_outcome_construct(
    constructs: Sequence[ConstructSpec],
    outcome: ConstructId | None,
) -> ConstructSpec | None:
    """Resolve the selected outcome from its canonical construct identity."""
    if outcome is None:
        return None
    return next((item for item in constructs if item.id == outcome), None)


def get_outcome_name(
    constructs: Sequence[ConstructSpec], outcome: ConstructId | None
) -> str | None:
    """Resolve the selected outcome display name from its construct."""
    construct = get_outcome_construct(constructs, outcome)
    return construct.name if construct is not None else None


# ---------------------------------------------------------------------------
# Graph utilities (merged from effects.py)
# ---------------------------------------------------------------------------


def build_digraph(
    constructs: Sequence[ConstructSpec], edges: Sequence[CausalEdgeSpec]
) -> nx.DiGraph:
    """Build a directed graph including constructs with no incident edges."""
    graph = nx.DiGraph()
    graph.add_nodes_from(construct.name for construct in constructs)
    graph.add_edges_from((edge.cause.name, edge.effect.name) for edge in edges)
    return graph


def get_all_treatments(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    outcome_id: ConstructId | None,
) -> list[str]:
    """Return construct names with a directed path to the selected outcome."""
    outcome = get_outcome_name(constructs, outcome_id)
    if outcome is None:
        return []
    graph = build_digraph(constructs, edges)
    return sorted(nx.ancestors(graph, outcome))
