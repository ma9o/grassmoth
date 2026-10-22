"""Canonical graph entities for topology tests, including isolated constructs."""

from collections.abc import Mapping, Sequence
from typing import Any

from grassmoth.artifacts.model.construct import CausalEdgeSpec, ConstructSpec
from grassmoth_test_support.helpers import fixture_entity_id


def make_graph(
    constructs: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> tuple[tuple[ConstructSpec, ...], tuple[CausalEdgeSpec, ...]]:
    by_name = {
        item["name"]: ConstructSpec(
            id=fixture_entity_id("construct", item["name"]),
            name=item["name"],
            description=item["name"],
            role=item.get("role", "endogenous"),
            temporal_status=item.get("temporal_status", "time_invariant"),
        )
        for item in constructs
    }
    causal_edges = tuple(
        CausalEdgeSpec(
            id=fixture_entity_id("edge", f"{index}:{edge['cause']}->{edge['effect']}"),
            cause=by_name[edge["cause"]],
            effect=by_name[edge["effect"]],
            description=f"{edge['cause']} causes {edge['effect']}",
        )
        for index, edge in enumerate(edges)
    )
    return tuple(by_name.values()), causal_edges
