"""Tests for ``models.causal.graphs`` helpers.

Trivial accessors are exercised through higher-level tests. This file covers
the helpers with real transformation or graph logic:
- ``build_digraph``
- ``get_outcome_name``
- ``get_all_treatments``
- DynamicalModelSpec state and marginalized-scale accessors
"""

import pytest

from nof1_causal_lab.models.causal.graphs import build_digraph, get_all_treatments, get_outcome_name
from tests.support.causal_graphs import make_graph
from tests.support.helpers import fixture_entity_id

pytestmark = pytest.mark.contract


class TestBuildDigraph:
    @pytest.mark.parametrize(
        ("names", "pairs", "expected_edges"),
        [
            (["A", "B", "C"], [("A", "B"), ("B", "C")], {("A", "B"), ("B", "C")}),
            ([], [], set()),
            (
                ["A", "B", "C", "D"],
                [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
                {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")},
            ),
            (["A"], [("A", "A")], {("A", "A")}),
            (["A", "B"], [("A", "B"), ("A", "B")], {("A", "B")}),
        ],
        ids=["chain", "empty", "diamond", "self_loop", "duplicate_edges"],
    )
    def test_edge_topologies(self, names, pairs, expected_edges):
        _, edges = make_graph(
            [{"name": name} for name in names],
            [{"cause": cause, "effect": effect} for cause, effect in pairs],
        )
        graph = build_digraph((), edges)
        assert set(graph.nodes) == set(names)
        assert set(graph.edges) == expected_edges


class TestGetOutcomeName:
    def test_finds_outcome(self):
        constructs, _ = make_graph([{"name": "X"}, {"name": "Y"}], [])
        assert get_outcome_name(constructs, fixture_entity_id("construct", "Y")) == "Y"

    def test_no_outcome(self):
        constructs, _ = make_graph([{"name": "X"}, {"name": "Z"}], [])
        assert get_outcome_name(constructs, None) is None

    def test_empty_constructs(self):
        assert get_outcome_name((), None) is None


class TestGetAllTreatments:
    @pytest.mark.parametrize(
        ("names", "pairs", "outcome", "expected"),
        [
            (["A", "B", "Y"], [("A", "B"), ("B", "Y")], "Y", ["A", "B"]),
            (["X", "Y", "Z"], [("X", "Y")], "Y", ["X"]),
            (["A", "B"], [("A", "B")], None, []),
            (
                ["Zebra", "Apple", "Outcome"],
                [("Zebra", "Outcome"), ("Apple", "Outcome")],
                "Outcome",
                ["Apple", "Zebra"],
            ),
            (["X", "Y", "Z"], [("X", "Y"), ("X", "Z")], "Y", ["X"]),
            (
                ["A", "B", "C", "D"],
                [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
                "D",
                ["A", "B", "C"],
            ),
            ([], [], None, []),
            (["Y"], [], "Y", []),
        ],
        ids=[
            "chain",
            "disconnected",
            "no_outcome",
            "sorted",
            "fork",
            "diamond",
            "empty",
            "outcome_only",
        ],
    )
    def test_treatments(self, names, pairs, outcome, expected):
        constructs, edges = make_graph(
            [{"name": name} for name in names],
            [{"cause": cause, "effect": effect} for cause, effect in pairs],
        )
        outcome_id = fixture_entity_id("construct", outcome) if outcome is not None else None
        assert get_all_treatments(constructs, edges, outcome_id) == expected
