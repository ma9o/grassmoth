"""Explicit temporal DAGs for finite-window identification counterexamples.

The notebook reproduces the documented temporal edge convention independently of
Grassmoth's implementation. Its production comparison is a saved public HTTP action.
Only y0's latent projection and complete ID algorithm perform causal identification.
All displayed graphs keep their hidden common causes explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import Variable
from y0.graph import NxMixedGraph

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_ORIGINAL_WITNESS = frozenset({("U_-1", "X_0"), ("U_-1", "U_0"), ("U_0", "Y_1")})
_DELAYED_WITNESS = frozenset(
    {
        ("U_-2", "V_-1"),
        ("V_-1", "X_0"),
        ("U_-2", "U_-1"),
        ("U_-1", "U_0"),
        ("U_0", "Y_1"),
    }
)


@dataclass(frozen=True)
class QueryResult:
    """The fixed query's estimand and its minimal non-identification witness."""

    estimand: str | None
    has_hedge: bool

    @property
    def identified(self) -> bool:
        return self.estimand is not None


def unroll(
    first_slice: int,
    *,
    persistent_hidden: bool = True,
    observe_hidden: bool = False,
) -> nx.DiGraph:
    """Build a first-order SSM DAG ending at 1, keeping the query X_0 -> Y_1 fixed."""
    graph = nx.DiGraph()
    for time in range(first_slice, 2):
        for construct in ("U", "X", "Y"):
            graph.add_node(
                f"{construct}_{time}",
                hidden=construct == "U" and not observe_hidden,
                construct=construct,
                time=time,
            )
    cross_edges = (("U", "X"), ("U", "Y"), ("X", "Y"))
    carryover = (("X", "X"), ("Y", "Y")) + ((("U", "U"),) if persistent_hidden else ())
    for time in range(first_slice, 1):
        for cause, effect in (*cross_edges, *carryover):
            graph.add_edge(f"{cause}_{time}", f"{effect}_{time + 1}")
    return graph


def unroll_delayed(
    first_slice: int,
    *,
    persistent_hidden: bool = True,
    observe_relay: bool = False,
) -> nx.DiGraph:
    """Replace U -> X with U -> V -> X; all temporal edges still have lag one."""
    graph = unroll(first_slice, persistent_hidden=persistent_hidden)
    for time in range(first_slice, 2):
        graph.add_node(f"V_{time}", hidden=not observe_relay, construct="V", time=time)
    for time in range(first_slice, 1):
        graph.remove_edge(f"U_{time}", f"X_{time + 1}")
        graph.add_edges_from(
            (
                (f"U_{time}", f"V_{time + 1}"),
                (f"V_{time}", f"X_{time + 1}"),
                (f"V_{time}", f"V_{time + 1}"),
            )
        )
    return graph


def _project_hidden(graph: nx.DiGraph) -> NxMixedGraph:
    """Project hidden nodes internally, including hidden nodes with hidden parents."""
    renamed = nx.relabel_nodes(graph, {node: Variable(node) for node in graph}, copy=True)
    simplified = simplify_latent_dag(renamed).graph
    projected = NxMixedGraph.from_latent_variable_dag(simplified)
    for node, attributes in simplified.nodes(data=True):
        if not attributes["hidden"]:
            projected.add_node(node)
    return projected


def identify(graph: nx.DiGraph) -> QueryResult:
    """Run ID and check the two-node hedge F={X_0,Y_1}, F'={Y_1}, R={Y_1}."""
    projected = _project_hidden(graph)
    treatment, outcome = Variable("X_0"), Variable("Y_1")
    expression = identify_outcomes(
        projected, treatments={treatment}, outcomes={outcome}, strict=False
    )
    # A directed edge plus a latent common-cause connection gives this hedge.
    # Both forests root at the outcome, which is its own ancestor after do(X_0).
    hedge = projected.directed.has_edge(treatment, outcome) and projected.undirected.has_edge(
        treatment, outcome
    )
    return QueryResult(None if expression is None else str(expression), hedge)


def history_marginalized() -> nx.DiGraph:
    """Keep the full three-slice causal law but observe only X,Y at times 0 and 1."""
    graph = unroll(-1)
    for _, attributes in graph.nodes(data=True):
        if attributes["time"] < 0:
            attributes["hidden"] = True
    return graph


def read_api_evidence():
    """Read the unmodified saved HTTP response and the requests that produced it."""
    path = Path(__file__).resolve().parents[1] / "data" / "two_slice_identification.json"
    return json.loads(path.read_text())


def _draw_dag(
    axis: Axes, graph: nx.DiGraph, title: str, witness: frozenset[tuple[str, str]]
) -> None:
    """Draw explicit hidden states; red arrows expose the omitted history path."""
    constructs = sorted({attributes["construct"] for _, attributes in graph.nodes(data=True)})
    row = {construct: 1.2 * index for index, construct in enumerate(reversed(constructs))}
    top = max(row.values())
    times = sorted({attributes["time"] for _, attributes in graph.nodes(data=True)})
    positions = {
        node: (2.0 * (attributes["time"] - times[0]), row[attributes["construct"]])
        for node, attributes in graph.nodes(data=True)
    }
    for time in times:
        x = 2.0 * (time - times[0])
        axis.axvspan(x - 0.48, x + 0.48, color="#f1f5f9", zorder=0)
        axis.text(x, top + 0.65, f"time {time}", ha="center", color="#475569", fontsize=11)

    internal_nodes = {node for edge in witness for node in edge} - {"X_0", "Y_1"}
    witness_active = witness.issubset(graph.edges) and all(
        graph.nodes[node]["hidden"] for node in internal_nodes
    )
    for edge in graph.edges:
        causal = edge == ("X_0", "Y_1")
        highlighted = witness_active and edge in witness
        color = "#dc2626" if highlighted else "#2563eb" if causal else "#94a3b8"
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=[edge],
            node_size=1050,
            arrowsize=17,
            width=2.6 if highlighted or causal else 1.1,
            edge_color=color,
            alpha=1.0 if highlighted or causal else 0.7,
            connectionstyle="arc3,rad=0.04" if causal else "arc3,rad=0",
        )

    for node, attributes in graph.nodes(data=True):
        hidden = attributes["hidden"]
        query_node = node in {"X_0", "Y_1"}
        fill = "#fff7ed" if hidden else "#dbeafe" if query_node else "#ffffff"
        border = "#c2410c" if hidden else "#1d4ed8" if query_node else "#64748b"
        artist = nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            nodelist=[node],
            node_size=1050,
            node_color=fill,
            edgecolors=border,
            linewidths=1.8,
        )
        artist.set_linestyle("dashed" if hidden else "solid")
        label = rf"${attributes['construct']}_{{{attributes['time']}}}$"
        nx.draw_networkx_labels(graph, positions, labels={node: label}, ax=axis, font_size=13)

    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22)
    axis.set_xlim(-0.75, 2.0 * (times[-1] - times[0]) + 0.75)
    axis.set_ylim(-0.55, top + 0.9)
    axis.set_axis_off()


def comparison_figure() -> Figure:
    """Compare the exact same query in two and three displayed slices."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=(2, 3))
    _draw_dag(axes[0], unroll(0), "Two slices: ID returns an estimand", _ORIGINAL_WITNESS)
    _draw_dag(axes[1], unroll(-1), "Three slices: ID returns not identified", _ORIGINAL_WITNESS)
    figure.text(
        0.5,
        0.02,
        "Dashed orange nodes: unmeasured U  |  Blue: causal edge of interest  |  Red: hidden backdoor path",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    plt.close(figure)
    return figure


def explorer_figure(graph: nx.DiGraph) -> Figure:
    """Draw the interactive window without displaying the internal mixed graph."""
    first = min(attributes["time"] for _, attributes in graph.nodes(data=True))
    figure, axis = plt.subplots(figsize=(max(7.0, (2 - first) * 2.0), 4.5))
    _draw_dag(axis, graph, r"Same query throughout: $P(Y_1\mid do(X_0=x))$", _ORIGINAL_WITNESS)
    figure.tight_layout()
    plt.close(figure)
    return figure


def delayed_comparison_figure() -> Figure:
    """Show the new hidden path that first appears when the fourth slice is added."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), width_ratios=(3, 4))
    _draw_dag(axes[0], unroll_delayed(-1), "Three slices: identified", _DELAYED_WITNESS)
    _draw_dag(axes[1], unroll_delayed(-2), "Four slices: not identified", _DELAYED_WITNESS)
    figure.text(
        0.5,
        0.02,
        "U and V are hidden  |  Every arrow has lag one  |  The query remains X₀ to Y₁",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    plt.close(figure)
    return figure


def delayed_explorer_figure(graph: nx.DiGraph) -> Figure:
    """Draw the four-state counterexample at the selected history depth."""
    first = min(attributes["time"] for _, attributes in graph.nodes(data=True))
    figure, axis = plt.subplots(figsize=(max(7.0, (2 - first) * 2.0), 5.5))
    _draw_dag(axis, graph, r"Delayed hidden cause: $P(Y_1\mid do(X_0=x))$", _DELAYED_WITNESS)
    figure.tight_layout()
    plt.close(figure)
    return figure


def construct_figure() -> Figure:
    """Show the authored construct DAG; carryover is implicit only in this panel."""
    graph = nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")])
    positions = {"U": (0.0, 1.1), "X": (-1.0, 0.0), "Y": (1.0, 0.0)}
    figure, axis = plt.subplots(figsize=(7, 3.0))
    nx.draw_networkx_edges(graph, positions, ax=axis, node_size=1500, arrowsize=20, width=1.7)
    for node, fill, border in (
        ("U", "#fff7ed", "#c2410c"),
        ("X", "#dbeafe", "#1d4ed8"),
        ("Y", "#dbeafe", "#1d4ed8"),
    ):
        artist = nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            nodelist=[node],
            node_size=1500,
            node_color=fill,
            edgecolors=border,
            linewidths=2,
        )
        artist.set_linestyle("dashed" if node == "U" else "solid")
    nx.draw_networkx_labels(graph, positions, ax=axis, font_size=16)
    axis.text(0, 1.48, "No indicators", ha="center", color="#9a3412", fontsize=11)
    axis.text(-1, -0.33, "Measured treatment", ha="center", color="#1d4ed8", fontsize=11)
    axis.text(1, -0.33, "Measured outcome", ha="center", color="#1d4ed8", fontsize=11)
    axis.set_xlim(-1.8, 1.8)
    axis.set_ylim(-0.55, 1.7)
    axis.set_axis_off()
    figure.tight_layout()
    plt.close(figure)
    return figure
