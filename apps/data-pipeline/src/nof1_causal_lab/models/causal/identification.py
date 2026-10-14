"""Identifiability checking using y0's ID algorithm.

Uses Pearl's do-calculus via y0 to check if causal effects are identifiable
given observed/unobserved constructs. This properly handles:
- Backdoor criterion
- Front-door criterion
- Instrumental variables (under a parametric linearity assumption; gated by ``iv_allowed``)
- Other identification strategies from Shpitser & Pearl

Design principle: Users specify DAGs with explicit latent confounders. We convert
to ADMG internally using y0's from_latent_variable_dag() for identification.

Note on IV: y0's nonparametric do-calculus cannot identify effects via IV alone.
``check_identifiability`` can additionally report graph-theoretic IV candidates
when the caller explicitly allows the parametric linearity assumption. The default
returns only nonparametric do-calculus identifications. Nonlinear DynamicalModelSpec
authoring and causal reporting do not permit the linear-IV argument.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.models.causal.graphs import build_digraph, get_all_treatments, get_outcome_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.model.construct import CausalEdgeSpec, ConstructSpec


@dataclass(frozen=True, kw_only=True)
class IdentifiedQuery:
    """An identified causal query with its estimand and supporting confounder or instrument findings."""

    estimand: str
    marginalized_confounders: tuple[str, ...]
    instruments: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class UnidentifiedQuery:
    """A causal query blocked by the reported confounding structure, with optional explanatory notes."""

    confounders: tuple[str, ...]
    notes: str | None = None


@dataclass(frozen=True, kw_only=True)
class IdentificationResult:
    """y0 query output; names are resolved to canonical IDs by the report producer."""

    identifiable_treatments: Mapping[str, IdentifiedQuery]
    non_identifiable_treatments: Mapping[str, UnidentifiedQuery]

    def __post_init__(self) -> None:
        """Own immutable collections of causal identification findings after construction."""
        freeze_fields(self)


logger = logging.getLogger(__name__)


def check_identifiability(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    *,
    outcome_id: ConstructId | None,
    observed_constructs: set[str],
    iv_allowed: bool = False,
) -> IdentificationResult:
    """Check which treatment effects are identifiable using y0's ID algorithm.

    Applies ID to the selected treatment/outcome query in a 2-timestep graph.
    The temporal scope and its limitations are documented in
    docs/assumptions.md#causal-identification.

    Args:
        constructs: Canonical construct definitions, including latent confounders
        edges: Canonical directed causal assumptions
        outcome_id: Identity of the question's outcome construct
        observed_constructs: Names of constructs with measurements
        iv_allowed: When explicitly True and y0's nonparametric check fails,
            report IV identification via ``find_instruments`` under the
            caller's parametric linearity assumption. When False, only
            nonparametric do-calculus identifications are returned (the default).

    Returns:
        Dict with:
            - identifiable_treatments: Map of treatment -> identification details
                * estimand: Closed-form estimand or IV placeholder
                * marginalized_confounders: Unobserved constructs the estimand integrates out
                * instruments: Optional list of IVs when applicable
            - non_identifiable_treatments: Map of treatment -> confounder context
                * confounders: Unobserved constructs blocking identification
                * notes: Optional explanation when confounders cannot be enumerated

    Raises:
        ValueError: No outcome construct is selected for the identification query.
    """
    outcome = get_outcome_name(constructs, outcome_id)
    if not outcome:
        raise ValueError("No outcome selected for identification")

    # Get all potential treatments (observed constructs with paths to outcome)
    # Only observed constructs can be treatments - you can't do(X) on unobserved X
    all_treatments = [
        treatment
        for treatment in get_all_treatments(constructs, edges, outcome_id)
        if treatment in observed_constructs
    ]

    # Determine if outcome is time-varying or time-invariant
    outcome_is_time_varying = _is_time_varying(constructs, outcome)

    # Check each treatment
    identifiable_treatments: dict[str, IdentifiedQuery] = {}
    non_identifiable_treatments: dict[str, UnidentifiedQuery] = {}

    # If outcome itself is unobserved, no effects are identifiable
    if outcome not in observed_constructs:
        for treatment in all_treatments:
            non_identifiable_treatments[treatment] = UnidentifiedQuery(
                confounders=(outcome,), notes="outcome is unobserved"
            )
        return IdentificationResult(
            identifiable_treatments=identifiable_treatments,
            non_identifiable_treatments=non_identifiable_treatments,
        )

    # Convert DAG to ADMG via 2-timestep unrolling
    admg, unobserved_confounders = dag_to_admg(constructs, edges, observed_constructs)

    for treatment in all_treatments:
        # Build timestamped variable names for y0 query
        treatment_node = _get_treatment_query_node(constructs, treatment)

        if outcome_is_time_varying:
            outcome_node = _node_name(outcome, "t")
        else:
            outcome_node = outcome

        treatment_var = Variable(treatment_node)
        outcome_var = Variable(outcome_node)

        estimand = identify_outcomes(
            admg,
            treatments={treatment_var},
            outcomes={outcome_var},
        )
        if estimand is not None:  # ty: ignore[redundant-condition-strict]  # pyright: ignore[reportUnnecessaryComparison] - y0 returns None for unidentifiable queries; static analysis narrows its exception path incorrectly.
            # Map estimand back to original names for readability
            estimand_str = _canonicalize_estimand_string(str(estimand))
            identifiable_treatments[treatment] = IdentifiedQuery(
                estimand=estimand_str,
                marginalized_confounders=tuple(sorted(unobserved_confounders)),
            )
        else:
            # y0's nonparametric check failed; optionally report IV
            # identification under the caller's parametric assumption.
            instruments = (
                find_instruments(constructs, edges, observed_constructs, treatment, outcome)
                if iv_allowed
                else []
            )
            if instruments:
                # IV identification available under the caller's linearity assumption.
                iv_list = ", ".join(instruments)
                identifiable_treatments[treatment] = IdentifiedQuery(
                    estimand=f"IV({iv_list}) [requires linearity]",
                    marginalized_confounders=tuple(sorted(unobserved_confounders)),
                    instruments=tuple(instruments),
                )
            else:
                if treatment_node == _node_name(treatment, "{t-1}"):
                    blockers = find_blocking_confounders_for_query(
                        constructs,
                        edges,
                        observed_constructs,
                        treatment_node=treatment_node,
                        outcome_node=outcome_node,
                    )
                else:
                    blockers = find_blocking_confounders(
                        constructs, edges, observed_constructs, treatment, outcome
                    )
                non_identifiable_treatments[treatment] = UnidentifiedQuery(
                    confounders=tuple(blockers)
                )

    return IdentificationResult(
        identifiable_treatments=identifiable_treatments,
        non_identifiable_treatments=non_identifiable_treatments,
    )


def _is_time_varying(constructs: Sequence[ConstructSpec], construct_name: str) -> bool:
    """Check if a construct is time-varying (vs time-invariant)."""
    for construct in constructs:
        if construct.name == construct_name:
            return construct.is_dynamic
    raise ValueError(f"Construct '{construct_name}' not found in latent structure")


def get_observed_constructs(constructs: Sequence[ConstructSpec]) -> set[str]:
    """Return constructs known through measurements or supplied deterministic inputs."""
    return {
        construct.name
        for construct in constructs
        if construct.indicators or construct.role == "exogenous"
    }


def _node_name(construct: str, timestep: str) -> str:
    """Create timestamped node name like 'X_t' or 'X_{t-1}'."""
    return f"{construct}_{timestep}"


def _canonicalize_estimand_string(estimand: str) -> str:
    """Stabilize y0 estimand text across Python hash seeds."""

    def sort_sum_variables(match: re.Match[str]) -> str:
        variables = [item.strip() for item in match.group(1).split(",") if item.strip()]
        return "Sum[" + ", ".join(sorted(variables)) + "]"

    return re.sub(r"Sum\[([^\]]*)\]", sort_sum_variables, estimand)


def _get_treatment_query_node(
    constructs: Sequence[ConstructSpec],
    treatment: str,
) -> str:
    """Intervene on the preceding state, or on a static construct without a slice."""
    return _node_name(treatment, "{t-1}") if _is_time_varying(constructs, treatment) else treatment


def unroll_temporal_dag(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    observed_constructs: set[str],
) -> nx.DiGraph:
    """Unroll a temporal causal graph to a 2-timestep DAG for identification.

    This constructs the finite graph used by the current identifier. See
    docs/assumptions.md#causal-identification for its temporal scope.

    Node creation:
    - Time-varying constructs → C_t, C_{t-1}
    - Time-invariant constructs → C (single node, no timestep suffix)

    Edge creation:
    - Dynamic cause: cause_{t-1} → effect_t
    - Carryover for every time-varying state: C_{t-1} → C_t
    - Time-invariant to time-varying: C → effect_t (for each timestep)

    Hidden labels:
    - Observed constructs: all timesteps have hidden=False
    - Unobserved constructs: all timesteps have hidden=True

    Args:
        constructs: Canonical construct definitions
        edges: Declared direct causal assumptions; mechanisms use the current state
        observed_constructs: Set of construct names that have measurements

    Returns:
        nx.DiGraph with timestamped nodes and hidden labels for y0
    """
    dag = nx.DiGraph()

    # Categorize constructs by temporal status
    time_varying: list[str] = []
    time_invariant: list[str] = []

    for construct in constructs:
        name = construct.name
        if not construct.is_dynamic:
            time_invariant.append(name)
        else:
            time_varying.append(name)

    time_invariant_set = set(time_invariant)

    # Add nodes for time-varying constructs (both timesteps)
    for name in time_varying:
        is_hidden = name not in observed_constructs
        dag.add_node(_node_name(name, "t"), hidden=is_hidden, construct=name, timestep="t")
        dag.add_node(_node_name(name, "{t-1}"), hidden=is_hidden, construct=name, timestep="{t-1}")

    # Add nodes for time-invariant constructs (single node)
    for name in time_invariant:
        is_hidden = name not in observed_constructs
        dag.add_node(name, hidden=is_hidden, construct=name, timestep=None)

    # State transitions carry every evolving state, including exogenous and hidden states.
    for name in time_varying:
        dag.add_edge(_node_name(name, "{t-1}"), _node_name(name, "t"))

    # Add the authored causal edges.
    for edge in edges:
        cause = edge.cause.name
        effect = edge.effect.name

        cause_is_time_invariant = cause in time_invariant_set
        effect_is_time_invariant = effect in time_invariant_set

        if cause_is_time_invariant and effect_is_time_invariant:
            # Both time-invariant: single edge
            dag.add_edge(cause, effect)
        elif cause_is_time_invariant:
            # Time-invariant cause affects time-varying effect at current time
            # (Time-invariant constructs represent stable traits that affect all timepoints)
            dag.add_edge(cause, _node_name(effect, "t"))
            # Also affects t-1 if we're modeling the full 2-timestep window
            dag.add_edge(cause, _node_name(effect, "{t-1}"))
        elif effect_is_time_invariant:
            # Time-varying cause cannot affect time-invariant effect
            # (This would violate the definition of time-invariant)
            # Skip this edge - should be caught by schema validation
            continue
        else:
            # A transition slice expresses evolution, not a physical delay in the equations.
            dag.add_edge(_node_name(cause, "{t-1}"), _node_name(effect, "t"))

    return dag


def dag_to_admg(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    observed_constructs: set[str],
) -> tuple[NxMixedGraph, set[str]]:
    """Convert a temporal DAG to ADMG via 2-timestep unrolling.

    Projects the finite graph using y0's latent simplification and ADMG conversion.
    Temporal placement follows construct status, independently of solver step size.
    See docs/assumptions.md#causal-identification for the temporal limitation.

    Args:
        constructs: Canonical construct definitions
        edges: Canonical directed causal assumptions
        observed_constructs: Set of construct names that have measurements

    Returns:
        Tuple of (NxMixedGraph, set of unobserved confounder names)

    """
    # Build 2-timestep unrolled DAG
    dag = unroll_temporal_dag(constructs, edges, observed_constructs)

    # Find unobserved constructs that will create confounding
    # An unobserved node with 2+ observed children creates bidirected edges
    all_constructs = {construct.name for construct in constructs}
    unobserved = all_constructs - observed_constructs

    unobserved_confounders: set[str] = set()
    for node in dag.nodes():
        if not dag.nodes[node].get("hidden", False):
            continue

        # Get the original construct name
        construct = dag.nodes[node].get("construct", node)
        if construct not in unobserved:
            continue

        # Count observed children (children with hidden=False)
        observed_children = [
            child for child in dag.successors(node) if not dag.nodes[child].get("hidden", False)
        ]
        if len(observed_children) >= 2:
            unobserved_confounders.add(construct)

    # y0's converter expects root latents with observed children. Simplify first
    # so hidden carryover and hidden mediators never become observed ADMG nodes.
    projection = nx.relabel_nodes(dag, {node: Variable(node) for node in dag})
    simplified = simplify_latent_dag(projection).graph
    admg = NxMixedGraph.from_latent_variable_dag(simplified)
    for node, data in simplified.nodes(data=True):
        if not data["hidden"]:
            admg.add_node(node)

    return admg, unobserved_confounders


def find_blocking_confounders(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    observed_constructs: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Find unobserved constructs that confound the treatment-outcome relationship.

    A confounder U creates a backdoor path from treatment to outcome. For U to
    be a *blocking* confounder (one that needs a proxy):
    - U must be an ancestor of treatment
    - U must reach outcome through a path that does NOT go through treatment
    - U must have at least one direct child that is observed

    The last condition ensures U is a "proximal" confounder. If U only affects
    observed nodes through other unobserved nodes (U1 → U2 → X), then U2 is the
    proximal confounder that needs a proxy, not U1. Observing U1 wouldn't help
    if U2 remains unobserved.

    Note: This may over-report if identification strategies like front-door
    handle some confounding. The actual identification decision is made by
    y0's identify_outcomes() algorithm.
    """
    G = build_digraph(constructs, edges)

    all_constructs = {construct.name for construct in constructs}
    unobserved = all_constructs - observed_constructs

    # Create graph with treatment removed to check backdoor paths
    G_sans_treatment = G.copy()
    if treatment in G_sans_treatment:
        G_sans_treatment.remove_node(treatment)

    blocking = []
    for u in unobserved:
        if u not in G:
            continue
        if u in (treatment, outcome):
            continue

        # Check if U has any direct observed children
        # If not, U's confounding effect is mediated through other unobserved nodes
        direct_children = list(G.successors(u))
        has_observed_child = any(c in observed_constructs for c in direct_children)
        if not has_observed_child:
            continue

        # U is a blocking confounder if:
        # 1. U is an ancestor of treatment
        # 2. U reaches outcome WITHOUT going through treatment (backdoor path)
        is_ancestor_of_treatment = nx.has_path(G, u, treatment)
        reaches_outcome_via_backdoor = (
            u in G_sans_treatment
            and outcome in G_sans_treatment
            and nx.has_path(G_sans_treatment, u, outcome)
        )

        if is_ancestor_of_treatment and reaches_outcome_via_backdoor:
            blocking.append(u)

    return blocking


def find_blocking_confounders_for_query(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    observed_constructs: set[str],
    *,
    treatment_node: str,
    outcome_node: str,
) -> list[str]:
    """Find unobserved constructs blocking a specific unrolled treatment query."""
    dag = unroll_temporal_dag(constructs, edges, observed_constructs)

    dag_sans_treatment = dag.copy()
    if treatment_node in dag_sans_treatment:
        dag_sans_treatment.remove_node(treatment_node)

    blocking: set[str] = set()
    for node, attrs in dag.nodes(data=True):
        if not attrs.get("hidden", False):
            continue
        if node in (treatment_node, outcome_node):
            continue

        construct = attrs.get("construct", node)
        if construct in (treatment_node, outcome_node):
            continue

        has_observed_child = any(
            not dag.nodes[child].get("hidden", False) for child in dag.successors(node)
        )
        if not has_observed_child:
            continue

        is_ancestor_of_treatment = treatment_node in dag and nx.has_path(dag, node, treatment_node)
        reaches_outcome_via_backdoor = (
            node in dag_sans_treatment
            and outcome_node in dag_sans_treatment
            and nx.has_path(dag_sans_treatment, node, outcome_node)
        )

        if is_ancestor_of_treatment and reaches_outcome_via_backdoor:
            blocking.add(str(construct))

    return sorted(blocking)


def find_instruments(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    observed_constructs: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Find valid instrumental variables for the treatment-outcome relationship.

    Based on DoWhy's graph-theoretic approach (py-why/dowhy), adapted to handle
    explicit unobserved confounders. A valid instrument Z for X → Y requires:

    1. Relevance: Z is a direct parent of X (Z → X edge exists)
    2. Exclusion: Z is not an ancestor of Y when X's incoming edges are removed
       (Z affects Y only through X)
    3. As-if-random (Exogeneity): Z is not a descendant of any node that causes Y
       (Z is not affected by confounders of the X-Y relationship)

    This enables IV identification under linear SEM assumptions, even when
    y0's nonparametric do-calculus says the effect is not identifiable.

    Reference: https://github.com/py-why/dowhy/blob/main/dowhy/graph.py

    Args:
        constructs: Canonical construct definitions
        edges: Canonical directed causal assumptions
        observed_constructs: Set of observed construct names
        treatment: The treatment variable name
        outcome: The outcome variable name

    Returns:
        List of valid instrument names (observed constructs that satisfy IV conditions)
    """
    G = build_digraph(constructs, edges)

    if treatment not in G or outcome not in G:
        return []

    # Get direct parents of treatment (potential instruments must be parents)
    parents_treatment = set(G.predecessors(treatment))

    # Do surgery: remove incoming edges to treatment
    G_surgered = G.copy()
    incoming_to_treatment = list(G_surgered.in_edges(treatment))
    G_surgered.remove_edges_from(incoming_to_treatment)

    # Get ancestors of outcome in the surgered graph
    ancestors_outcome = nx.ancestors(G_surgered, outcome) if outcome in G_surgered else set()

    # Condition 1 & 2 (Relevance + Exclusion):
    # Instruments must be parents of treatment AND not ancestors of outcome
    candidate_instruments = parents_treatment - ancestors_outcome

    # Condition 3 (As-if-random/Exogeneity):
    # Instruments must not be descendants of any ancestor of outcome
    # This ensures Z is not affected by confounders
    descendants_of_ancestors = set()
    for ancestor in ancestors_outcome:
        descendants_of_ancestors.update(nx.descendants(G_surgered, ancestor))

    valid_instruments = candidate_instruments - descendants_of_ancestors

    # Filter to only observed instruments
    return [z for z in valid_instruments if z in observed_constructs]
