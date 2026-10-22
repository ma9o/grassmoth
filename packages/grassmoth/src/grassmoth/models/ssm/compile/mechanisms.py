"""Bind scientific expression references to numerical state and parameter coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.model.construct import CausalEdgeSpec
from grassmoth.artifacts.model.expressions import expression_states, linear_coefficient
from grassmoth.artifacts.model.parameter_context import coefficient_value
from grassmoth.compilation_errors import AggregatedCompileError, IncompleteModelError
from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
from grassmoth.models.structure import selected_edges, selected_state_ids

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

    from grassmoth.artifacts.identity import ConstructId
    from grassmoth.artifacts.model.construct import ConstructSpec
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.model.mechanism import DynamicsMechanismSpec
    from grassmoth.models.structure import StructuralSelection


def _is_projected_loading(
    owner: ConstructSpec | CausalEdgeSpec,
    mechanism: DynamicsMechanismSpec,
    retained_states: Collection[str],
) -> bool:
    if not isinstance(owner, CausalEdgeSpec) or owner.cause.id in retained_states:
        return False
    weight = coefficient_value(linear_coefficient(mechanism.expression, owner.cause.id))
    if weight is None:
        raise AggregatedCompileError(["Marginalized confounders support fixed linear loadings"])
    return True


def lower_mechanisms(selection: StructuralSelection) -> tuple[ExpressionComponentSpec, ...]:
    """Require executable coverage, then bind every scalar expression without kind dispatch."""
    dynamical_model_spec = selection.dynamical_model_spec
    states = set(selected_state_ids(selection))
    retained_edges = {edge.id for edge in selected_edges(selection)}
    modeled_edges: set[str] = set()
    modeled_nodes: set[str] = set()
    for owner, mechanism in dynamical_model_spec.iter_mechanisms():
        target = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        if target not in states:
            continue
        if _is_projected_loading(owner, mechanism, states):
            continue
        if isinstance(owner, CausalEdgeSpec):
            dependencies = expression_states(mechanism.expression)
            modeled_edges.update(
                edge.id
                for edge in dynamical_model_spec.edges
                if edge.effect.id == owner.effect.id and edge.cause.id in dependencies
            )
        else:
            modeled_nodes.add(owner.id)
    expected_nodes = {
        key
        for key in states
        if dynamical_model_spec.get_construct(key).is_dynamic
        and dynamical_model_spec.get_construct(key).role == "endogenous"
    }
    if modeled_nodes != expected_nodes or modeled_edges != retained_edges:
        raise IncompleteModelError(
            "Mechanisms must cover the retained dynamic states and edges: "
            f"missing states={sorted(expected_nodes - modeled_nodes)}, "
            f"missing edges={sorted(retained_edges - modeled_edges)}"
        )
    return tuple(
        component
        for _, component in iter_mechanism_components(
            dynamical_model_spec, selected_state_ids(selection)
        )
    )


def iter_mechanism_components(
    dynamical_model_spec: DynamicalModelSpec, state_order: Sequence[ConstructId]
) -> Iterator[tuple[DynamicsMechanismSpec, ExpressionComponentSpec]]:
    """Emit a bound expression alongside the exact scientific term that produced it."""
    state_index = {key: index for index, key in enumerate(state_order)}

    for owner, mechanism in dynamical_model_spec.iter_mechanisms():
        target_id = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        if target_id not in state_index:
            continue
        if _is_projected_loading(owner, mechanism, state_index):
            continue
        if isinstance(owner, CausalEdgeSpec):
            target = state_index[owner.effect.id]
        else:
            target = state_index[owner.id]
        yield (
            mechanism,
            ExpressionComponentSpec(
                expression=mechanism.expression,
                target=target,
                state_ids=tuple(state_order),
                source=state_index[owner.cause.id] if isinstance(owner, CausalEdgeSpec) else None,
                kind=mechanism.kind,
            ),
        )
