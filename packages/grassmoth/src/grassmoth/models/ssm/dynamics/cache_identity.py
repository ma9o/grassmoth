"""Deterministic scientific-expression descriptions for runtime cache keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from grassmoth.artifacts.model.expressions import MechanismKind
from grassmoth.json_types import JsonObject
from grassmoth.models.ssm.dynamics.expression import EdgeDrift

if TYPE_CHECKING:
    from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
    from grassmoth.models.ssm.dynamics.spec import DynamicsSpec


class ComponentDescription(TypedDict):
    """Serialized drift or potential expression with its target and compiled state-axis identities."""

    kind: MechanismKind
    expression: JsonObject
    target: int
    source: int | None
    state_ids: list[str]


class DynamicsDescription(TypedDict):
    """Serialized vector-field dimension and ordered component descriptions."""

    n_latent: int
    components: list[ComponentDescription]


def _component_description(component: ExpressionComponentSpec) -> ComponentDescription:
    """Describe an edge's term as a drift with its source state."""
    role = component.role
    return {
        "kind": "drift" if isinstance(role, EdgeDrift) else role,
        "expression": component.expression.model_dump(mode="json"),
        "target": component.target,
        "source": role.source if isinstance(role, EdgeDrift) else None,
        "state_ids": list(component.state_ids),
    }


def dynamics_spec_to_dict(spec: DynamicsSpec) -> DynamicsDescription:
    """Describe bound expressions without persisting an executable model."""
    return {
        "n_latent": spec.n_latent,
        "components": [_component_description(component) for component in spec.components],
    }
