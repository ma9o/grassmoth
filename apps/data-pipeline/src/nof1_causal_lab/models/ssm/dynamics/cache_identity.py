"""Deterministic scientific-expression descriptions for runtime cache keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from nof1_causal_lab.json_types import JsonObject

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec


class ComponentDescription(TypedDict):
    """Serialized drift or potential expression with its target and compiled state-axis identities."""

    kind: Literal["drift", "potential"]
    expression: JsonObject
    target: int
    source: int | None
    state_ids: list[str]


class DynamicsDescription(TypedDict):
    """Serialized vector-field dimension and ordered component descriptions."""

    n_latent: int
    components: list[ComponentDescription]


def dynamics_spec_to_dict(spec: DynamicsSpec) -> DynamicsDescription:
    """Describe bound expressions without persisting an executable model."""
    return {
        "n_latent": spec.n_latent,
        "components": [
            {
                "kind": component.kind,
                "expression": component.expression.model_dump(mode="json"),
                "target": component.target,
                "source": component.source,
                "state_ids": list(component.state_ids),
            }
            for component in spec.components
        ],
    }
