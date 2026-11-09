"""Canonical model values projected into the inputs consumed by numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.model.document import entity_document

if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.json_types import JsonObject, JsonValue


def graph_input(dynamical_model_spec: DynamicalModelSpec) -> JsonObject:
    """Serialized directed assumptions used to fingerprint identification inputs."""
    result: dict[str, JsonValue] = {
        "constructs": [
            item.model_dump(
                mode="json",
                include={"id", "name", "reasoning", "role", "temporal_status"},
            )
            for item in dynamical_model_spec.constructs
        ],
        "edges": [
            {
                **item.model_dump(mode="json", include={"id", "reasoning"}),
                "cause_id": item.cause.id,
                "effect_id": item.effect.id,
            }
            for item in dynamical_model_spec.edges
        ],
    }
    return entity_document(result)


def indicator_rows(dynamical_model_spec: DynamicalModelSpec) -> list[JsonObject]:
    """Measurement inputs with ownership derived from canonical containment."""
    return [
        {
            **indicator.model_dump(mode="json", exclude={"likelihood", "reasoning"}),
            "construct_id": construct.id,
        }
        for construct, indicator in dynamical_model_spec.iter_indicators()
    ]


def observation_input(dynamical_model_spec: DynamicalModelSpec) -> JsonObject:
    """Project the model clock and observation definitions used to fingerprint measurement inputs."""
    return entity_document(
        {
            "model_clock": dynamical_model_spec.time.step.source,
            "indicators": indicator_rows(dynamical_model_spec),
        }
    )


def identification_input(dynamical_model_spec: DynamicalModelSpec) -> JsonObject:
    """Project graph and observation inputs that determine causal identification findings."""
    return {
        "graph": graph_input(dynamical_model_spec),
        "observations": observation_input(dynamical_model_spec),
    }


def compilation_input(dynamical_model_spec: DynamicalModelSpec) -> JsonObject:
    """Structure and constants needed by execution, independent of the current law."""
    result: dict[str, JsonValue] = {
        "time": dynamical_model_spec.time.model_dump(mode="json"),
        "parameters": [
            parameter.model_dump(mode="json", exclude={"distribution", "transform", "reasoning"})
            for parameter in dynamical_model_spec.parameters
        ],
        "constructs": [
            construct.model_dump(mode="json", exclude={"distribution"})
            for construct in dynamical_model_spec.constructs
        ],
        "edges": [
            {
                **edge.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause_id": edge.cause.id,
                "effect_id": edge.effect.id,
            }
            for edge in dynamical_model_spec.edges
        ],
    }
    return entity_document(result)


def input_fingerprints(dynamical_model_spec: DynamicalModelSpec) -> dict[str, str]:
    """Content identities of the values supplied to each numerical boundary."""
    from grassmoth.artifacts.identity import scientific_id

    values = {
        "observations": observation_input(dynamical_model_spec),
        "identification": identification_input(dynamical_model_spec),
        "compilation": compilation_input(dynamical_model_spec),
        "belief": dynamical_model_spec.model_dump(mode="json"),
    }
    return {
        purpose: scientific_id("input", ["additive-model-v1", value])
        for purpose, value in values.items()
    }
