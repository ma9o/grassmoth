"""Shared model fixtures and async test helpers."""

from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal, overload

from grassmoth.artifacts.identity import ConstructId, EdgeId, IndicatorId, MechanismId


@overload
def fixture_entity_id(kind: Literal["construct"], initial_identity: str) -> ConstructId: ...


@overload
def fixture_entity_id(kind: Literal["edge"], initial_identity: str) -> EdgeId: ...


@overload
def fixture_entity_id(kind: Literal["indicator"], initial_identity: str) -> IndicatorId: ...


@overload
def fixture_entity_id(kind: Literal["mechanism"], initial_identity: str) -> MechanismId: ...


@overload
def fixture_entity_id(kind: str, initial_identity: str) -> str: ...


def fixture_entity_id(kind: str, initial_identity: str) -> str:
    """Assign reproducible IDs when constructing a new test fixture."""
    return f"{kind}:{sha256(f'{kind}\0{initial_identity}'.encode()).hexdigest()[:20]}"


def invalid_dict_payload(value: object) -> Any:
    return value


def make_model(state_names: list[str], edges: Sequence[tuple[str, str]] = ()):
    """A connected test graph; uncoupled states share an unmeasured downstream outcome."""
    from grassmoth.artifacts.model.construct import CausalEdgeSpec, ConstructSpec
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec

    constructs = {
        name: ConstructSpec.model_validate(
            {
                "id": fixture_entity_id("construct", name),
                "name": name,
                "description": name,
                "role": "endogenous",
                "temporal_status": "time_varying",
                "indicators": [
                    {
                        "observation": {
                            "id": fixture_entity_id("indicator", name + "_obs"),
                            "name": name + "_obs",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        "construct_polarity": "positive",
                    }
                ],
            }
        )
        for name in state_names
    }
    if edges:
        assert set(state_names) == {name for pair in edges for name in pair}
    if not edges:
        outcome = "unmeasured_outcome"
        constructs[outcome] = ConstructSpec(
            id=fixture_entity_id("construct", outcome),
            name=outcome,
            description="A downstream response outside the numerical fixture's measured states.",
            role="endogenous",
            temporal_status="time_varying",
        )
        edges = [(name, outcome) for name in state_names]
    return DynamicalModelSpec.from_entities(
        edges=tuple(
            CausalEdgeSpec(
                id=fixture_entity_id("edge", cause + "->" + effect),
                cause=constructs[cause],
                effect=constructs[effect],
                description=cause + " causes " + effect,
            )
            for cause, effect in edges
        ),
        measurement_clock="1d",
    )


def graph_constructs(payload):
    """Writable construct definitions in an identity-addressed test document."""
    return list(payload["constructs"].values())
