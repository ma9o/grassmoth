"""Explicit question and time domains for small scientific test models."""

from typing import TypedDict

from grassmoth.artifacts.identity import ConstructId
from grassmoth.artifacts.model.construct import ConstructSpec
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.time import ModelTimeSpec
from grassmoth.artifacts.simulation import SimulationSpec


class ModelMetadata(TypedDict):
    question: str
    outcome: ConstructId
    time: ModelTimeSpec
    core_query: SimulationSpec


def model_metadata(
    outcome, *, question="Test model.", start="2024-01-01", horizon="2d", step="1d", core_query=None
) -> ModelMetadata:
    time = ModelTimeSpec.model_validate(
        {"window": {"start": start, "horizon": horizon}, "step": step}
    )
    return {
        "question": question,
        "outcome": ConstructId(outcome),
        "time": time,
        "core_query": core_query if core_query is not None else SimulationSpec(window=time.window),
    }


def initial_model(
    *, question="What does it answer?", outcome="construct:outcome", core_query=None, time=None
):
    construct = ConstructSpec.model_validate(
        {
            "id": outcome,
            "name": str(outcome).removeprefix("construct:"),
            "reasoning": {"text": question},
            "role": "endogenous",
            "temporal_status": "time_varying",
        }
    )
    metadata: ModelMetadata = model_metadata(outcome, question=question, core_query=core_query)
    if time is not None:
        metadata = {**metadata, "time": time}
    return DynamicalModelSpec.from_entities(constructs=(construct,), **metadata)
