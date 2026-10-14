"""The user's question in model terms, established before model authoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.simulation import SimulationSpec

type QueryName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QuestionSpec(Value):
    """What the study asks: the user's words, the outcome, and named contrasts.

    Each query is a contrast of its interventions against the recorded course.
    Constructs are named by identity before a model defines them.
    """

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        description="The user's question in their own words."
    )
    outcome: ConstructId = Field(description="The construct whose course answers the question.")
    queries: Mapping[QueryName, SimulationSpec] = Field(
        default_factory=dict,
        description="Named contrasts against the recorded course, each with at least one intervention.",
    )

    @model_validator(mode="after")
    def validate_queries(self) -> Self:
        """Require an intervention contrast for every query and prohibit intervening on the outcome."""
        for name, query in self.queries.items():
            if not query.interventions:
                raise ValueError(f"Query {name!r} needs an intervention to contrast")
            if any(event.target == self.outcome for event in query.interventions):
                raise ValueError(f"Query {name!r} intervenes on the outcome")
        return self

    @property
    def targets(self) -> frozenset[ConstructId]:
        """Distinct construct IDs targeted by the question's intervention queries."""
        return frozenset(
            event.target for query in self.queries.values() for event in query.interventions
        )
