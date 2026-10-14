"""Extraction context composes owned definitions, observation windows and source references."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.observations.duration import Duration
from nof1_causal_lab.artifacts.observations.preparation import (
    DataVariableSpec,
    FilePreparationSpec,
    FileSourceRef,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.observations.spec import ObservationWindow


class MeasurementContext(Value):
    """One extraction selection, retaining its authored variables and source span."""

    source: FileSourceRef
    model_clock: Duration
    indicators: tuple[DataVariableSpec, ...] = Field(min_length=1)

    def window(self, indicator: DataVariableSpec) -> ObservationWindow:
        """Resolve an indicator's authored observation window, using the model clock when omitted."""
        return indicator.observation.observation_window or self.model_clock

    def select(self, indicator: DataVariableSpec) -> MeasurementContext:
        """A chunk carries the same source and one actual observation definition."""
        return MeasurementContext(
            source=self.source, model_clock=self.model_clock, indicators=(indicator,)
        )


def extraction_context(preparation: FilePreparationSpec) -> MeasurementContext:
    """Project the captured source, model clock, and variable definitions into worker context."""
    return MeasurementContext(
        source=preparation.source,
        model_clock=preparation.definition.default_window,
        indicators=preparation.definition.variables,
    )
