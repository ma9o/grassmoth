"""Observation access owns definition matching before numerical execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import polars as pl
import pyarrow as pa
from pydantic import ConfigDict

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.observations.rows import (
    observation_row_schema,
    validate_observation_rows,
)
from nof1_causal_lab.artifacts.observations.spec import (
    ObservationDefinitionSpec,
    ResolvedObservationSpec,
)


class MissingObservation(Value):
    """A required model indicator for which the selected data supplies no observation definition."""

    indicator_id: IndicatorId

    @property
    def message(self) -> str:
        """Explanation identifying the required indicator missing from the selected observations."""
        return f"No observations define required indicator {self.indicator_id}"


class ObservationDefinitionMismatch(Value):
    """Conflicting model and recorded measurement definitions for the same observation identity."""

    indicator_id: IndicatorId
    expected: ObservationDefinitionSpec
    recorded: ObservationDefinitionSpec

    @property
    def message(self) -> str:
        """Explanation retaining both expected and recorded definitions for the incompatible indicator."""
        return (
            f"Observation {self.indicator_id} has incompatible measurement definitions: "
            f"expected {self.expected.model_dump_json()}, recorded {self.recorded.model_dump_json()}"
        )


type ObservationSelectionFailure = MissingObservation | ObservationDefinitionMismatch


class ObservationSeries(Value):
    """An immutable table with the definition under which its values were recorded."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: ResolvedObservationSpec
    rows: pa.Table


class SelectedObservations(Value):
    """The requested variables in request order, retaining their matching definitions."""

    series: tuple[ObservationSeries, ...]
    time_origin: datetime

    @property
    def frame(self) -> pl.DataFrame:
        """Selected observation rows concatenated into a frame, preserving the schema when empty."""
        return (
            pl.concat(tuple(pl.DataFrame(series.rows) for series in self.series))
            if self.series
            else pl.DataFrame(schema=observation_row_schema()).with_columns(
                pl.col("value").cast(pl.Float64)
            )
        )


class ObservationDataset(Value):
    """Recorded variables; numerical consumers select their own required definitions."""

    series: Mapping[IndicatorId, ObservationSeries]
    time_origin: datetime

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        variables: tuple[ResolvedObservationSpec, ...],
        *,
        time_origin: datetime,
    ) -> ObservationDataset:
        """Parse external rows once under their recorded observation definitions."""
        frame = validate_observation_rows(frame, variables)
        return cls(
            series={
                variable.id: ObservationSeries(
                    observation=variable,
                    rows=frame.filter(pl.col("indicator_id") == variable.id).to_arrow(),
                )
                for variable in variables
            },
            time_origin=time_origin,
        )

    def select(
        self, required: tuple[ResolvedObservationSpec, ...]
    ) -> SelectedObservations | ObservationSelectionFailure:
        """Select by identity and meaning, before handing rows to execution."""
        for observation in required:
            recorded = self.series.get(observation.id)
            if recorded is None:
                return MissingObservation(indicator_id=observation.id)
            if recorded.observation.definition != observation.definition:
                return ObservationDefinitionMismatch(
                    indicator_id=observation.id,
                    expected=observation.definition,
                    recorded=recorded.observation.definition,
                )
        return SelectedObservations(
            series=tuple(self.series[observation.id] for observation in required),
            time_origin=self.time_origin,
        )

    @property
    def variables(self) -> tuple[ResolvedObservationSpec, ...]:
        """Resolved observation definitions in the dataset's series order."""
        return tuple(series.observation for series in self.series.values())

    @property
    def recorded(self) -> SelectedObservations:
        """The original histories for profiling and comparison, without a model reinterpretation."""
        return SelectedObservations(
            series=tuple(self.series.values()), time_origin=self.time_origin
        )
