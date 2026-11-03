"""Observation access owns definition matching before numerical execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Self

import numpy as np
from pydantic import ModelWrapValidatorHandler, field_validator, model_validator

from grassmoth.artifacts.arrays import vector_data, vector_mask, vector_values
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.observations.history import ObservationData, ObservationHistory
from grassmoth.artifacts.observations.spec import (
    ObservationDefinitionSpec,
    ResolvedObservationSpec,
)
from grassmoth.artifacts.observations.time import OBSERVATION_TIME_TOLERANCE, ObservationInstant


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
    """A checked numerical history under the definition that owns its readings."""

    observation: ResolvedObservationSpec
    history: ObservationHistory

    @model_validator(mode="wrap")
    @classmethod
    def admit_readings(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Admit raw histories once and preserve already admitted immutable series."""
        if isinstance(value, cls):
            return value
        series = handler(value)
        history, observation = series.history, series.observation
        times = history.times.values
        tolerance = OBSERVATION_TIME_TOLERANCE
        if not np.isfinite(times).all():
            raise ValueError("Observation anchors must be finite")
        if np.any(np.diff(times) <= tolerance):
            raise ValueError(f"Observation {observation.id} has duplicate or unordered anchors")
        raw = vector_data(history.values)
        present = vector_mask(history.values)
        if np.isinf(raw).any() or not np.isfinite(raw[present]).all():
            raise ValueError("Observed values must be finite; use null for missing observations")
        start, end = vector_values(history.support_start), vector_values(history.support_end)
        known_start, known_end = np.isfinite(start), np.isfinite(end)
        if np.isinf(start).any() or np.isinf(end).any() or np.any(known_start != known_end):
            raise ValueError(f"Observation {observation.id} requires paired finite support bounds")
        known = known_start & known_end
        bound = start if observation.anchor_policy == "support_start" else end
        if (
            np.any(present & ~known)
            or np.any(end[known] < start[known] - tolerance)
            or np.any(np.abs(bound[known] - times[known]) > tolerance)
            or (
                observation.support_kind == "interval"
                and np.any(end[known] - start[known] <= tolerance)
            )
        ):
            raise ValueError(f"Observation {observation.id} has invalid measurement support")
        values = raw[present]
        dtype = observation.measurement.dtype
        if dtype == "binary" and not np.isin(values, (0, 1)).all():
            raise ValueError(f"Binary observation {observation.id} requires values 0 or 1")
        if dtype == "count" and np.any((values < 0) | (values != np.floor(values))):
            raise ValueError(f"Count observation {observation.id} requires non-negative integers")
        if observation.levels and np.any(
            (values < 0) | (values >= len(observation.levels)) | (values != np.floor(values))
        ):
            raise ValueError(f"Observation {observation.id} has values outside its codebook")
        return series


class SelectedObservations(Value):
    """The requested variables in request order, retaining their matching definitions."""

    series: tuple[ObservationSeries, ...]
    time_origin: datetime


class ObservationDataset(Value):
    """Recorded variables; numerical consumers select their own required definitions."""

    series: Mapping[IndicatorId, ObservationSeries]
    time_origin: datetime

    @field_validator("time_origin")
    @classmethod
    def utc_origin(cls, value: datetime) -> datetime:
        """Own the UTC calendar origin used to interpret model days."""
        return ObservationInstant(value).value

    @model_validator(mode="wrap")
    @classmethod
    def named_series(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Match map identities to their already admitted series."""
        if isinstance(value, cls):
            return value
        dataset = handler(value)
        if any(identity != series.observation.id for identity, series in dataset.series.items()):
            raise ValueError("Observation map keys must equal their series identities")
        return dataset

    @classmethod
    def from_histories(
        cls,
        histories: ObservationData,
        variables: tuple[ResolvedObservationSpec, ...],
        *,
        time_origin: datetime,
    ) -> ObservationDataset:
        """Admit exactly the histories supplied under their recorded definitions."""
        identities = {variable.id for variable in variables}
        if len(identities) != len(variables):
            raise ValueError("Observation variables must have unique identities")
        if identities != set(histories):
            raise ValueError("Observation histories must match their variable definitions exactly")
        return cls(
            series={
                variable.id: ObservationSeries(observation=variable, history=histories[variable.id])
                for variable in variables
            },
            time_origin=time_origin,
        )

    def select(
        self, required: tuple[ResolvedObservationSpec, ...]
    ) -> SelectedObservations | ObservationSelectionFailure:
        """Select by identity and meaning, before handing histories to execution."""
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
