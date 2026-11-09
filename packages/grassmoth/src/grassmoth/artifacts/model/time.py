"""Calendar windows and the fixed state grid owned by a scientific model."""

from datetime import UTC, date, datetime, time, timedelta
from typing import Self

from pydantic import Field, model_validator

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.observations.duration import Duration
from grassmoth.json_types import JsonObject


class TimeWindowSpec(Value):
    """A calendar start and positive elapsed duration, including both state endpoints."""

    start: date = Field(description="Calendar start at midnight UTC.")
    horizon: Duration = Field(description="Elapsed duration from the start to the final state.")

    @property
    def start_instant(self) -> datetime:
        """Calendar origin at midnight UTC."""
        return datetime.combine(self.start, time(), tzinfo=UTC)

    @property
    def end_instant(self) -> datetime:
        """Calendar instant of the final state."""
        return self.start_instant + timedelta(seconds=self.horizon.seconds)

    def start_day(self, origin: datetime) -> float:
        """Start expressed in days from an explicit calendar origin."""
        return (self.start_instant - origin).total_seconds() / 86400

    def end_day(self, origin: datetime) -> float:
        """Final state expressed in days from an explicit calendar origin."""
        return self.start_day(origin) + self.horizon.days


class TimeGridFailure(Value):
    """An authored evaluation window cannot use the model's fixed state grid."""

    reason: str


class ModelTimeSpec(Value):
    """The fitting window, initial-state origin, and regular state interval."""

    window: TimeWindowSpec
    step: Duration = Field(
        description="Fixed state interval and default measurement and coefficient-authoring interval."
    )

    @model_validator(mode="after")
    def aligned_horizon(self) -> Self:
        """The fitting span contains a whole number of state transitions."""
        if self.window.horizon.seconds % self.step.seconds:
            raise ValueError("Model horizon must be a whole number of time steps")
        return self

    @property
    def time_points(self) -> tuple[float, ...]:
        """Regular model-day coordinates, including initial and final states."""
        return tuple(
            index * self.step.days
            for index in range(self.window.horizon.seconds // self.step.seconds + 1)
        )

    @classmethod
    def from_document(cls, document: JsonObject) -> Self:
        """Resolve merged time fields at their authoring boundary."""
        return cls.model_validate(document)

    def query_grid(self, window: TimeWindowSpec) -> tuple[float, ...] | TimeGridFailure:
        """Admit a query window onto this grid, allowing extension into the future."""
        offset = (window.start_instant - self.window.start_instant).total_seconds()
        if offset < 0:
            return TimeGridFailure(reason="Query cannot start before the model's initial state")
        if offset % self.step.seconds or window.horizon.seconds % self.step.seconds:
            return TimeGridFailure(
                reason="Query start and horizon must align with the model time step"
            )
        first = int(offset // self.step.seconds)
        count = window.horizon.seconds // self.step.seconds
        return tuple(index * self.step.days for index in range(first, first + count + 1))
