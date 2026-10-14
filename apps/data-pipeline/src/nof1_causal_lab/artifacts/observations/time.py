"""Calendar instants and relative model days meet only at this binding owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class ObservationInstant:
    """One UTC instant; canonical panel datetimes are UTC with the zone omitted."""

    value: datetime

    def __post_init__(self) -> None:
        """Normalize observation instants to UTC, interpreting timezone-naive values as UTC."""
        utc = (
            self.value.replace(tzinfo=UTC)
            if self.value.tzinfo is None
            else self.value.astimezone(UTC)
        )
        object.__setattr__(self, "value", utc)

    def relative_to(self, origin: ObservationInstant) -> ModelTime:
        """Express this observation instant as fractional model days after the supplied origin."""
        return ModelTime((self.value - origin.value).total_seconds() / 86400)


@dataclass(frozen=True)
class ModelTime:
    """A relative time on the engine's day axis, distinct from a calendar instant."""

    days: float

    def at(self, origin: ObservationInstant) -> ObservationInstant:
        """Convert relative model days to an observation instant using the supplied calendar origin."""
        return ObservationInstant(origin.value + timedelta(days=self.days))

    @staticmethod
    def bind_column(instants: pl.Expr, origin: ObservationInstant) -> pl.Expr:
        """Lower a canonical UTC-naive datetime column to native model-day numbers."""
        return (
            (instants - pl.lit(origin.value.replace(tzinfo=None))).dt.total_microseconds()
            / 86400000000
        ).cast(pl.Float64)
