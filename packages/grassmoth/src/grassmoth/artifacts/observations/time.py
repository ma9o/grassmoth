"""Calendar instants and relative model days meet only at this binding owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from grassmoth.artifacts.observations.duration import CalendarDuration
from grassmoth.artifacts.observations.semantics import AnchorPolicy

if TYPE_CHECKING:
    from grassmoth.artifacts.observations.spec import ObservationWindow


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

    @staticmethod
    def normalize_column(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        """Normalize the raw temporal axis to UTC-naive datetimes."""
        dtype = df.schema[time_col]
        value = pl.col(time_col)
        if dtype == pl.String:
            value = value.str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
        elif dtype == pl.Date:
            value = value.cast(pl.Datetime)
        elif isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return df.with_columns(value.alias(time_col))

    @staticmethod
    def support_bounds(
        anchor: pl.Expr, window: ObservationWindow, *, anchor_policy: AnchorPolicy
    ) -> tuple[pl.Expr, pl.Expr]:
        """Resolve UTC support bounds; calendar windows exist only at calendar boundaries."""
        if isinstance(window, CalendarDuration):
            anchor = pl.when(anchor == anchor.dt.truncate(window.polars_interval)).then(anchor)
        if anchor_policy == AnchorPolicy.SUPPORT_END:
            return anchor.dt.offset_by("-" + window.polars_interval), anchor
        return anchor, anchor.dt.offset_by(window.polars_interval)


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
