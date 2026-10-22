"""Shared structural validation of prepared and compared observation histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from grassmoth.artifacts.observations.duration import CalendarDuration
from grassmoth.artifacts.observations.time import ObservationInstant

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.artifacts.observations.spec import (
        ObservationWindow,
    )


def observation_window_boundaries(
    window: ObservationWindow, start: datetime, end: datetime
) -> tuple[datetime, ...]:
    """Calendar boundaries that a simulation grid must retain inside its requested span."""
    if not isinstance(window, CalendarDuration):
        return ()
    first = pl.Series([start]).dt.truncate(window.polars_interval)[0]
    instants: list[datetime] = pl.datetime_range(
        first, end, interval=window.polars_interval, eager=True
    ).to_list()
    return tuple(instant for instant in instants if instant >= start)


def pivot_to_wide(df: pl.DataFrame, *, time_origin: datetime) -> pl.DataFrame:
    """Pivot recorded observations onto a shared model-day axis.

    Args:
        df: Long-format rows with ``indicator_id``, ``value``, and ``anchor_time``.
        time_origin: Calendar instant of model day zero, or ``None`` to use the
            calendar-free origin owned by ``ObservationInstant``.

    Returns:
        A frame with a fractional-day ``time`` column and one numeric column per
        indicator. Repeated indicator/anchor pairs are averaged; missing cells
        remain null. Empty input returns an empty frame.

    Raises:
        ValueError: Nonempty input has no ``anchor_time`` column.
    """
    if df.is_empty():
        return pl.DataFrame()

    time_col = "anchor_time"
    if time_col not in df.columns:
        raise ValueError("Observation data must include an 'anchor_time' column.")

    # Parse string timestamps to datetime before pivoting so the
    # datetime→fractional-days conversion below always triggers.
    df = ObservationInstant.normalize_column(df, time_col)

    wide_data = (
        df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .pivot(on="indicator_id", index=time_col, values="value", aggregate_function="mean")
        .sort(time_col)
    )

    if wide_data.schema[time_col].base_type() in (pl.Datetime, pl.Date):
        from grassmoth.artifacts.observations.time import ModelTime

        origin = ObservationInstant(time_origin)
        wide_data = wide_data.with_columns(
            ModelTime.bind_column(pl.col(time_col), origin).alias(time_col)
        )

    if time_col in wide_data.columns:
        wide_data = wide_data.rename({time_col: "time"})

    return wide_data
