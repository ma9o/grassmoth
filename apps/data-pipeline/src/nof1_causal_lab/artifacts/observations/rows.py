"""Shared structural validation of prepared and compared observation histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.observations.spec import (
    canonical_observation_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec


OBSERVATION_ROW_SCHEMA = {
    "indicator_id": pl.Utf8,
    "value": pl.Utf8,
    "anchor_time": pl.Utf8,
    "support_kind": pl.Utf8,
    "summary_operator": pl.Utf8,
    "anchor_policy": pl.Utf8,
    "observation_window": pl.Utf8,
    "support_start": pl.Utf8,
    "support_end": pl.Utf8,
}


def validate_observation_rows(
    data: pl.DataFrame, variables: Sequence[ResolvedObservationSpec]
) -> pl.DataFrame:
    """Normalize UTC timestamps and validate values/support without filling or encoding."""
    columns = observation_row_schema()
    if missing := columns.keys() - set(data.columns):
        raise ValueError(f"Observation table is missing canonical columns: {sorted(missing)}")
    identities = {variable.id for variable in variables}
    if len(identities) != len(variables):
        raise ValueError("Observation variables must have unique identities")
    if unknown := set(data["indicator_id"]) - identities:
        raise ValueError(f"Observations have no variable definition: {sorted(unknown)}")
    data = data.select(*columns).with_columns(pl.col("value").cast(pl.Float64, strict=True))
    for column in ("anchor_time", "support_start", "support_end"):
        dtype = data.schema[column]
        value = pl.col(column)
        if dtype == pl.String:
            value = value.str.to_datetime(time_zone="UTC")
        elif dtype == pl.Date or dtype == pl.Null:
            value = value.cast(pl.Datetime).dt.replace_time_zone("UTC")
        elif isinstance(dtype, pl.Datetime):
            value = (
                value.dt.convert_time_zone("UTC")
                if dtype.time_zone is not None
                else value.dt.replace_time_zone("UTC")
            )
        else:
            raise ValueError(f"Observation {column} must contain timestamps")
        data = data.with_columns(value.dt.replace_time_zone(None).alias(column))
    if data["anchor_time"].null_count():
        raise ValueError("Observations require non-null anchor_time")
    if not data["value"].drop_nulls().is_finite().all():
        raise ValueError("Observed values must be finite; use null for missing observations")
    for variable in variables:
        rows = data.filter(pl.col("indicator_id") == variable.id)
        if rows["anchor_time"].n_unique() != rows.height:
            raise ValueError(f"Observation {variable.id} has duplicate anchors within a history")
        for field, expected in (
            ("support_kind", variable.support_kind.value),
            ("summary_operator", variable.summary_operator.value),
            ("anchor_policy", variable.anchor_policy.value),
        ):
            if rows[field].null_count() or not (rows[field] == expected).all():
                raise ValueError(f"Observation {variable.id} has inconsistent {field}")
        if any(
            window is None or canonical_observation_window(window) != variable.definition.window
            for window in rows["observation_window"].unique()
        ):
            raise ValueError(f"Observation {variable.id} has an incompatible observation window")
        start, end, anchor = pl.col("support_start"), pl.col("support_end"), pl.col("anchor_time")
        bound = start if variable.anchor_policy == "support_start" else end
        invalid = (
            (end < start)
            | (bound != anchor)
            | (start.is_null() != end.is_null())
            | (pl.col("value").is_not_null() & (start.is_null() | end.is_null()))
        )
        if variable.support_kind == "interval":
            invalid |= end == start
        if rows.select(invalid.any()).item():
            raise ValueError(f"Observation {variable.id} has invalid measurement support")
        values = rows["value"].drop_nulls()
        dtype = variable.measurement_dtype
        if dtype == "binary" and not values.is_in([0, 1]).all():
            raise ValueError(f"Binary observation {variable.id} requires values 0 or 1")
        if dtype == "count" and ((values < 0) | (values != values.floor())).any():
            raise ValueError(f"Count observation {variable.id} requires non-negative integers")
        levels = variable.ordinal_levels or variable.categorical_levels
        if (
            levels is not None
            and ((values < 0) | (values >= len(levels)) | (values != values.floor())).any()
        ):
            raise ValueError(f"Observation {variable.id} has values outside its codebook")
    return data.sort("indicator_id", "anchor_time")


def observation_row_schema() -> dict[str, pl.DataType | type[pl.DataType]]:
    """Schema for canonical long-format observation rows."""
    return dict(OBSERVATION_ROW_SCHEMA)
