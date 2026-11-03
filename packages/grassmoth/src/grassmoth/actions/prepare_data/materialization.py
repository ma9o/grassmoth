"""Recorded readings and the canonical panel assembled from extracted rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from grassmoth.actions.prepare_data.aggregations import _encode_non_continuous
from grassmoth.actions.prepare_data.rows import OBSERVATION_ROW_SCHEMA
from grassmoth.artifacts.observations.preparation import RecordedExtractionSpec
from grassmoth.artifacts.observations.time import ObservationInstant

if TYPE_CHECKING:
    from grassmoth.actions.contracts.io import SourceTable
    from grassmoth.artifacts.observations.preparation import DataPreparationSpec
    from grassmoth.artifacts.observations.spec import ResolvedObservationSpec


def recorded_observations(source: SourceTable, definition: DataPreparationSpec) -> pl.DataFrame:
    """Import measured values with their original anchors, supports and missingness."""
    raw_df = pl.DataFrame(source.rows)
    frames = [
        raw_df.select(
            pl.lit(observation.id).alias("indicator_id"),
            pl.col(extraction.value_column).cast(pl.String).alias("value"),
            pl.col(extraction.anchor_column).cast(pl.String).alias("anchor_time"),
            pl.col(extraction.support_start_column).cast(pl.String).alias("support_start"),
            pl.col(extraction.support_end_column).cast(pl.String).alias("support_end"),
        ).filter(pl.col("anchor_time").is_not_null())
        for variable, observation in zip(
            definition.variables, definition.observation_schema(), strict=True
        )
        if isinstance(extraction := variable.extraction, RecordedExtractionSpec)
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=OBSERVATION_ROW_SCHEMA)


def materialize_panel(rows: pl.DataFrame, definition: DataPreparationSpec) -> pl.DataFrame:
    """Encode observation rows into the canonical panel, including a typed empty table."""
    if rows.is_empty():
        return rows

    data_for_model = _encode_non_continuous(
        rows,
        {
            variable.observation.id: variable.observation.measurement
            for variable in definition.variables
        },
    )
    data_for_model = data_for_model.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("value"),
    )
    return parse_observation_rows(data_for_model, definition.observation_schema())


def parse_observation_rows(
    rows: pl.DataFrame, variables: tuple[ResolvedObservationSpec, ...]
) -> pl.DataFrame:
    """Parse the external row format before histories own scientific admission."""
    if missing := OBSERVATION_ROW_SCHEMA.keys() - set(rows.columns):
        raise ValueError(f"Observation table is missing canonical columns: {sorted(missing)}")
    if unknown := set(rows["indicator_id"]) - {variable.id for variable in variables}:
        raise ValueError(f"Observations have no variable definition: {sorted(unknown)}")
    rows = rows.select(*OBSERVATION_ROW_SCHEMA).with_columns(pl.col("value").cast(pl.Float64))
    for column in ("anchor_time", "support_start", "support_end"):
        if rows.schema[column] == pl.Null:
            rows = rows.with_columns(pl.col(column).cast(pl.Datetime))
        rows = ObservationInstant.normalize_column(rows, column)
        if rows.schema[column].base_type() != pl.Datetime:
            raise ValueError(f"Observation {column} must contain timestamps")
    if rows["anchor_time"].null_count():
        raise ValueError("Observations require non-null anchor_time")
    return rows.sort("indicator_id", "anchor_time")
