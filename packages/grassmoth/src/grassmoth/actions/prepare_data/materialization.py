"""Recorded readings and the canonical panel assembled from extracted rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from grassmoth.actions.prepare_data.aggregations import _encode_non_continuous
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.observations.preparation import RecordedExtractionSpec
from grassmoth.artifacts.observations.time import ObservationInstant

if TYPE_CHECKING:
    from grassmoth.actions.contracts.io import SourceTable
    from grassmoth.artifacts.observations.preparation import DataPreparationSpec


def recorded_observations(source: SourceTable, definition: DataPreparationSpec) -> pl.DataFrame:
    """Import measured values with their original anchors, supports and missingness."""
    raw_df = pl.DataFrame(source.rows)
    frames = [
        raw_df.select(
            pl.lit(observation.id).alias("indicator_id"),
            pl.col(extraction.value_column).cast(pl.String).alias("value"),
            pl.col(extraction.anchor_column).cast(pl.String).alias("anchor_time"),
            pl.lit(observation.support_kind.value).alias("support_kind"),
            pl.lit(observation.summary_operator.value).alias("summary_operator"),
            pl.lit(observation.anchor_policy.value).alias("anchor_policy"),
            pl.lit(str(observation.observation_window)).alias("observation_window"),
            pl.col(extraction.support_start_column).cast(pl.String).alias("support_start"),
            pl.col(extraction.support_end_column).cast(pl.String).alias("support_end"),
        ).filter(pl.col("anchor_time").is_not_null())
        for variable, observation in zip(
            definition.variables, definition.observation_schema(), strict=True
        )
        if isinstance(extraction := variable.extraction, RecordedExtractionSpec)
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=ObservationDataset.row_schema())


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
    for column in ("anchor_time", "support_start", "support_end"):
        data_for_model = ObservationInstant.normalize_column(data_for_model, column)
    return data_for_model.sort("indicator_id", "anchor_time")
