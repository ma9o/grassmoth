"""extraction materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from grassmoth_server.actions.prepare_data.context import MeasurementContext
    from grassmoth_server.actions.prepare_data.records import ObservationRecord


def recorded_observations(
    raw_df: pl.DataFrame,
    measurement_structure: MeasurementContext,
) -> list[ObservationRecord]:
    """Import measured values with their original anchors, supports and missingness."""
    import polars as pl
    from pydantic import TypeAdapter

    from grassmoth.public import RecordedExtractionSpec
    from grassmoth_server.actions.prepare_data.records import ObservationRecord

    rows: list[ObservationRecord] = []
    for indicator in measurement_structure.indicators:
        extraction = indicator.extraction
        if not isinstance(extraction, RecordedExtractionSpec):
            continue
        observation = indicator.observation.resolved(measurement_structure.window(indicator))
        frame = raw_df.select(
            pl.lit(observation.id).alias("indicator_id"),
            pl.col(extraction.value_column).cast(pl.String).alias("value"),
            pl.col(extraction.anchor_column).cast(pl.String).alias("anchor_time"),
            pl.col(extraction.support_start_column).cast(pl.String).alias("support_start"),
            pl.col(extraction.support_end_column).cast(pl.String).alias("support_end"),
            pl.lit(observation.support_kind.value).alias("support_kind"),
            pl.lit(observation.summary_operator.value).alias("summary_operator"),
            pl.lit(observation.anchor_policy.value).alias("anchor_policy"),
            pl.lit(str(observation.observation_window)).alias("observation_window"),
        ).filter(pl.col("anchor_time").is_not_null())
        rows.extend(TypeAdapter(list[ObservationRecord]).validate_python(frame.to_dicts()))
    return rows


def materialize_panel(
    observation_rows: list[ObservationRecord],
    measurement_structure: MeasurementContext,
) -> pl.DataFrame:
    """Encode observation rows into the canonical panel, including a typed empty table."""
    from grassmoth.public import ObservationDataset, ObservationInstant
    from grassmoth_server.actions.prepare_data.aggregations import _encode_non_continuous

    if observation_rows:
        data_for_model = pl.DataFrame(observation_rows)
    else:
        data_for_model = pl.DataFrame(schema=ObservationDataset.row_schema())

    if len(data_for_model) > 0:
        dtype_lookup: dict[str, str] = {
            indicator.observation.id: indicator.observation.measurement_dtype
            for indicator in measurement_structure.indicators
        }
        ordinal_levels_lookup: dict[str, tuple[str, ...]] = {
            ind.observation.id: levels
            for ind in measurement_structure.indicators
            if (levels := ind.observation.ordinal_levels)
        }
        categorical_levels_lookup: dict[str, tuple[str, ...]] = {
            ind.observation.id: levels
            for ind in measurement_structure.indicators
            if (levels := ind.observation.categorical_levels)
        }
        data_for_model = _encode_non_continuous(
            data_for_model, dtype_lookup, ordinal_levels_lookup, categorical_levels_lookup
        )
        data_for_model = data_for_model.with_columns(
            pl.col("value").cast(pl.Float64, strict=False).alias("value"),
        )
        for column in ("anchor_time", "support_start", "support_end"):
            data_for_model = ObservationInstant.normalize_column(data_for_model, column)
        data_for_model = data_for_model.sort("indicator_id", "anchor_time")

    return data_for_model
