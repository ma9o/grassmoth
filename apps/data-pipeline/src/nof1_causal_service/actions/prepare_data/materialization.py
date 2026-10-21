"""extraction materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
    from nof1_causal_lab.artifacts.observations.records import ObservationRecord


def recorded_observations(
    raw_df: pl.DataFrame,
    measurement_structure: MeasurementContext,
) -> list[ObservationRecord]:
    """Import measured values with their original anchors, supports and missingness."""
    import polars as pl
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.observations.preparation import RecordedExtractionSpec
    from nof1_causal_lab.artifacts.observations.records import ObservationRecord

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
