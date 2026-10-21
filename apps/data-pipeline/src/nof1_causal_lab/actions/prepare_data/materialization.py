"""extraction materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
    from nof1_causal_lab.artifacts.observations.records import ObservationRecord


def materialize_panel(
    observation_rows: list[ObservationRecord],
    measurement_structure: MeasurementContext,
) -> pl.DataFrame:
    """Encode observation rows into the canonical panel, including a typed empty table."""
    from nof1_causal_lab.artifacts.observations.rows import observation_row_schema
    from nof1_causal_lab.observations.aggregations import _encode_non_continuous
    from nof1_causal_lab.observations.rows import ensure_datetime_column

    if observation_rows:
        data_for_model = pl.DataFrame(observation_rows)
    else:
        data_for_model = pl.DataFrame(schema=observation_row_schema())

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
            data_for_model = ensure_datetime_column(data_for_model, column)
        data_for_model = data_for_model.sort("indicator_id", "anchor_time")

    return data_for_model
