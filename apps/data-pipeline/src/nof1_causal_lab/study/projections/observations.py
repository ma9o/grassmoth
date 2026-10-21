"""Pure projections of retained observation and numerical values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.arrays import resolve_vector
from nof1_causal_lab.artifacts.observations.comparison import DataSeries, Dataset
from nof1_causal_lab.artifacts.observations.rows import observation_row_schema
from nof1_causal_lab.artifacts.observations.time import ModelTime, ObservationInstant
from nof1_causal_lab.artifacts.reports.data_comparison import DataPoint

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.actions.contracts.bodies import PrepareDataOutput
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.artifacts.observations.reference import DataRef


def prepared_frame(result: PrepareDataOutput) -> pl.DataFrame:
    """Decode the preparation's saved history into the scientific observation table."""
    rows = []
    for variable in result.metadata.variables:
        history = result.data[variable.id]
        origin = ObservationInstant(result.metadata.time_origin)

        def instant(day: float | None, origin: ObservationInstant = origin) -> datetime | None:
            return ModelTime(day).at(origin).value.replace(tzinfo=None) if day is not None else None

        starts = resolve_vector(history.support_start)
        ends = resolve_vector(history.support_end)
        values = resolve_vector(history.values)
        rows.extend(
            {
                "indicator_id": variable.id,
                "value": value,
                "anchor_time": instant(day),
                "support_start": instant(start),
                "support_end": instant(end),
                "support_kind": variable.support_kind.value,
                "summary_operator": variable.summary_operator.value,
                "anchor_policy": variable.anchor_policy.value,
                "observation_window": str(variable.observation_window),
            }
            for day, start, end, value in zip(history.times, starts, ends, values, strict=True)
        )
    schema = observation_row_schema() | {
        "value": pl.Float64,
        **{name: pl.Datetime("us") for name in ("anchor_time", "support_start", "support_end")},
    }
    return pl.DataFrame(rows, schema=schema).sort("indicator_id", "anchor_time")


def read_dataset(
    source: DataRef[GitOid, int],
    observations: ObservationDataset,
) -> Dataset:
    """Project selected observations into per-indicator dated series with their exact source reference."""
    frame = observations.recorded.frame.with_columns(
        pl.col("anchor_time", "support_start", "support_end").dt.replace_time_zone("UTC")
    )
    series = {
        variable.id: DataSeries(
            variable=variable,
            points=tuple(
                DataPoint(anchor_time=anchor, support_start=start, support_end=end, value=value)
                for anchor, start, end, value in frame.filter(pl.col("indicator_id") == variable.id)
                .select("anchor_time", "support_start", "support_end", "value")
                .iter_rows()
            ),
        )
        for variable in observations.variables
    }

    return Dataset(source=source, time_origin=observations.time_origin, series=series)
