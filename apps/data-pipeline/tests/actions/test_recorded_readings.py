"""Recorded-reading preparation preserves support and missingness without aggregation."""

import polars as pl
import pytest

from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
from nof1_causal_lab.actions.prepare_data.materialization import materialize_panel
from nof1_causal_lab.artifacts.observations.preparation import (
    DataVariableSpec,
    FileSourceRef,
    RecordedExtractionSpec,
)
from nof1_causal_lab.artifacts.observations.rows import validate_observation_rows
from nof1_causal_lab.artifacts.observations.spec import AuthoredObservationSpec
from nof1_causal_service.actions.prepare_data.materialization import recorded_observations

pytestmark = pytest.mark.contract


def test_recorded_readings_preserve_irregular_overlapping_support_and_missing_values():
    variables = tuple(
        DataVariableSpec(
            observation=AuthoredObservationSpec(
                id="indicator:" + name,
                name=name,
                measurement_dtype="continuous",
                aggregation=aggregation,
                observation_window="1d",
            ),
            extraction=RecordedExtractionSpec(
                value_column=name, support_start_column=start, support_end_column="timestamp"
            ),
        )
        for name, aggregation, start in (
            ("point", "last", "timestamp"),
            ("interval", "mean", "start"),
        )
    )
    context = MeasurementContext(
        source=FileSourceRef(files=("input/recorded.csv",)), model_clock="1d", indicators=variables
    )
    data = pl.DataFrame(
        {
            "timestamp": [
                "2020-01-01T04:12:00+01:00",
                "2020-01-01T06:51:00+01:00",
                "2020-01-01T09:13:00+01:00",
            ],
            "start": ["2020-01-01T01:00:00Z", "2020-01-01T02:10:00Z", "2020-01-01T04:30:00Z"],
            "point": [2.0, None, 4.0],
            "interval": [1.0, 2.0, 3.0],
        }
    )
    rows = recorded_observations(data, context)
    panel = validate_observation_rows(
        materialize_panel(rows, context),
        tuple(v.observation.resolved(context.window(v)) for v in variables),
    )
    assert panel.height == 6
    point = panel.filter(pl.col("indicator_id") == "indicator:point")
    intervals = panel.filter(pl.col("indicator_id") == "indicator:interval")
    assert point["value"].to_list() == [2.0, None, 4.0]
    assert (point["support_start"] == point["support_end"]).all()
    assert intervals["support_start"][1] < intervals["support_end"][0]
    assert intervals["anchor_time"].dt.hour().to_list() == [3, 5, 8]
