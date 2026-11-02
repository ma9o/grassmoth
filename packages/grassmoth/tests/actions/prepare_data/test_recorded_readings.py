"""Recorded-reading preparation preserves support and missingness without aggregation."""

import polars as pl
import pytest

from grassmoth.actions.contracts.io import SourceTable
from grassmoth.actions.prepare_data.materialization import (
    materialize_panel,
    recorded_observations,
)
from grassmoth.artifacts.observations.preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    RecordedExtractionSpec,
)
from grassmoth.artifacts.observations.rows import validate_observation_rows
from grassmoth.artifacts.observations.spec import AuthoredObservationSpec, NumericMeasurement

pytestmark = pytest.mark.contract


def test_recorded_readings_preserve_irregular_overlapping_support_and_missing_values():
    variables = tuple(
        DataVariableSpec(
            observation=AuthoredObservationSpec(
                id="indicator:" + name,
                name=name,
                measurement=NumericMeasurement(dtype="continuous"),
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
    definition = DataPreparationSpec(default_window="1d", variables=variables)
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
    rows = recorded_observations(
        SourceTable(rows=data.to_arrow(), start=None, end=None), definition
    )
    panel = validate_observation_rows(
        materialize_panel(rows, definition), definition.observation_schema()
    )
    assert panel.height == 6
    point = panel.filter(pl.col("indicator_id") == "indicator:point")
    intervals = panel.filter(pl.col("indicator_id") == "indicator:interval")
    assert point["value"].to_list() == [2.0, None, 4.0]
    assert (point["support_start"] == point["support_end"]).all()
    assert intervals["support_start"][1] < intervals["support_end"][0]
    assert intervals["anchor_time"].dt.hour().to_list() == [3, 5, 8]
