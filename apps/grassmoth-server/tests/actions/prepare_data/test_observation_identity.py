"""Observation subjects remain stable and invalid references fail at production."""

from datetime import datetime

import polars as pl
import pytest

from grassmoth.artifacts.observations.spec import (
    AuthoredObservationSpec,
    ResolvedObservationSpec,
)
from grassmoth.observations.quality.profile import validate_extraction
from grassmoth_server.actions.prepare_data.aggregations import compute_indicators
from grassmoth_server.actions.prepare_data.context import MeasurementContext
from grassmoth_server.actions.prepare_data.rows import annotate_observation_rows
from grassmoth_server.actions.prepare_data.worker_output import validate_worker_output
from tests.support.helpers import make_model

pytestmark = pytest.mark.contract


def test_one_observation_definition_is_shared_by_preparation_and_model():
    from grassmoth.artifacts.model.indicator import IndicatorSpec
    from grassmoth.artifacts.observations.preparation import (
        ComputedExtractionSpec,
        DataPreparationSpec,
        DataVariableSpec,
    )

    observation = AuthoredObservationSpec(
        id="indicator:dose",
        name="dose",
        measurement_dtype="continuous",
        aggregation="last",
        observation_window="1d",
    )
    recipe = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                observation=observation,
                extraction=ComputedExtractionSpec(
                    source_columns=("dose",),
                    how_to_measure="Read dose",
                    fill_null="forward",
                    fill_null_limit=2,
                ),
            ),
        ),
    )
    assert recipe.observation_schema() == (observation,)
    indicator = IndicatorSpec(observation=observation, construct_polarity="positive")
    assert indicator.observation == observation
    assert set(indicator.model_dump()) == {"observation", "likelihood", "construct_polarity"}
    assert "fill_null" not in recipe.observation_schema()[0].model_dump()


def _measurement(name="Mood"):
    return MeasurementContext.model_validate(
        {
            "source": {"files": ["source.csv"]},
            "model_clock": "1d",
            "indicators": [
                {
                    "observation": {
                        "id": "indicator:mood",
                        "name": name,
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    "extraction": {
                        "kind": "computed",
                        "how_to_measure": "Mean score",
                        "source_columns": ["score"],
                    },
                }
            ],
        }
    )


def test_computed_and_semantic_extraction_preserve_the_same_subject_after_rename():
    raw = pl.DataFrame({"timestamp": [datetime(2026, 1, 1)], "score": [4.0]})
    outputs = []
    for label in ("Mood", "Renamed mood"):
        measurement = _measurement(label)
        computed = compute_indicators(raw, measurement, "timestamp")
        assert computed["indicator_id"].to_list() == ["indicator:mood"]
        worker, errors = validate_worker_output(
            {
                "extractions": [
                    {
                        "indicator_id": "indicator:mood",
                        "window_start": "2026-01-01T00:00:00",
                        "value": 4.0,
                    }
                ]
            },
            measurement,
        )
        assert errors == []
        assert worker is not None
        semantic = worker.to_dataframe()
        assert semantic["indicator_id"].to_list() == ["indicator:mood"]
        outputs.append(
            annotate_observation_rows(
                semantic,
                (
                    ResolvedObservationSpec(
                        id="indicator:mood",
                        name=label,
                        measurement_dtype="continuous",
                        aggregation="mean",
                        observation_window="1d",
                    ),
                ),
            )
        )
    assert outputs[0].equals(outputs[1])


def test_observation_producers_reject_owners_outside_the_pinned_measurement():
    rows = pl.DataFrame(
        {"indicator_id": ["indicator:absent"], "value": [1.0], "timestamp": ["2026-01-01T00:00:00"]}
    )
    with pytest.raises(ValueError, match="unknown indicators"):
        annotate_observation_rows(
            rows,
            (
                ResolvedObservationSpec(
                    id="indicator:mood",
                    name="Mood",
                    measurement_dtype="continuous",
                    aggregation="mean",
                    observation_window="1d",
                ),
            ),
        )
    report = validate_extraction(make_model(["mood"]), [rows])
    assert any(
        finding.kind == "evaluated" and finding.outcome == "failed" for finding in report.findings
    )
    worker, errors = validate_worker_output(
        {
            "extractions": [
                {"indicator_id": "indicator:absent", "window_start": "2026-01-01", "value": 1.0}
            ]
        },
        _measurement(),
    )
    assert worker is None
    assert any("not in indicators" in error for error in errors)
