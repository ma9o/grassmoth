"""Preparation preserves one origin across selected model indicators."""

from datetime import UTC, datetime

import polars as pl
import pytest

from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.models.ssm.preparation.binding import (
    PanelPreparationFailure,
    project_observation_data,
    select_model_observations,
)
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import one_state_gaussian_model
from grassmoth_test_support.values import revise_fixture


@pytest.mark.contract
def test_selected_indicators_preserve_the_model_grid_across_sparse_readings():
    from grassmoth.actions.prepare_data.rows import prepared_time_origin
    from grassmoth.artifacts.observations.dataset import SelectedObservations
    from grassmoth_test_support.helpers import observation_dataset

    early = revise_fixture(
        one_state_gaussian_model(),
        time={"window": {"start": "2024-01-01", "horizon": "12d"}, "step": "1d"},
    )
    late = DynamicalModelSpec.model_validate_json(
        early.model_dump_json()
        .replace(str(early.indicators[0].observation.id), "indicator:late")
        .replace(early.indicators[0].observation.name, "late_obs")
    ).materialized()
    rows = pl.DataFrame(
        {
            "indicator_id": [
                early.indicators[0].observation.id,
                late.indicators[0].observation.id,
            ],
            "value": [1.0, 2.0],
            "anchor_time": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
            "support_start": [datetime(2024, 1, 1), datetime(2024, 1, 11)],
            "support_end": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
            "support_kind": ["point", "point"],
        }
    )
    origin = prepared_time_origin(rows, None)
    assert origin == datetime(2024, 1, 1, tzinfo=UTC)
    variables = tuple(
        compile_model_fixture(model).observations[0].observation for model in (early, late)
    )
    dataset = observation_dataset(rows, variables, time_origin=origin)
    for model, expected in ((early, 1.0), (late, 11.0)):
        compiled = compile_model_fixture(model)
        selected = dataset.select(tuple(item.observation for item in compiled.observations))
        assert isinstance(selected, SelectedObservations)
        selected = select_model_observations(selected, model.time)
        assert not isinstance(selected, PanelPreparationFailure)
        wide = project_observation_data(selected, compiled_dynamical_model=compiled)
        assert not isinstance(wide, ObservationPreflightFailure)
        assert wide["time"].to_list() == list(range(13))
        assert wide[model.indicators[0].observation.name][int(expected)] is not None
        assert wide[model.indicators[0].observation.name][0] is None
