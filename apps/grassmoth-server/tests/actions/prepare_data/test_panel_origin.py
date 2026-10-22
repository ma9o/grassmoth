"""Server preparation preserves one origin across selected model indicators."""

from datetime import UTC, datetime

import polars as pl
import pytest

from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.models.ssm.preparation.binding import project_observation_data
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import one_state_gaussian_model


@pytest.mark.contract
def test_selected_indicators_keep_the_panel_origin_and_initial_grid_point():
    from grassmoth.models.ssm.preparation.observations import (
        augment_wide_data_with_support_boundaries,
    )
    from grassmoth_server.actions.prepare_data.rows import prepared_time_origin

    early = one_state_gaussian_model()
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
    for model, expected in ((early, 1.0), (late, 11.0)):
        projected = project_observation_data(
            rows, compiled_dynamical_model=compile_model_fixture(model), time_origin=origin
        )
        assert not isinstance(projected, ObservationPreflightFailure)
        (wide, selected) = projected
        assert wide["time"].to_list() == [expected]
        augmented = augment_wide_data_with_support_boundaries(selected, wide, time_origin=origin)
        assert augmented["time"].to_list() == [0.0, expected]
        assert augmented[model.indicators[0].observation.name][0] is None
