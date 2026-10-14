from datetime import UTC, datetime

import polars as pl
import pytest

from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset, SelectedObservations
from nof1_causal_lab.models.ssm import compiled_access as numeric
from tests.support.inference import compile_model_fixture
from tests.support.model import mixed_family_model

pytestmark = pytest.mark.contract


def _single_row_panel(spec, **overrides: float) -> ObservationDataset:
    values = {
        "stress_cont": 0.0,
        "adherence_flag": 1.0,
        "steps_count": 10.0,
        "fatigue_t": 0.0,
        "screen_gap": 1.0,
        "sleep_efficiency": 0.8,
        "symptom_severity": 0.0,
        "coping_style": 0.0,
        "rumination_count": 1.0,
        "focus_cont": 0.0,
    }
    values.update(overrides)
    variables = tuple(item.observation for item in compile_model_fixture(spec).observations)
    return ObservationDataset.from_frame(
        pl.DataFrame(
            [
                {
                    "indicator_id": variable.id,
                    "value": values[variable.name],
                    "anchor_time": "2024-01-02",
                    "support_start": "2024-01-01",
                    "support_end": "2024-01-02",
                    "support_kind": variable.support_kind.value,
                    "summary_operator": variable.summary_operator.value,
                    "anchor_policy": variable.anchor_policy.value,
                    "observation_window": variable.observation_window.source,
                }
                for variable in variables
            ]
        ),
        variables,
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_declared_discrete_levels_allow_one_observed_level():
    dynamical_model_spec = mixed_family_model()

    dataset = _single_row_panel(dynamical_model_spec)
    assert isinstance(dataset.select(dataset.variables), SelectedObservations)

    counts = dict(
        zip(
            numeric.observation_names(compile_model_fixture(dynamical_model_spec)),
            numeric.observation_level_counts(compile_model_fixture(dynamical_model_spec)),
            strict=True,
        )
    )
    assert counts["symptom_severity"] == counts["coping_style"] == 4


def test_declared_discrete_levels_reject_out_of_range_code():
    dynamical_model_spec = mixed_family_model()

    with pytest.raises(ValueError, match="outside its codebook"):
        _single_row_panel(dynamical_model_spec, symptom_severity=4.0)


def test_missing_declared_levels_are_rejected_in_the_scientific_definition():

    indicator = next(
        item
        for item in mixed_family_model().indicators
        if item.observation.measurement_dtype == "ordinal"
    )
    with pytest.raises(ValueError, match="ordinal_levels"):
        indicator.revised(observation=indicator.observation.revised(ordinal_levels=None))
