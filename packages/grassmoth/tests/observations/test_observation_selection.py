"""Data consumers must select both variable identity and measurement meaning."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from grassmoth.artifacts.observations.dataset import (
    MissingObservation,
    ObservationDataset,
    ObservationDefinitionMismatch,
    SelectedObservations,
)
from grassmoth.artifacts.observations.spec import ResolvedObservationSpec
from grassmoth.models.ssm.preparation.binding import PanelPreparationFailure, bind_panel
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import one_state_gaussian_model

pytestmark = pytest.mark.contract
_ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def _dataset(variable):
    anchors = [_ORIGIN + timedelta(days=i) for i in range(2)]
    return ObservationDataset.from_frame(
        pl.DataFrame(
            {
                "indicator_id": [variable.id] * 2,
                "value": [0.0, 1.0] if variable.measurement_dtype == "categorical" else [1.0, 2.0],
                "anchor_time": anchors,
                "support_start": anchors,
                "support_end": anchors,
                "support_kind": [variable.support_kind.value] * 2,
                "summary_operator": [variable.summary_operator.value] * 2,
                "anchor_policy": [variable.anchor_policy.value] * 2,
                "observation_window": [variable.observation_window.source] * 2,
            }
        ),
        (variable,),
        time_origin=_ORIGIN,
    )


def test_selection_accepts_renames_and_equivalent_windows_but_requires_identity():
    observation = compile_model_fixture(one_state_gaussian_model()).observations[0].observation
    dataset = _dataset(observation)
    renamed = observation.revised(name="A new display name", observation_window="24h")
    selected = dataset.select((renamed,))
    assert isinstance(selected, SelectedObservations)
    assert selected.series[0].observation == observation
    assert selected.frame["value"].to_list() == [1, 2]
    assert selected.time_origin == _ORIGIN
    missing = dataset.select((renamed.revised(id="indicator:another"),))
    assert isinstance(missing, MissingObservation)
    assert missing.indicator_id == "indicator:another"
    assert isinstance(dataset.select(()), SelectedObservations)


@pytest.mark.parametrize(
    "changes",
    [{"aggregation": "mean"}, {"observation_window": "2d"}, {"measurement_dtype": "count"}],
)
def test_fit_rejects_changed_definitions_before_projecting_observations(changes, monkeypatch):
    from grassmoth.models.ssm.preparation import binding as runtime

    compiled_dynamical_model = compile_model_fixture(one_state_gaussian_model())
    required = compiled_dynamical_model.observations[0].observation
    # The recorded panel has a valid, independently declared definition.
    recorded = required.revised(**changes)
    if recorded.support_kind == "interval":
        original = _dataset(required).recorded.frame
        frame = original.with_columns(
            (pl.col("support_start") - pl.duration(days=1)).alias("support_start"),
            pl.lit("interval").alias("support_kind"),
            pl.lit(recorded.aggregation.value).alias("summary_operator"),
        )
        dataset = ObservationDataset.from_frame(frame, (recorded,), time_origin=_ORIGIN)
    else:
        dataset = _dataset(recorded)
    mismatch = dataset.select((required,))
    assert isinstance(mismatch, ObservationDefinitionMismatch)
    assert mismatch.expected == required.definition
    assert mismatch.recorded == recorded.definition

    def unexpected(*_args, **_kwargs):
        pytest.fail("Mismatched observations reached numerical preparation")

    monkeypatch.setattr(runtime, "project_observation_data", unexpected)
    result = bind_panel(
        dataset,
        compiled_dynamical_model=compiled_dynamical_model,
        time_origin=_ORIGIN,
        compiler=compile_program,
    )
    assert isinstance(result, PanelPreparationFailure)
    assert result.message == mismatch.message


def test_definition_equality_preserves_ordered_codebooks():
    observation = ResolvedObservationSpec(
        id="indicator:category",
        name="Category",
        measurement_dtype="categorical",
        aggregation="last",
        observation_window="1d",
        categorical_levels=("home", "work"),
    )
    assert (
        observation.definition
        == observation.revised(
            id="indicator:renamed", name="Renamed", observation_window="24h"
        ).definition
    )
    assert (
        observation.definition
        != observation.revised(categorical_levels=("work", "home")).definition
    )

    mismatch = _dataset(observation).select(
        (observation.revised(categorical_levels=("work", "home")),)
    )
    assert isinstance(mismatch, ObservationDefinitionMismatch)
    with pytest.raises(ValueError, match="ordinal_levels requires"):
        observation.revised(ordinal_levels=("low", "high"))
