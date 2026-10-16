"""Recorded simulation histories are consumed directly, preserving selection and coordinates."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.bodies import NewFitTarget, FitInput
from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import FitRequest, PrepareDataRequest
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.observations.preparation import FileSourceRef
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.observations.semantics import SummaryOperator
from nof1_causal_lab.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.simulation import (
    ModelSimulationResult,
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
    SingleArmSimulation,
)
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.preparation.panel import project_observation_data
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.queries.data import read_data_history, read_simulation_observations
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import StudyState
from tests.support.action import applied_record, empty_simulation_summary
from tests.support.data import metadata_for_model, simulation_layout
from tests.support.git import git_oid
from tests.support.helpers import make_model, run_async
from tests.support.inference import compile_model_fixture
from tests.support.model import construct_named, indicator_named, x_y_model


def _recorded_replicate_becomes_a_compatible_panel_complete_test_model() -> DynamicalModelSpec:
    dynamical_model_spec = x_y_model()
    x = construct_named(dynamical_model_spec, "X")
    x_obs = indicator_named(dynamical_model_spec, "X_obs")
    y = construct_named(dynamical_model_spec, "Y")
    y_obs = indicator_named(dynamical_model_spec, "Y_obs")
    x_obs_revised = x_obs.revised(
        observation=x_obs.observation.revised(aggregation=SummaryOperator.LAST)
    )
    x_revised = x.revised(indicators=(x_obs_revised,))
    y_obs_revised = y_obs.revised(
        observation=y_obs.observation.revised(aggregation=SummaryOperator.LAST)
    )
    y_revised = y.revised(indicators=(y_obs_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                x_revised,
                y_revised,
            ),
        )
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.contract


def _model(**observation_changes):
    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(
                    indicators=tuple(
                        indicator.revised(
                            observation=indicator.observation.revised(**observation_changes)
                        )
                        for indicator in construct.indicators
                    )
                )
                for construct in dynamical_model_spec.constructs
            ),
        )
    )


def test_fit_reads_selected_history_without_preparing_a_panel(tmp_path, monkeypatch):
    from nof1_causal_lab.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("TEST"), StudyRepository("TEST")
    dynamical_model_spec = _recorded_replicate_becomes_a_compatible_panel_complete_test_model()
    model_info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    from tests.support.helpers import write_question

    state = StudyState().with_artifacts([write_question(store), model_info])
    model_commit = history.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=[model_info])),
            seq=1,
            ts="2026-09-25T12:00:00Z",
            trace_ids=[],
        )
    ).commit_id
    times = (0.0, 0.5, 2.0)
    draws = np.array([[[91, 92], [93, 94], [95, 96]], [[2, 4], [3, 8], [6, 12]]], dtype=float)
    design = SimulationSpec(start=date(2026, 1, 6), horizon="2d")
    report = SimulationReport(
        summary=empty_simulation_summary(),
        fit_reliability="not_fitted",
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            times=tuple(time + 5 for time in times),
            draws=2,
            seed=0,
            state_ids=tuple(numeric.state_ids(compile_model_fixture(dynamical_model_spec))),
            parameter_draws={"known_truth": NumericalArray.from_numpy(np.asarray([100.0, 100.0]))},
            arms=SingleArmSimulation(
                action=SimulationArm(
                    latent_paths=NumericalArray.from_numpy(
                        np.zeros(
                            (
                                2,
                                3,
                                len(numeric.state_ids(compile_model_fixture(dynamical_model_spec))),
                            )
                        )
                    ),
                    observations=NumericalArray.from_numpy(draws),
                ),
            ),
            observation_layout=simulation_layout(
                dynamical_model_spec,
                tuple(t + 5 for t in times),
                np.ones_like(draws, dtype=bool),
                NumericalArray.from_numpy,
            ),
            assignments=design.assignments(datetime(2026, 1, 1, tzinfo=UTC)),
        ),
    )
    simulation_record = applied_record(
        store.workspace_id,
        Applied(
            result=ModelSimulationResult(evidence=report.evidence),
            effects=ActionEffects(reports={"simulation": store.write_report(report)}),
        ),
        seq=2,
        ts="2026-09-25T12:01:00Z",
        trace_ids=[],
    )
    source_commit = history.append(simulation_record).commit_id
    # A later simulation must not replace the explicitly selected source.
    later = report.revised(
        evidence=report.evidence.revised(
            arms=report.evidence.arms.revised(
                action=report.evidence.arms.action.revised(
                    observations=NumericalArray.from_numpy(np.zeros_like(draws))
                )
            )
        )
    )
    history.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=ModelSimulationResult(evidence=later.evidence),
                effects=ActionEffects(reports={"simulation": store.write_report(later)}),
            ),
            seq=3,
        )
    )
    source = DataRef[GitOid, int](revision=source_commit, replicate_index=1)
    refs_before = set(store.repo.references)

    def observations_only(_identity):
        raise AssertionError("Saved numerical evidence must be self-contained")

    monkeypatch.setattr(store, "read_array", observations_only)
    with pytest.raises(StudyLookupError, match="applied prepare_data or simulate"):
        read_data_history(store, source.revised(revision=model_commit))
    selected = read_data_history(store, source)
    assert selected.source == source
    assert selected.metadata is None
    assert set(store.repo.references) == refs_before
    panel = selected.observations.recorded.frame
    projected = project_observation_data(
        panel,
        compiled_dynamical_model=compile_model_fixture(dynamical_model_spec),
        time_origin=selected.time_origin,
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    (wide, _) = projected
    np.testing.assert_allclose(wide["time"].to_numpy(), np.asarray(times) + 5)
    np.testing.assert_allclose(
        wide.select(
            numeric.observation_names(compile_model_fixture(dynamical_model_spec))
        ).to_numpy(),
        draws[1],
    )
    assert panel["anchor_time"].min() == datetime(2026, 1, 6)
    assert set(panel["support_kind"]) == {"point"}

    from nof1_causal_lab.actions.fit import action as fit_module
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from tests.support.inference import _report

    def fit(**kwargs):
        assert kwargs["data_for_model"].recorded.frame.equals(panel)
        assert kwargs["time_origin"] == selected.time_origin
        return {
            "_dynamical_model_spec": dynamical_model_spec,
            "evidence": InferenceEvidence(),
            "metadata": _report(dynamical_model_spec).core.inference_metadata,
        }

    monkeypatch.setattr(fit_module, "fit", fit)
    monkeypatch.setattr(
        fit_module, "read_inference_report", lambda *_: _report(dynamical_model_spec)
    )
    request = FitRequest[GitOid](
        input=FitInput[GitOid](target=NewFitTarget(
            dynamical_model_spec_ref=model_info.revision,
            data_ref=DataRef(revision=source.revision, replicate_index=source.replicate_index)),)
    )
    applied = run_async(run_action("TEST", request, state.revised(data=source)))
    assert request.input.data_ref == source
    assert {info.artifact_id for info in applied.effects.produced} == {"model"}
    assert not any(ref.startswith("refs/artifacts/panel/") for ref in store.repo.references)
    for index in (2, 20):
        with pytest.raises(StudyLookupError, match="replicate"):
            run_async(
                run_action(
                    "TEST",
                    request.revised(
                        input=request.input.revised(data_ref=source.revised(replicate_index=index))
                    ),
                    state,
                )
            )


@pytest.mark.parametrize("interval", [False, True])
def test_reading_preserves_measurement_support_and_numeric_codes(interval):
    dynamical_model_spec = _model(
        **(
            {"aggregation": "mean", "observation_window": "1d"}
            if interval
            else {
                "aggregation": "last",
                "measurement_dtype": "categorical",
                "categorical_levels": ("a", "b", "c"),
            }
        )
    )
    values = np.array([[[np.nan, np.nan], [2, 2], [2, 2]]])
    if not interval:
        values[0, 0] = 2
    report = SimulationReport(
        summary=empty_simulation_summary(),
        fit_reliability="not_fitted",
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            times=(0, 1, 2.5),
            draws=1,
            seed=0,
            state_ids=tuple(item.id for item in dynamical_model_spec.constructs),
            parameter_draws={},
            arms=SingleArmSimulation(
                action=SimulationArm(
                    latent_paths=NumericalArray.from_numpy(np.zeros_like(values)),
                    observations=NumericalArray.from_numpy(values),
                ),
            ),
            observation_layout=SimulationObservationLayout(
                variables=metadata_for_model(dynamical_model_spec).variables,
                support_start_times=NumericalArray.from_numpy(
                    np.array([[np.nan, np.nan], [0, 0], [1.5, 1.5]])
                    if interval
                    else np.array([[0, 0], [1, 1], [2.5, 2.5]])
                ),
                support_end_times=NumericalArray.from_numpy(
                    np.array([[np.nan, np.nan], [1, 1], [2.5, 2.5]])
                    if interval
                    else np.array([[0, 0], [1, 1], [2.5, 2.5]])
                ),
                mask=NumericalArray.from_numpy(np.isfinite(values)),
            ),
            assignments=SimulationSpec(start=date(2026, 1, 1), horizon="60h").assignments(
                datetime(2026, 1, 1, tzinfo=UTC)
            ),
        ),
    )
    panel = read_simulation_observations(report.evidence, 0)
    assert panel["value"].drop_nulls().to_list() == [2.0] * (4 if interval else 6)
    assert panel["anchor_time"].min() == datetime(2026, 1, 1)
    if interval:
        rows = panel.filter(pl.col("value").is_not_null())
        assert (rows["support_end"] - rows["support_start"]).to_list() == [timedelta(days=1)] * 4
        assert panel.filter(pl.col("value").is_null())["support_start"].null_count() == 2
    else:
        assert panel["support_start"].to_list() == panel["anchor_time"].to_list()
    from nof1_causal_lab.study.errors import StudyLookupError

    with pytest.raises(StudyLookupError, match="replicate"):
        read_simulation_observations(
            report.evidence,
            1,
        )
    values[0, 1, 0] = np.nan
    report = report.revised(
        evidence=report.evidence.revised(
            arms=SingleArmSimulation(
                action=report.evidence.arms.action.revised(
                    observations=NumericalArray.from_numpy(values)
                )
            )
        )
    )
    with pytest.raises(ValueError, match="non-finite emissions"):
        read_simulation_observations(
            report.evidence,
            0,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"input": {"source": {"files": ["observations.csv"]}}},
        {"input": {"revision": git_oid(1), "replicate": -1}},
        {
            "input": {"revision": git_oid(1), "replicate": 0},
            "dynamical_model_spec_ref": git_oid(2),
        },
        {"input": {"revision": git_oid(1), "replicate": 0, "max_windows": 1}},
        {"input": "panel"},
    ],
)
def test_data_sources_require_one_unambiguous_origin(payload):
    with pytest.raises(ValidationError):
        PrepareDataRequest[GitOid, FileSourceRef].model_validate(payload)
