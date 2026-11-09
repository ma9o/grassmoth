"""Recorded simulation histories are consumed directly, preserving selection and coordinates."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.observations.semantics import SummaryOperator
from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from grassmoth.artifacts.simulation import (
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
    SingleArmSimulation,
)
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.preparation.binding import project_observation_data
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
from grassmoth_server.actions.contracts.preparation import FileSourceRef
from grassmoth_server.actions.contracts.requests import FitRequest, PrepareDataRequest
from grassmoth_server.actions.runners import run_action
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.queries.data import _read_history, read_data_history
from grassmoth_server.study.records import Applied
from grassmoth_server.study.state import StudyState
from grassmoth_server_test_support.action import applied_record, empty_simulation_summary
from grassmoth_server_test_support.data import metadata_for_model, simulation_layout
from grassmoth_server_test_support.helpers import run_async
from grassmoth_server_test_support.storage import result_blob, seed_record
from grassmoth_test_support.git import git_oid
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import construct_named, indicator_named, x_y_model
from grassmoth_test_support.values import revise_fixture


def _recorded_replicate_becomes_a_compatible_panel_complete_test_model() -> DynamicalModelSpec:
    dynamical_model_spec = x_y_model()
    x = construct_named(dynamical_model_spec, "X")
    x_obs = indicator_named(dynamical_model_spec, "X_obs")
    y = construct_named(dynamical_model_spec, "Y")
    y_obs = indicator_named(dynamical_model_spec, "Y_obs")
    x_obs_revised = revise_fixture(
        x_obs, observation=revise_fixture(x_obs.observation, aggregation=SummaryOperator.LAST)
    )
    x_revised = revise_fixture(x, indicators=(x_obs_revised,))
    y_obs_revised = revise_fixture(
        y_obs, observation=revise_fixture(y_obs.observation, aggregation=SummaryOperator.LAST)
    )
    y_revised = revise_fixture(y, indicators=(y_obs_revised,))
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
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.contract


def _model(**observation_changes):
    from grassmoth_test_support.helpers import make_model

    dynamical_model_spec = (
        x_y_model()
        if observation_changes["aggregation"] == "mean"
        else make_model(["X", "Y"], [("X", "Y")])
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                revise_fixture(
                    construct,
                    indicators=tuple(
                        revise_fixture(
                            indicator,
                            observation=revise_fixture(
                                indicator.observation, **observation_changes
                            ),
                        )
                        for indicator in construct.indicators
                    ),
                )
                for construct in dynamical_model_spec.constructs
            ),
        )
    )


def test_fit_reads_selected_history_without_preparing_a_panel(tmp_path, monkeypatch):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("TEST"), StudyRepository("TEST")
    dynamical_model_spec = _recorded_replicate_becomes_a_compatible_panel_complete_test_model()
    model_info = seed_record(
        store,
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    from grassmoth_server_test_support.helpers import write_question

    state = StudyState().with_artifacts([write_question(store), model_info])
    model_commit = history.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=[model_info])),
            seq=1,
            ts="2026-09-25T12:00:00Z",
        )
    ).commit_id
    times = (0.0, 0.5, 2.0)
    draws = np.array([[[91, 92], [93, 94], [95, 96]], [[2, 4], [3, 8], [6, 12]]], dtype=float)
    design = SimulationSpec(start=date(2026, 1, 6), horizon="2d")
    report = SimulationReport(
        summary=empty_simulation_summary(),
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
            result=report.evidence,
            effects=ActionEffects(reports={"simulation": result_blob(store, report)}),
        ),
        seq=2,
        ts="2026-09-25T12:01:00Z",
    )
    source_commit = history.append(simulation_record).commit_id
    # A later simulation must not replace the explicitly selected source.
    later = revise_fixture(
        report,
        evidence=revise_fixture(
            report.evidence,
            arms=revise_fixture(
                report.evidence.arms,
                action=revise_fixture(
                    report.evidence.arms.action,
                    observations=NumericalArray.from_numpy(np.zeros_like(draws)),
                ),
            ),
        ),
    )
    history.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=later.evidence,
                effects=ActionEffects(reports={"simulation": result_blob(store, later)}),
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
        read_data_history(store, revise_fixture(source, revision=model_commit))
    selected = read_data_history(store, source)
    assert selected.source == source
    assert selected.metadata is None
    assert set(store.repo.references) == refs_before
    panel = selected.observations
    from grassmoth.artifacts.observations.dataset import SelectedObservations

    selection = panel.select(panel.variables)
    assert isinstance(selection, SelectedObservations)
    projected = project_observation_data(
        selection,
        compiled_dynamical_model=compile_model_fixture(dynamical_model_spec),
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    assert projected["time"][0] == 0
    wide = projected.filter(pl.col("time") > 0)
    np.testing.assert_allclose(wide["time"].to_numpy(), np.asarray(times) + 5)
    np.testing.assert_allclose(
        wide.select(
            numeric.observation_names(compile_model_fixture(dynamical_model_spec))
        ).to_numpy(),
        draws[1],
    )
    assert all(series.history.times.values[0] == 5 for series in panel.series.values())
    assert all(series.observation.support_kind == "point" for series in panel.series.values())

    from grassmoth_server.actions.fit import action as fit_module
    from grassmoth_test_support.inference import _report

    def fit(inputs, **_kwargs):
        assert inputs.data_for_model == panel
        assert inputs.data_for_model.time_origin == selected.time_origin
        from grassmoth.actions.checks import fit_check_report
        from grassmoth.actions.contracts.io import FitOutput

        return FitOutput(
            dynamical_model_spec=dynamical_model_spec,
            checks=fit_check_report(inputs.selection, inputs.question, inputs.data_for_model),
            inference=_report(dynamical_model_spec),
        )

    monkeypatch.setattr(fit_module, "run_fit", fit)
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=model_info.revision,
                data_ref=DataRef(revision=source.revision, replicate_index=source.replicate_index),
            ),
        )
    )
    applied = run_async(run_action("TEST", request, revise_fixture(state, data=source)))
    assert isinstance(request.input.target, NewFitTarget)
    assert request.input.target.data_ref == source
    assert {info.artifact_id for info in applied.effects.produced} == {"model"}
    published = history.append(
        applied_record("TEST", applied, request=request, seq=history.latest_seq() + 1)
    )
    tree = store.repo[published.commit_id].tree
    assert "records/model" in tree
    assert "records/panel" not in tree
    for index in (2, 20):
        with pytest.raises(StudyLookupError, match="replicate"):
            run_async(
                run_action(
                    "TEST",
                    revise_fixture(
                        request,
                        input=revise_fixture(
                            request.input,
                            target=revise_fixture(
                                request.input.target,
                                data_ref=revise_fixture(source, replicate_index=index),
                            ),
                        ),
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
                "measurement": {"dtype": "categorical", "levels": ("a", "b", "c")},
            }
        )
    )
    values = np.repeat(np.array([[[np.nan, np.nan], [2, 2], [2, 2]]]), 4, axis=0)
    if not interval:
        values[:, 0] = 2
    report = SimulationReport(
        summary=empty_simulation_summary(),
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            times=(0, 1, 2.5),
            draws=4,
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
                    np.array([[np.nan, np.nan], [1 + 5e-9, 1 + 5e-9], [2.5, 2.5]])
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
    from grassmoth_server_test_support.data import simulation_output

    output = simulation_output(report)
    panel = output.observations(0)
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth_server.actions.fit.remote import pack_observations
    from grassmoth_server.study.persistence.codec import unpack_result

    packed = pack_observations(panel)
    restored = ObservationDataset.model_validate(unpack_result(packed))
    assert report.evidence.arms.action.observations.npy not in packed
    assert report.evidence.observation_layout.mask.npy not in packed
    assert restored.time_origin == panel.time_origin
    assert {
        identity: series.history.readings() for identity, series in restored.series.items()
    } == {identity: series.history.readings() for identity, series in panel.series.items()}
    if interval:
        from grassmoth.models.ssm.preparation.binding import BoundPanel, bind_panel
        from grassmoth_test_support.compute import compile_program

        bound = bind_panel(
            restored,
            compiled_dynamical_model=compile_model_fixture(dynamical_model_spec),
            compiler=compile_program,
        )
        assert isinstance(bound, BoundPanel)
        np.testing.assert_allclose(bound.times, [0, 1, 1.5, 2.5])
        assert np.all(np.diff(bound.times) > 1e-8)
    readings = tuple(
        reading for series in panel.series.values() for reading in series.history.readings()
    )
    assert [reading[1] for reading in readings if reading[1] is not None] == [2.0] * (
        4 if interval else 6
    )
    assert all(series.history.times.values[0] == 0 for series in panel.series.values())
    if interval:
        assert [
            reading[3] - reading[2] for reading in readings if reading[1] is not None
        ] == pytest.approx([1.0] * 4)
        assert sum(reading[2] is None for reading in readings) == 2
    else:
        assert all(reading[2] == reading[0] for reading in readings)
    from grassmoth_server.study.errors import StudyLookupError

    with pytest.raises(StudyLookupError, match="replicate"):
        _read_history(DataRef[GitOid, int](revision=git_oid(1), replicate_index=4), output)
    values[0, 1, 0] = np.nan
    report = revise_fixture(
        report,
        evidence=revise_fixture(
            report.evidence,
            arms=SingleArmSimulation(
                action=revise_fixture(
                    report.evidence.arms.action, observations=NumericalArray.from_numpy(values)
                )
            ),
        ),
    )
    with pytest.raises(ValueError, match="Observed values must be finite"):
        simulation_output(report).observations(0)


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
