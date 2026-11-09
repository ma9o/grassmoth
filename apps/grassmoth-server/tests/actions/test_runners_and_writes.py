"""Authoring commits, optional extraction outputs, and atomic derivation cascades."""

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.observations.preparation import FailedExtractionChunk
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.models.causal.analysis import identify_model
from grassmoth.models.structure import StructuralSelection, selected_state_ids
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.requests import EditModelRequest
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied
from grassmoth_server.study.state import StudyState, apply_effects
from grassmoth_server_test_support.action import applied_record, edit_and_check, model_root
from grassmoth_server_test_support.data import metadata_for_model
from grassmoth_server_test_support.git import artifact_revision, artifact_revisions
from grassmoth_server_test_support.storage import seed_record, seed_source_table
from grassmoth_test_support.helpers import make_model
from grassmoth_test_support.purpose import initial_model
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import ArtifactId


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def _model():
    return make_model(["Stress", "Perf"], [("Stress", "Perf")])


def _outcome():
    return _model().constructs[1].id


def _rooted(workspace):
    """The study state after edit_model, asking about Perf."""
    root = model_root(
        workspace,
        initial_model(question="does stress hurt performance?", outcome=_outcome()),
    )
    return StudyRepository(workspace).state(root.commit_id)


def _exact_measurement(model):
    from grassmoth.artifacts.model.expressions import state
    from grassmoth.artifacts.model.likelihood import LikelihoodSpec

    owner = model.constructs[0]
    indicator = revise_fixture(
        owner.indicators[0],
        reasoning={"text": "An exact observation of the construct"},
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(owner.id)),
        ),
    )
    return model.with_entities(
        edges=replace_constructs(
            model.edges,
            (
                revise_fixture(owner, indicators=(indicator,)),
                *model.constructs[1:],
            ),
        )
    )


def _write(store, artifact_id, payload, pins=None):

    return seed_record(
        store,
        artifact_id,
        derived_from=pins or {},
        produced_by=None,
        json_files={{"model": "model.json", "question": "model.json"}[artifact_id]: payload},
    )


def test_identification_records_positive_and_absent_queries():
    report = identify_model(StructuralSelection(_model(), _outcome()))
    assert report.estimable_treatments == (_model().constructs[0].id,)
    assert report.outcome == _outcome()
    unmeasured = initial_model(question="Does a new outcome change?")
    absent = StructuralSelection.for_model(unmeasured).identification
    assert absent.outcome == unmeasured.outcome
    assert absent.treatments == {}


def test_identification_preserves_negative_findings():
    from grassmoth.artifacts.model.construct import replace_constructs

    dynamical_model_spec = _model()
    stress, perf = (construct.id for construct in dynamical_model_spec.constructs)
    unmeasured = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                revise_fixture(construct, indicators=()) if construct.id == perf else construct
                for construct in dynamical_model_spec.constructs
            ),
        )
    )
    report = identify_model(StructuralSelection(unmeasured, perf))
    assert not report.estimable_treatments
    assert report.non_identifiable[stress].confounders == (perf,)
    assert "notes" not in report.non_identifiable[stress].model_dump(mode="json")


@pytest.mark.parametrize("nonempty", [False, True])
def test_extraction_requires_some_observations(workspace, tmp_path, nonempty):
    from grassmoth_server.actions.temporal.extraction.activities import (
        finalize_measurements_activity,
    )
    from grassmoth_server.actions.temporal.extraction.messages import MeasurementsFinalizeInput

    store = ArtifactStore(workspace)
    dynamical_model_spec = _model()
    raw = seed_source_table(store, pl.DataFrame({"timestamp": [datetime(2026, 1, 1)]}), "run-1")
    state = StudyState().with_artifacts(
        [raw, _write(store, "model", dynamical_model_spec.model_dump(mode="json"))]
    )
    pins: dict[ArtifactId, GitOid] = {"raw_data": raw.revision}
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "preparation": {
                    "source": {"files": ["observations.csv"]},
                    "definition": metadata_for_model(dynamical_model_spec).preparation.model_dump(
                        mode="json"
                    ),
                },
                "empty_output": {
                    "extractions": [
                        {
                            "indicator_id": dynamical_model_spec.indicators[0].observation.id,
                            "value": "3.0",
                            "window_start": "2026-01-01",
                        }
                    ]
                    if nonempty
                    else []
                },
                "chunks": [{"worker_id": 7, "n_windows": 1, "spec_ref": "unused.json"}],
            }
        )
    )
    pending = finalize_measurements_activity(
        MeasurementsFinalizeInput(
            workspace_id=workspace,
            run_id="run-1",
            plan_ref=str(path),
            pins=pins,
            chunk_results=[
                FailedExtractionChunk(
                    worker_id=7,
                    status="failed",
                    n_extractions=0,
                    n_windows=1,
                )
            ],
        )
    )
    if not nonempty:
        from temporalio.exceptions import ApplicationError

        with pytest.raises(ApplicationError, match="Extraction produced no observations"):
            asyncio.run(pending)
        return
    effects = asyncio.run(pending)

    assert ("panel" in {info.artifact_id for info in effects.effects.produced}) == nonempty
    assert "measurements" not in {info.artifact_id for info in effects.effects.produced}
    from grassmoth.actions.contracts.io import PrepareDataOutput

    prepared_result = store.read_result(effects.result, PrepareDataOutput)
    assert len(prepared_result.extraction.workers) == 1
    worker = prepared_result.extraction.workers[0]
    assert (worker.worker_id, worker.status, worker.n_extractions, worker.n_windows) == (
        7,
        "failed",
        0,
        1,
    )
    assert worker.status == "failed"
    assert "error" not in worker.model_dump(mode="json")
    from grassmoth_server.study.persistence.history import StudyRepository

    journal = StudyRepository(workspace)
    record = applied_record(workspace, effects, seq=1, ts="2026-01-01T00:00:00Z")
    journal.append(record)
    outcome = journal.attempts()[0].record.attempt.outcome
    assert outcome.status == "applied"
    assert outcome.effects.produced == effects.effects.produced
    assert journal.read_file(journal.head(), "result.msgpack")
    current = apply_effects(state, effects.effects.produced, effects.effects.retracted)
    if not nonempty:
        assert not (current.get("panel") is not None)
    if nonempty:
        panel = next(info for info in effects.effects.produced if info.artifact_id == "panel")
        assert panel.derived_from == {
            "raw_data": artifact_revision(workspace, "raw_data", 1),
        }


def test_model_write_cascades_without_parallel_scientific_catalogs(workspace):
    root = _rooted(workspace)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["model"].revision,
                    "dynamical_model_spec": _model().model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    assert {info.artifact_id for info in effects.effects.produced} == {
        "model",
    }
    assert not effects.effects.retracted
    assert all(
        info.derived_from
        == {
            "model": artifact_revision(workspace, "model", 1),
        }
        for info in effects.effects.produced
        if info.artifact_id != "model"
    )
    assert next(
        info for info in effects.effects.produced if info.artifact_id == "model"
    ).derived_from == {
        "model": root.current["model"].revision,
    }


def test_exact_measurement_preserves_execution_layout(workspace):
    dynamical_model_spec = _exact_measurement(_model())
    root = _rooted(workspace)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["model"].revision,
                    "dynamical_model_spec": dynamical_model_spec.model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    store = ArtifactStore(workspace)
    info = next(info for info in effects.effects.produced if info.artifact_id == "model")
    payload = store.read_document("model", info.revision)
    plan = DynamicalModelSpec.model_validate(payload).materialized()
    assert [
        plan.get_construct(identity).name
        for identity in selected_state_ids(StructuralSelection(plan, None))
    ] == ["Perf", "Stress"]
    exact = next(
        item
        for item in plan.indicators
        if item.observation.id == dynamical_model_spec.indicators[0].observation.id
    )
    assert exact.likelihood is not None
    assert exact.likelihood.law.family == "delta"


def test_model_edit_is_independent_of_stale_extraction(workspace):
    store = ArtifactStore(workspace)
    dynamical_model_spec = _model()
    root = _rooted(workspace)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["model"].revision,
                    "dynamical_model_spec": dynamical_model_spec.model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    state = apply_effects(root, effects.effects.produced)
    from grassmoth_server_test_support.runner_fixtures import panel_frame

    variables = metadata_for_model(dynamical_model_spec).variables
    frame = panel_frame(n_days=3).with_columns(
        pl.col("indicator_id").replace_strict(
            {
                source: variable.id
                for source, variable in zip(
                    panel_frame(n_days=3)["indicator_id"].unique(maintain_order=True),
                    variables,
                    strict=True,
                )
            }
        )
    )
    panel = seed_record(
        store,
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={
            "metadata.json": metadata_for_model(dynamical_model_spec).model_dump(
                mode="json", round_trip=True
            )
        },
        parquet_files={"panel.parquet": frame},
    )
    from grassmoth.artifacts.observations.preparation import DataPreparationResult
    from grassmoth_server_test_support.action import prepared_profile

    prepared_profile(
        workspace,
        state,
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel])),
    )
    state = revise_fixture(
        state.with_artifacts([panel]),
        data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
    )
    changed = revise_fixture(dynamical_model_spec, question="Revised question")
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": artifact_revision(workspace, "model", 1),
                    "dynamical_model_spec": changed.model_dump(mode="json"),
                }
            }
        ),
        state,
    )
    from grassmoth_server_test_support.action import model_findings

    checks, _ = model_findings(
        workspace, state.with_artifacts(effects.effects.produced), action="edit_model"
    )
    assert all(finding.code not in {"window", "range"} for finding in checks.question.findings)
    assert "panel" not in {item.artifact_id for item in effects.effects.produced}


def test_failed_check_publishes_no_state(workspace, monkeypatch):
    from grassmoth.models.causal import analysis as identification

    def fail(*_args, **_kwargs):
        raise RuntimeError("identification failed")

    root = _rooted(workspace)
    monkeypatch.setattr(identification, "check_identifiability", fail)
    with pytest.raises(RuntimeError, match="identification failed"):
        edit_and_check(
            workspace,
            EditModelRequest[GitOid].model_validate(
                {
                    "input": {
                        "parent_ref": root.current["model"].revision,
                        "dynamical_model_spec": _model().model_dump(mode="json"),
                    }
                }
            ),
            root,
        )
    assert StudyRepository(workspace).state() == root


def test_invalid_model_rejected_before_any_write(workspace):
    with pytest.raises(ValidationError):
        edit_and_check(
            workspace,
            EditModelRequest[GitOid].model_validate(
                {
                    "input": {
                        "parent_ref": GitOid("0" * 40),
                        "dynamical_model_spec": {"constructs": [{"id": "construct:invalid"}]},
                    }
                }
            ),
            StudyState(),
        )
    assert artifact_revisions(ArtifactStore(workspace), "model") == []


def test_failed_tree_write_publishes_no_artifact(workspace, monkeypatch):
    from grassmoth_server.study.persistence import store as store_module

    def fail_tree(*_args, **_kwargs):
        raise OSError("cannot write metadata")

    monkeypatch.setattr(store_module, "write_blob", fail_tree)
    with pytest.raises(OSError, match="metadata"):
        _write(ArtifactStore(workspace), "model", _model().model_dump(mode="json"))
    assert artifact_revisions(ArtifactStore(workspace), "model") == []


def test_question_write_requires_text():
    with pytest.raises(ValidationError):
        EditModelRequest.model_validate({"question": {"text": "   "}})
