"""Authoring commits, optional extraction outputs, and atomic derivation cascades."""

import asyncio
import json
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import EditModelRequest, EditQuestionRequest
from nof1_causal_lab.artifacts.identity import ConstructId, GitOid
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.likelihood import DeltaLawSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.preparation import FailedExtractionChunk, FileSourceRef
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.models.causal.analysis import identify_model
from nof1_causal_lab.models.structure import StructuralSelection, selected_state_ids
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import StudyState, apply_effects
from tests.support.action import applied_record, edit_and_check, question_root
from tests.support.data import metadata_for_model
from tests.support.git import artifact_revision, artifact_revisions
from tests.support.helpers import make_model

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def _model():
    return make_model(["Stress", "Perf"], [("Stress", "Perf")])


def _outcome():
    return _model().constructs[1].id


def _rooted(workspace):
    """The study state after edit_question, asking about Perf."""
    root = question_root(
        workspace, QuestionSpec(text="does stress hurt performance?", outcome=_outcome())
    )
    return StudyRepository(workspace).state(root.commit_id)


def _exact_measurement(model):
    from nof1_causal_lab.artifacts.model.expressions import state
    from nof1_causal_lab.artifacts.model.likelihood import LikelihoodSpec

    owner = model.constructs[0]
    indicator = owner.indicators[0].revised(
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(owner.id)),
            reasoning="An exact observation of the construct",
        )
    )
    return model.with_entities(
        edges=replace_constructs(
            model.edges,
            (
                owner.revised(indicators=(indicator,)),
                *model.constructs[1:],
            ),
        )
    )


def _write(store, artifact_id, payload, pins=None):
    from nof1_causal_lab.study.artifact_files import artifact_file_spec

    return store.write_artifact(
        artifact_id,
        derived_from=pins or {},
        produced_by=None,
        json_files={next(iter(artifact_file_spec(artifact_id).json_files.values())): payload},
    )


def test_identification_records_positive_and_absent_queries():
    report = identify_model(StructuralSelection(_model(), _outcome()))
    assert report.estimable_treatments == (_model().constructs[0].id,)
    assert report.outcome == _outcome()
    question = QuestionSpec(
        text="Does stress affect performance?", outcome=ConstructId("construct:not_defined_yet")
    )
    absent = identify_model(StructuralSelection.for_question(_model(), question))
    assert absent.outcome is None
    assert absent.estimable_treatments == ()


def test_identification_preserves_negative_findings(monkeypatch):
    from dataclasses import replace

    from nof1_causal_lab.models.causal import analysis as identification
    from nof1_causal_lab.models.causal.identification import UnidentifiedQuery

    dynamical_model_spec = _model()
    result = identification.check_identifiability(
        dynamical_model_spec.constructs,
        dynamical_model_spec.edges,
        outcome_id=_outcome(),
        observed_constructs={construct.name for construct in dynamical_model_spec.constructs},
    )

    monkeypatch.setattr(
        identification,
        "check_identifiability",
        lambda *_, **__: replace(
            result,
            identifiable_treatments={},
            non_identifiable_treatments={
                "Stress": UnidentifiedQuery(confounders=(), notes="Unidentified")
            },
        ),
    )
    report = identify_model(StructuralSelection(_model(), _outcome()))
    assert not report.estimable_treatments
    assert report.non_identifiable[_model().constructs[0].id].notes == "Unidentified"


@pytest.mark.parametrize("nonempty", [False, True])
def test_extraction_requires_some_observations(workspace, tmp_path, nonempty):
    from nof1_causal_lab.actions.temporal.extraction.activities import (
        finalize_measurements_activity,
    )
    from nof1_causal_lab.actions.temporal.extraction.messages import MeasurementsFinalizeInput

    store = ArtifactStore(workspace)
    dynamical_model_spec = _model()
    raw = store.write_artifact("raw_data", derived_from={}, produced_by="prepare_data")
    state = StudyState().with_artifacts(
        [raw, _write(store, "model", dynamical_model_spec.model_dump(mode="json"))]
    )
    pins: dict[ArtifactId, GitOid] = {"raw_data": raw.revision}
    from nof1_causal_lab.actions.prepare_data.context import MeasurementContext

    metadata = metadata_for_model(dynamical_model_spec)
    context = MeasurementContext(
        source=FileSourceRef(files=("input/test.csv",), hashes={}),
        model_clock=dynamical_model_spec.measurement_clock,
        indicators=metadata.preparation.variables,
    )
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "measurement_structure": context.model_dump(mode="json"),
                "preparation": {
                    "source": {"files": ["observations.csv"]},
                    "definition": metadata_for_model(dynamical_model_spec).preparation.model_dump(
                        mode="json"
                    ),
                },
                "recorded_dicts": [],
                "computed_dicts": [
                    {
                        "indicator_id": dynamical_model_spec.indicators[0].observation.id,
                        "value": "3.0",
                        "timestamp": "2026-01-01",
                    }
                ]
                if nonempty
                else [],
                "empty_output": {"extractions": []},
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
                    error="No usable extraction",
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
    assert len(effects.result.workers) == 1
    worker = effects.result.workers[0]
    assert (worker.worker_id, worker.status, worker.n_extractions, worker.n_windows) == (
        7,
        "failed",
        0,
        1,
    )
    assert worker.status == "failed"
    assert worker.error == "No usable extraction"
    from nof1_causal_lab.study.persistence.history import StudyRepository

    journal = StudyRepository(workspace)
    record = applied_record(workspace, effects, seq=1, ts="2026-01-01T00:00:00Z", trace_ids=[])
    journal.append(record)
    outcome = journal.attempts()[0].record.attempt.outcome
    assert outcome.status == "applied"
    assert outcome.effects.produced == effects.effects.produced
    assert journal.read_file(journal.head(), "result.msgpack")
    current = apply_effects(state, effects.effects.produced, effects.effects.retracted)
    if not nonempty:
        assert not current.has("panel")
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
                    "parent_ref": root.current["question"].revision,
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
            "question": root.current["question"].revision,
            "model": artifact_revision(workspace, "model", 1),
        }
        for info in effects.effects.produced
        if info.artifact_id != "model"
    )
    assert next(
        info for info in effects.effects.produced if info.artifact_id == "model"
    ).derived_from == {"question": root.current["question"].revision}


def test_exact_measurement_preserves_execution_layout(workspace):
    dynamical_model_spec = _exact_measurement(_model())
    root = _rooted(workspace)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["question"].revision,
                    "dynamical_model_spec": dynamical_model_spec.model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    store = ArtifactStore(workspace)
    info = next(info for info in effects.effects.produced if info.artifact_id == "model")
    payload = store.read_json_file("model", info.revision, "model.json")
    plan = DynamicalModelSpec.model_validate(payload).materialized()
    assert [
        plan.get_construct(identity).name
        for identity in selected_state_ids(StructuralSelection(plan, None))
    ] == ["Stress", "Perf"]
    assert plan.indicators[0].likelihood is not None
    assert plan.indicators[0].likelihood.law.family == "delta"


def test_model_edit_is_independent_of_stale_extraction(workspace):
    store = ArtifactStore(workspace)
    dynamical_model_spec = _model()
    root = _rooted(workspace)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["question"].revision,
                    "dynamical_model_spec": dynamical_model_spec.model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    state = apply_effects(root, effects.effects.produced)
    from tests.integration.runner_fixtures import panel_frame

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
    panel = store.write_artifact(
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
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks
    from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult

    evaluate_data_checks(
        workspace,
        state,
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel])),
    )
    state = state.with_artifacts([panel]).revised(
        data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0)
    )
    changed = dynamical_model_spec.revised(measurement_clock="2d")
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
    from nof1_causal_lab.actions.model_checks import read_model_checks

    checks, _ = read_model_checks(
        workspace, state.with_artifacts(effects.effects.produced), action="edit_model"
    )
    assert all(finding.code not in {"window", "range"} for finding in checks.question.findings)
    assert "panel" not in {item.artifact_id for item in effects.effects.produced}


def test_failed_check_publishes_no_state(workspace, monkeypatch):
    from nof1_causal_lab.models.causal import analysis as identification

    def fail(*_args, **_kwargs):
        raise RuntimeError("identification failed")

    monkeypatch.setattr(identification, "check_identifiability", fail)
    root = _rooted(workspace)
    with pytest.raises(RuntimeError, match="identification failed"):
        edit_and_check(
            workspace,
            EditModelRequest[GitOid].model_validate(
                {
                    "input": {
                        "parent_ref": root.current["question"].revision,
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
    from nof1_causal_lab.study.persistence import store as store_module

    def fail_tree(*_args, **_kwargs):
        raise OSError("cannot write metadata")

    monkeypatch.setattr(store_module, "write_tree", fail_tree)
    with pytest.raises(OSError, match="metadata"):
        _write(ArtifactStore(workspace), "model", _model().model_dump(mode="json"))
    assert artifact_revisions(ArtifactStore(workspace), "model") == []


def test_question_write_requires_text():
    with pytest.raises(ValidationError):
        EditQuestionRequest.model_validate({"question": {"text": "   "}})
