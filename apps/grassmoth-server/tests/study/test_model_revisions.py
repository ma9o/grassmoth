"""Whole scientific values commit atomically and findings retain exact source revisions."""

import json

import pytest

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.models.inputs import input_fingerprints
from grassmoth_server.actions.contracts.requests import EditModelRequest
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.state import ArtifactRecord, StudyState, is_stale
from tests.support.action import edit_and_check, question_root
from tests.support.git import artifact_revision, artifact_revisions, git_oid
from tests.support.helpers import fixture_entity_id, make_model
from tests.support.model import x_y_model

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "additive-revisions"


def test_full_model_write_keeps_the_named_base_without_a_head_gate(workspace):
    question = QuestionSpec(text="Does X change Y?", outcome=fixture_entity_id("construct", "Y"))
    root = StudyRepository(workspace).state(question_root(workspace, question).commit_id)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["question"].revision,
                    "dynamical_model_spec": make_model(["X", "Y"], [("X", "Y")]).model_dump(
                        mode="json"
                    ),
                }
            }
        ),
        root,
    )
    state = root.with_artifacts(effects.effects.produced)
    assert state.current["model"].revision == effects.effects.produced[0].revision
    assert set(state.current) == {"question", "model"}
    from grassmoth_server.study.persistence.store import ArtifactStore

    assert artifact_revisions(ArtifactStore(workspace), "model") == [
        state.current["model"].revision
    ]
    second = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": artifact_revision(workspace, "model", 1),
                    "dynamical_model_spec": make_model(["X", "Y"], [("X", "Y")]).model_dump(
                        mode="json"
                    ),
                }
            }
        ),
        state,
    )
    from tests.support.action import model_findings

    checks, _ = model_findings(
        workspace, state.with_artifacts(second.effects.produced), action="edit_model"
    )
    assert checks.question is not None
    assert all(finding.code not in {"window", "range"} for finding in checks.question.findings)
    assert second.effects.produced[0].revision != effects.effects.produced[0].revision
    assert second.effects.produced[0].derived_from == {
        "question": root.current["question"].revision,
        "model": state.current["model"].revision,
    }


def test_model_input_identity_preserves_findings_and_original_pins():
    model = ArtifactRecord(artifact_id="model", revision=git_oid(2))
    panel = ArtifactRecord(artifact_id="panel", revision=git_oid(1))
    state = StudyState().with_artifacts([model, panel])
    assert not is_stale(state, "panel")
    changed = model.revised(revision=git_oid(3))
    assert not is_stale(state.with_artifacts([changed]), "panel")


def test_statistical_enrichment_preserves_structural_and_measurement_inputs():
    from tests.support.helpers import make_model

    measured = make_model(["X", "Y"], [("X", "Y")])
    specified = x_y_model()
    before, after = input_fingerprints(measured), input_fingerprints(specified)
    reloaded = DynamicalModelSpec.model_validate_json(
        json.dumps(specified.model_dump(mode="json"), sort_keys=True)
    ).materialized()
    assert input_fingerprints(reloaded) == after
    for purpose in ("observations", "identification"):
        assert before[purpose] == after[purpose]
    assert before["compilation"] != after["compilation"]

    changed = input_fingerprints(
        measured.with_entities(
            edges=(measured.edges[0].revised(description="Revised causal assumption"),)
        )
    )
    assert changed["observations"] == before["observations"]
    for purpose in ("identification", "compilation"):
        assert changed[purpose] != before[purpose]

    reasoned = input_fingerprints(
        specified.with_entities(
            parameters=tuple(
                parameter.revised(reasoning="A literature range for this quantity.")
                for parameter in specified.parameters
            )
        )
    )
    for purpose in ("observations", "identification", "compilation"):
        assert reasoned[purpose] == after[purpose]
    assert reasoned["belief"] != after["belief"]


def test_model_diff_addresses_question_and_model_without_checkpoint_traversal(workspace):
    from grassmoth_server.actions.model_diff import model_diff
    from grassmoth_server.study.errors import StudyLookupError
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.records import AttemptRecord, EditAttempt, Raised
    from tests.support.action import applied_record

    question = QuestionSpec(text="Does X change Y?", outcome=fixture_entity_id("construct", "Y"))
    root = question_root(workspace, question)
    repository = StudyRepository(workspace)
    state = repository.state(root.commit_id)
    definition = make_model(["X", "Y"], [("X", "Y")])
    request = EditModelRequest[GitOid].model_validate(
        {
            "input": {
                "parent_ref": state.current["question"].revision,
                "dynamical_model_spec": definition.model_dump(mode="json"),
            }
        }
    )
    applied = edit_and_check(workspace, request, state)
    edited = repository.append(applied_record(workspace, applied, request=request, seq=2))
    model = applied.effects.produced[0].revision
    empty = DynamicalModelSpec.from_entities()
    assert model_diff(
        workspace, state.current["question"].revision, model
    ).changes == definition.changes_from(empty)
    assert model_diff(
        workspace, model, state.current["question"].revision
    ).changes == empty.changes_from(definition)
    failed = repository.append(
        AttemptRecord(
            seq=3,
            ts="2026-10-07T12:00:00Z",
            attempt=EditAttempt(
                action="edit_model",
                request=request.revised(input=request.input.revised(parent_ref=model)),
                outcome=Raised(error_type="RuntimeError", error_message="Failed"),
            ),
        ),
    )
    assert failed.parent_ids == (edited.commit_id,)
    panel = ArtifactStore(workspace).write_artifact(
        "panel", produced_by="prepare_data", derived_from={}, json_files={"metadata.json": {}}
    )
    for invalid in (failed.commit_id, panel.revision):
        with pytest.raises(StudyLookupError):
            model_diff(workspace, model, invalid)
