"""Revision selectors pin published inputs before HTTP deduplication and dispatch."""

import hashlib
from datetime import UTC, date, datetime
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts.bodies import NewFitTarget, FitInput, SimulateInput
from nof1_causal_lab.actions.contracts.call_state import PendingCall
from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import (
    DataDiffRequest,
    FitRequest,
    ModelDiffRequest,
    ScientificActionRequest,
    SimulateRequest,
    call_identity,
)
from nof1_causal_lab.actions.input_resolution import input_state
from nof1_causal_lab.api import app as tool_server
from nof1_causal_lab.api import studies as study_api
from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.observations.reference import DataRef
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
from nof1_causal_lab.storage import paths as data
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    FitAttempt,
    Raised,
    SimulateAttempt,
)
from nof1_causal_lab.study.state import RetractedArtifact
from tests.support.action import applied_record, empty_simulation_summary, question_root
from tests.support.git import git_oid

pytestmark = pytest.mark.contract


def _publish(repository, *artifacts, retracted=()):
    return repository.append(
        applied_record(
            repository.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=artifacts, retracted=retracted)),
            seq=repository.latest_seq() + 1,
        )
    )


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    source = tmp_path / "LATEST/input/data.csv"
    source.parent.mkdir(parents=True)
    source.write_text("date,value\n2026-01-01,1\n")
    question_root("LATEST")
    repository, store = StudyRepository("LATEST"), ArtifactStore("LATEST")
    models = tuple(
        store.write_artifact(
            "model",
            produced_by="edit_model",
            derived_from={"question": repository.question().revision},
            json_files={
                "model.json": DynamicalModelSpec(measurement_clock=clock).model_dump(mode="json")
            },
        )
        for clock in ("1d", "2d")
    )
    for model in models:
        _publish(repository, model)
    raw = store.write_artifact(
        "raw_data",
        produced_by="prepare_data",
        derived_from={},
        json_files={"data.json": {"version": 1}},
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={"raw_data": raw.revision},
        json_files={"data.json": {}},
    )
    _publish(repository, raw, panel)
    evidence = SimulationEvidence(
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 1),
        draws=2,
        seed=0,
        state_ids=(),
        parameter_draws={},
        arms=SingleArmSimulation(
            action=SimulationArm(
                latent_paths=NumericalArray.from_numpy(np.zeros((2, 2, 0))),
                observations=NumericalArray.from_numpy(np.zeros((2, 2, 0))),
            ),
        ),
        observation_layout=SimulationObservationLayout(
            variables=(),
            support_start_times=NumericalArray.from_numpy(np.zeros((2, 0))),
            support_end_times=NumericalArray.from_numpy(np.zeros((2, 0))),
            mask=NumericalArray.from_numpy(np.zeros((2, 2, 0), dtype=bool)),
        ),
        assignments=SimulationSpec(start=date(2026, 1, 1), horizon="1d").assignments(
            datetime(2026, 1, 1, tzinfo=UTC)
        ),
    )
    for seed in (0, 1):
        simulation = repository.append(
            applied_record(
                repository.workspace_id,
                Applied(
                    result=ModelSimulationResult(evidence=evidence.revised(seed=seed)),
                    effects=ActionEffects(
                        reports={
                            "simulation": store.write_report(
                                SimulationReport(
                                    evidence=evidence.revised(seed=seed),
                                    summary=empty_simulation_summary(),
                                    law=AuthoredLawProvenance(),
                                    fit_reliability="not_fitted",
                                )
                            )
                        }
                    ),
                ),
                request=SimulateRequest(
                    input=SimulateInput(
                        dynamical_model_spec_ref=models[-1].revision,
                        simulation=SimulationSpec(start=date(2026, 1, 1), horizon="1d"),
                    )
                ),
                seq=repository.latest_seq() + 1,
            )
        )
    repository.append(
        AttemptRecord(
            seq=repository.latest_seq() + 1,
            ts="2026-01-01T00:00:00Z",
            attempt=SimulateAttempt(
                action="simulate",
                request=None,
                outcome=Raised(error_type="RuntimeError", error_message="failed"),
            ),
        )
    )
    # A staged artifact has a ref in the store but was never successfully published.
    store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={},
        json_files={"model.json": {"measurement_clock": "3d"}},
    )
    return repository, store, models, panel, simulation.commit_id


@pytest.fixture
def client(monkeypatch):
    running, dispatched = {}, []

    async def query(_name, identity, **_options):
        return running.get(identity)

    async def start_update(_name, action, **_options):
        dispatched.append(action.request)
        running[call_identity(action.request)] = PendingCall(
            attempt_id=action.attempt_id, request=action.request
        )

    async def handle(*_args):
        return SimpleNamespace(query=query, start_update=start_update)

    async def get(_provider):
        return SimpleNamespace(get_workflow_handle=lambda _identity: SimpleNamespace(query=query))

    monkeypatch.setattr(study_api, "_study_handle", handle)
    monkeypatch.setattr(study_api.TemporalClientProvider, "get", get)
    with TestClient(tool_server.app) as http:
        yield http, dispatched


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        (
            "edit_model",
            {
                "parent_ref": "latest",
                "dynamical_model_spec": {},
            },
        ),
        (
            "fit",
            {
                "dynamical_model_spec_ref": "latest",
                "data_ref": {"revision": "latest", "replicate_index": 0},
            },
        ),
        (
            "simulate",
            {
                "dynamical_model_spec_ref": "latest",
                "simulation": {"start": "2026-01-01", "horizon": "2d"},
            },
        ),
        (
            "prepare_data",
            {
                "dynamical_model_spec_ref": "latest",
                "source": "input",
                "extraction": {
                    "indicator:00000000000000000000": {
                        "kind": "semantic",
                        "how_to_measure": "Read the outcome",
                    }
                },
            },
        ),
        ("model_diff", {"before_ref": str(git_oid(1)), "after_ref": "latest"}),
        (
            "data_diff",
            {
                "left_ref": [{"revision": "latest", "replicate_index": 0}],
                "right_ref": [{"revision": str(git_oid(2)), "replicate_index": None}],
            },
        ),
    ],
)
def test_latest_is_pinned_by_input_type_before_running_call_deduplication(
    study, client, action, arguments
):
    _repository, _, models, panel, _simulation = study
    http, dispatched = client
    path = f"/api/studies/LATEST/{action}"
    response = http.post(
        path,
        json={
            "action": action,
            "input": arguments,
            "reasoning": "Choose the newest published inputs.",
        },
    )
    assert response.status_code == 200, response.text
    result = msgpack.unpackb(response.content)
    assert set(result) == {"call_id", "action", "status", "commit_id", "body", "messages"}
    pinned = dispatched[0].model_dump(mode="json")
    assert pinned["reasoning"] == "Choose the newest published inputs."
    inputs = pinned["input"]
    if action in {"fit", "simulate", "edit_model", "prepare_data"}:
        assert (
            inputs["parent_ref" if action == "edit_model" else "dynamical_model_spec_ref"]
            == models[-1].revision
        )
        if action == "fit":
            assert inputs["data_ref"] == {"revision": panel.revision, "replicate_index": 0}
    elif action == "model_diff":
        assert inputs["before_ref"] == git_oid(1)
        assert inputs["after_ref"] == models[-1].revision
    else:
        assert inputs["left_ref"] == [{"revision": panel.revision, "replicate_index": 0}]
        assert inputs["right_ref"] == [{"revision": git_oid(2), "replicate_index": None}]
    assert len(dispatched) == 1
    if action == "prepare_data":
        inputs["source"] = arguments["source"]
    assert msgpack.unpackb(http.post(path, json=pinned).content)["call_id"] == result["call_id"]
    assert (
        msgpack.unpackb(http.post(path, json={"action": action, "input": arguments}).content)[
            "call_id"
        ]
        == result["call_id"]
    )
    assert len(dispatched) == 1


def _prepare_folder(http, source):
    return http.post(
        "/api/studies/LATEST/prepare_data",
        json={
            "action": "prepare_data",
            "input": {
                "dynamical_model_spec_ref": "latest",
                "source": source,
                "extraction": {
                    "indicator:score": {"kind": "semantic", "how_to_measure": "Read the score"}
                },
            },
        },
    )


def test_prepare_data_captures_named_folder_recursively_and_pins_bytes(study, client, tmp_path):
    http, dispatched = client
    workspace = tmp_path / "LATEST"
    source = workspace / "exports"
    (source / "nested").mkdir(parents=True)
    expected = {
        "exports/a.csv": b"date,score\n2026-01-01,1\n",
        "exports/nested/b.csv": b"date,score\n2026-01-02,2\n",
    }
    for name, body in reversed(expected.items()):
        (workspace / name).write_bytes(body)

    response = _prepare_folder(http, "exports")
    assert response.status_code == 200, response.text
    original_id = msgpack.unpackb(response.content)["call_id"]
    captured = dispatched[0].input.source
    assert captured.files == tuple(expected)
    assert captured.hashes == {
        name: hashlib.sha256(body).hexdigest() for name, body in expected.items()
    }
    assert msgpack.unpackb(_prepare_folder(http, "exports").content)["call_id"] == original_id
    assert len(dispatched) == 1

    (source / "a.csv").write_bytes(b"date,score\n2026-01-01,3\n")
    changed_id = msgpack.unpackb(_prepare_folder(http, "exports").content)["call_id"]
    assert changed_id != original_id
    (source / "c.csv").write_bytes(b"date,score\n2026-01-03,4\n")
    added_id = msgpack.unpackb(_prepare_folder(http, "exports").content)["call_id"]
    assert added_id not in {original_id, changed_id}
    (source / "c.csv").unlink()
    assert msgpack.unpackb(_prepare_folder(http, "exports").content)["call_id"] == changed_id
    assert len(dispatched) == 3

    for name in expected:
        (workspace / name).unlink()
    (source / "nested").rmdir()
    source.rmdir()
    poll = http.get(f"/api/studies/LATEST/prepare_data/{original_id}")
    assert poll.status_code == 200
    assert msgpack.unpackb(poll.content)["call_id"] == original_id
    for name, body in expected.items():
        staged = workspace / "scratch" / "source-files" / captured.hashes[name] / name
        assert staged.read_bytes() == body


@pytest.mark.parametrize(
    "source",
    ["", ".", "..", "../input", "/input", "input/nested", "input\\nested", {"files": ["data.csv"]}],
)
def test_prepare_data_requires_one_folder_name(client, source):
    http, dispatched = client
    assert _prepare_folder(http, source).status_code == 422
    assert not dispatched


@pytest.mark.parametrize("kind", ["missing", "empty", "file", "outside_workspace"])
def test_prepare_data_rejects_unavailable_source_without_dispatch(study, client, tmp_path, kind):
    http, dispatched = client
    folder = tmp_path / "LATEST" / "exports"
    if kind == "empty":
        folder.mkdir()
    elif kind == "file":
        folder.write_bytes(b"not a folder")
    elif kind == "outside_workspace":
        folder.mkdir()
        outside = tmp_path / "private.csv"
        outside.write_bytes(b"private")
        (folder / "linked.csv").symlink_to(outside)
    response = _prepare_folder(http, "exports")
    assert response.status_code == 422, response.text
    assert not dispatched


def test_latest_reuses_explicit_saved_call_and_pinned_poll_survives_newer_model(
    study, client, monkeypatch
):
    repository, _, models, panel, _ = study
    http, dispatched = client
    request = FitRequest[GitOid](
        input=FitInput[GitOid](target=NewFitTarget(
            dynamical_model_spec_ref=models[-1].revision,
            data_ref=DataRef(revision=panel.revision, replicate_index=0)),)
    )
    saved = repository.append(
        AttemptRecord(
            seq=repository.latest_seq() + 1,
            ts="2026-01-01T00:00:00Z",
            attempt=FitAttempt(
                action="fit",
                request=request,
                outcome=Raised(error_type="TestError", error_message="Retained failed fit"),
            ),
        )
    )
    monkeypatch.setattr(
        study_api,
        "_saved_response",
        lambda _workspace, revision: Response(content=revision.commit_id),
    )
    path = "/api/studies/LATEST/fit"
    arguments = {
        "action": "fit",
        "input": {
            "dynamical_model_spec_ref": "latest",
            "data_ref": {"revision": "latest", "replicate_index": 0},
        },
    }
    assert http.post(path, json=arguments).text == saved.commit_id
    assert not dispatched
    _publish(repository, models[0])
    assert http.get(f"{path}/{call_identity(request)}").text == saved.commit_id
    assert http.post(path, json=request.model_dump(mode="json")).text == saved.commit_id
    response = http.post(path, json=arguments)
    assert response.status_code == 200
    assert dispatched[0].input.dynamical_model_spec_ref == models[0].revision
    assert len(dispatched) == 1


@pytest.mark.parametrize(
    ("action", "kind", "arguments"),
    [
        ("edit_model", "question", {"parent_ref": "latest", "dynamical_model_spec": {}}),
        (
            "fit",
            "model",
            {
                "dynamical_model_spec_ref": "latest",
                "data_ref": {"revision": str(git_oid(1)), "replicate_index": 0},
            },
        ),
        (
            "fit",
            "panel",
            {
                "dynamical_model_spec_ref": str(git_oid(1)),
                "data_ref": {"revision": "latest", "replicate_index": 0},
            },
        ),
    ],
)
def test_missing_latest_is_rejected_without_dispatch(
    tmp_path, monkeypatch, client, action, kind, arguments
):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    http, dispatched = client
    response = http.post(
        f"/api/studies/EMPTY/{action}", json={"action": action, "input": arguments}
    )
    assert response.status_code == 422
    assert kind in response.json()["detail"]
    assert not dispatched
    assert StudyRepository("EMPTY").attempts() == []


@pytest.mark.parametrize("invalidity", ["retracted", "stale"])
def test_latest_does_not_resurrect_retracted_or_stale_panel(study, client, invalidity):
    repository, store, models, panel, _ = study
    if invalidity == "retracted":
        _publish(
            repository,
            retracted=(RetractedArtifact(artifact_id="panel", reason_ref="empty extraction"),),
        )
    else:
        raw = store.write_artifact(
            "raw_data",
            produced_by="prepare_data",
            derived_from={},
            json_files={"data.json": {"version": 2}},
        )
        _publish(repository, raw)
    http, dispatched = client
    path = "/api/studies/LATEST/fit"
    assert (
        http.post(
            path,
            json={
                "action": "fit",
                "input": {
                    "dynamical_model_spec_ref": "latest",
                    "data_ref": {"revision": "latest", "replicate_index": 0},
                },
            },
        ).status_code
        == 422
    )
    assert not dispatched
    # An explicit historical input still has exactly its existing meaning.
    assert (
        http.post(
            path,
            json={
                "action": "fit",
                "input": {
                    "dynamical_model_spec_ref": models[-1].revision,
                    "data_ref": {"revision": panel.revision, "replicate_index": 0},
                },
            },
        ).status_code
        == 200
    )


@pytest.mark.parametrize("parent_kind", ["question", "model"])
def test_edit_model_parent_pins_question_for_identity_checks_and_provenance(
    study, client, parent_kind
):
    from nof1_causal_lab.actions.edit_model import edit_model
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_lab.study.records import record_dependencies
    from tests.support.helpers import make_model, write_question

    repository, store, models, _, _ = study
    http, dispatched = client
    path = "/api/studies/LATEST/edit_model"
    dynamical_model_spec = make_model(["X"])
    original_question = repository.question()
    parent = original_question if parent_kind == "question" else models[-1]
    parent_call = next(
        revision
        for revision in repository.attempts()
        if isinstance(revision.record.attempt.outcome, Applied)
        and parent in revision.record.attempt.outcome.effects.produced
    )
    arguments = {
        "action": "edit_model",
        "input": {
            "parent_ref": parent.revision,
            "dynamical_model_spec": dynamical_model_spec.model_dump(mode="json"),
        },
    }
    response = http.post(path, json=arguments)
    assert response.status_code == 200, response.text
    original = dispatched[0]
    question = write_question(
        store,
        QuestionSpec(text="A different question", outcome=dynamical_model_spec.edges[0].effect.id),
    )
    _publish(repository, question)
    assert (
        msgpack.unpackb(http.post(path, json=original.model_dump(mode="json")).content)["call_id"]
        == msgpack.unpackb(response.content)["call_id"]
    )
    changed = http.post(
        path,
        json={
            **arguments,
            "input": {**arguments["input"], "parent_ref": question.revision},
        },
    )
    assert changed.status_code == 200, changed.text
    assert (
        msgpack.unpackb(changed.content)["call_id"] != msgpack.unpackb(response.content)["call_id"]
    )
    selected = dispatched[1]
    assert selected.input.parent_ref == question.revision
    assert input_state(repository, selected).current == {"question": question}
    state = input_state(repository, original)
    assert state.current["question"] == original_question
    assert state.get("model") == (parent if parent_kind == "model" else None)
    outcome = edit_model(store.workspace_id, original)
    assert isinstance(outcome, Applied)
    assert outcome.effects.produced[0].derived_from == {
        "question": original_question.revision,
        **({"model": parent.revision} if parent_kind == "model" else {}),
    }
    checks, _ = evaluate_model_checks(store.workspace_id, state, outcome, action="edit_model")
    assert checks.question is not None
    assert all(item.code not in {"window", "range"} for item in checks.question.findings)
    saved = repository.append(
        applied_record(
            repository.workspace_id, outcome, request=original, seq=repository.latest_seq() + 1
        )
    )
    dependency = next(
        item for item in record_dependencies(repository.attempts()) if item.seq == saved.record.seq
    )
    assert (dependency.argument, dependency.source_seq) == (
        "parent",
        parent_call.record.seq,
    )


@pytest.mark.parametrize("parent", [{}, {"parent_ref": None}, {"parent_ref": "invalid"}])
def test_edit_model_requires_a_parent_selector(client, parent):
    http, dispatched = client
    response = http.post(
        "/api/studies/LATEST/edit_model",
        json={"action": "edit_model", "input": {**parent, "dynamical_model_spec": {}}},
    )
    assert response.status_code == 422
    assert all("parent_ref" in item["loc"] for item in response.json()["detail"])
    assert not dispatched


@pytest.mark.parametrize("selection", ["initial", "retracted"])
def test_latest_model_parent_selects_question_when_no_current_model(study, client, selection):
    repository, _, _, _, _ = study
    if selection == "initial":
        question_root("INITIAL")
        repository = StudyRepository("INITIAL")
    else:
        _publish(
            repository,
            retracted=(RetractedArtifact(artifact_id="model", reason_ref="removed"),),
        )
    http, dispatched = client
    response = http.post(
        f"/api/studies/{repository.workspace_id}/edit_model",
        json={
            "action": "edit_model",
            "input": {"parent_ref": "latest", "dynamical_model_spec": {}},
        },
    )
    assert response.status_code == 200, response.text
    assert dispatched[0].input.parent_ref == repository.question().revision


def test_edit_model_parent_rejects_an_unrelated_artifact(study):
    from nof1_causal_lab.actions.contracts.bodies import EditModelInput
    from nof1_causal_lab.actions.contracts.requests import EditModelRequest
    from nof1_causal_lab.study.errors import StudyLookupError

    repository, _, _, panel, _ = study
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=panel.revision, dynamical_model_spec=DynamicalModelSpec()
        )
    )
    with pytest.raises(StudyLookupError, match="question or model parent"):
        input_state(repository, request)


def test_only_http_inputs_accept_latest():
    adapter = TypeAdapter(
        ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "action": "fit",
                "input": {
                    "dynamical_model_spec_ref": "latest",
                    "data_ref": {"revision": git_oid(1), "replicate_index": 0},
                },
            }
        )
