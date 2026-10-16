"""Read-only facade replays saved calls and rejects new actions and uploads."""

import msgpack
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.contracts.bodies import EditModelInput
from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import EditModelRequest
from nof1_causal_lab.api.read_only import create_read_facade_app
from nof1_causal_lab.artifacts.identity import ConstructId, GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.records import Applied
from tests.support.action import applied_record, question_root
from tests.support.git import git_oid

pytestmark = pytest.mark.contract


def test_read_facade_serves_reads_and_rejects_actions(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    client = TestClient(create_read_facade_app())

    assert client.get("/api/workspaces").headers["X-Actions-Enabled"] == "false"
    status = client.get("/api/studies/WS-READONLY/timeline")
    assert status.status_code == 200
    assert status.json() == {"attempts": [], "dependencies": [], "running": None}

    action = client.post(
        "/api/studies/WS-READONLY/fit",
        json={
            "action": "fit",
            "input": {
                "target": {
                    "kind": "new",
                    "dynamical_model_spec_ref": str(git_oid(1)),
                    "data_ref": {"revision": str(git_oid(2)), "replicate_index": 0},
                }
            },
        },
    )
    assert action.status_code == 403
    upload = client.post(
        "/api/upload",
        data={"workspaceId": "WS-READONLY"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )
    assert upload.status_code == 403


def test_saved_call_serves_pinned_artifacts_without_starting_new_actions(monkeypatch, tmp_path):
    from nof1_causal_lab.study.persistence.history import StudyRepository
    from nof1_causal_lab.study.persistence.store import ArtifactStore

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    question_root("WS-ART")
    store, repository = ArtifactStore("WS-ART"), StudyRepository("WS-ART")
    client = TestClient(create_read_facade_app())
    saved = []
    for clock in ("1d", "2d"):
        dynamical_model_spec = DynamicalModelSpec(measurement_clock=clock)
        request = EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=repository.question().revision, dynamical_model_spec=dynamical_model_spec
            )
        )
        artifact = store.write_artifact(
            "model",
            derived_from={},
            produced_by="edit_model",
            json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
        )
        assert (
            client.post(
                "/api/studies/WS-ART/edit_model", json=request.model_dump(mode="json")
            ).status_code
            == 403
        )
        revision = repository.append(
            applied_record(
                "WS-ART",
                Applied(result=None, effects=ActionEffects(produced=(artifact,))),
                seq=repository.latest_seq() + 1,
                request=request,
            )
        )
        saved.append((request, revision, clock))
    for request, revision, clock in saved:
        response = client.post(
            "/api/studies/WS-ART/edit_model", json=request.model_dump(mode="json")
        )
        assert response.status_code == 200, response.text
        assert msgpack.unpackb(response.content)["commit_id"] == revision.commit_id
        assert (
            msgpack.unpackb(response.content)["body"]["dynamical_model_spec"]["measurement_clock"]
            == clock
        )
    assert len(repository.attempts()) == 3


def test_saved_call_returns_promoted_traces(monkeypatch, tmp_path):
    from nof1_causal_lab.llm.traces import LLMTrace, TraceMessage
    from nof1_causal_lab.storage import backend as storage
    from nof1_causal_lab.storage.traces import collect_run_traces
    from nof1_causal_lab.study.persistence.history import StudyRepository
    from nof1_causal_lab.study.persistence.store import ArtifactStore

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    question_root("WS-TRACE")
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=StudyRepository("WS-TRACE").question().revision,
            dynamical_model_spec=DynamicalModelSpec(),
        )
    )
    model = ArtifactStore("WS-TRACE").write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": request.input.dynamical_model_spec.model_dump(mode="json")},
    )
    source = str(tmp_path / "data/WS-TRACE/scratch/runs/seq-000002/llm/raw-data/trace.json")
    storage.write_text(
        source,
        LLMTrace(
            messages=[TraceMessage(role="assistant", content="profiled")],
            model="test-model",
        ).model_dump_json(),
    )
    logs = collect_run_traces("WS-TRACE", 2)
    StudyRepository("WS-TRACE").append(
        applied_record(
            "WS-TRACE",
            Applied(result=None, effects=ActionEffects(produced=(model,))),
            seq=2,
            request=request,
            ts="2026-07-09T00:00:00+00:00",
            trace_ids=["raw-data"],
        ),
        logs=logs,
    )
    client = TestClient(create_read_facade_app())

    timeline = client.get("/api/studies/WS-TRACE/timeline")
    assert timeline.json()["attempts"][-1]["record"]["trace_ids"] == ["raw-data"]
    result = client.post("/api/studies/WS-TRACE/edit_model", json=request.model_dump(mode="json"))
    assert result.status_code == 200, result.text
    assert (
        next(
            message
            for message in msgpack.unpackb(result.content)["messages"]
            if message["kind"] == "trace"
        )["trace"]["messages"][0]["content"]
        == "profiled"
    )


def test_workspaces_endpoint_lists_study_questions(monkeypatch, tmp_path):

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")

    question_root(
        "WS-LIST", QuestionSpec(text="does X cause Y?", outcome=ConstructId("construct:y"))
    )

    client = TestClient(create_read_facade_app())
    response = client.get("/api/workspaces")

    assert response.status_code == 200
    assert response.json() == {"WS-LIST": "does X cause Y?"}


def test_upload_endpoint_stages_input_file(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import backend as storage

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    client = TestClient(create_read_facade_app())

    response = client.post(
        "/api/upload",
        data={"workspaceId": "WS-UPLOAD"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == "WS-UPLOAD/input/data.csv"
    assert storage.read_text(str(tmp_path / "data" / "WS-UPLOAD" / "input" / "data.csv")) == (
        "x,y\n1,2\n"
    )


def test_full_facade_advertises_actions(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    from nof1_causal_lab.api import app as tool_server

    client = TestClient(tool_server.app)
    assert client.get("/api/workspaces").headers["X-Actions-Enabled"] == "true"
