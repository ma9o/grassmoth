"""One contract covers saved calls, in-flight deduplication and retained failures."""

import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from uuid import UUID

import msgpack
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.preparation import SemanticExtractionSpec
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth_server.actions.contracts.call_state import (
    CallProgress,
    CompletedCall,
    RunningCall,
)
from grassmoth_server.actions.contracts.inputs import (
    DataDiffInput,
    EditModelInput,
    EditQuestionInput,
    FitInput,
    ModelDiffInput,
    NewFitTarget,
    PrepareDataInput,
    SimulateInput,
)
from grassmoth_server.actions.contracts.preparation import FileSourceRef
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
    call_identity,
)
from grassmoth_server.actions.temporal.study import workflow as study_workflow
from grassmoth_server.actions.temporal.study.activities import (
    complete_result_activity,
    edit_question_activity,
    journal_activity,
    read_inputs_activity,
)
from grassmoth_server.actions.temporal.study.messages import ActionRequest, StudyInit
from grassmoth_server.api import app as tool_server
from grassmoth_server.storage import paths as data
from grassmoth_server.study import interface as study_api
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.records import Applied, Rejected
from tests.support.git import git_oid
from tests.support.helpers import fixture_entity_id

if TYPE_CHECKING:
    from collections.abc import Awaitable

pytestmark = pytest.mark.contract


async def _execute_action(
    machine: study_workflow.StudyWorkflow, request: ActionRequest
) -> CallProgress:
    return await cast(
        "Awaitable[CallProgress]", study_workflow.StudyWorkflow.execute_action(machine, request)
    )


@pytest.mark.parametrize(
    "action_request",
    [
        EditQuestionRequest(
            input=EditQuestionInput(
                question=QuestionSpec(
                    text="Does exercise improve sleep?",
                    outcome=fixture_entity_id("construct", "sleep"),
                )
            )
        ),
        EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=git_oid(3), dynamical_model_spec=DynamicalModelSpec()
            )
        ),
        PrepareDataRequest[GitOid, FileSourceRef](
            input=PrepareDataInput[GitOid, FileSourceRef](
                dynamical_model_spec_ref=git_oid(1),
                source=FileSourceRef(files=("data.csv",)),
                extraction={
                    fixture_entity_id("indicator", "outcome"): SemanticExtractionSpec(
                        how_to_measure="Read the outcome"
                    )
                },
            )
        ),
        FitRequest[GitOid](
            input=FitInput[GitOid](
                target=NewFitTarget(
                    dynamical_model_spec_ref=git_oid(1),
                    data_ref=DataRef(revision=git_oid(2), replicate_index=0),
                ),
            )
        ),
        SimulateRequest[GitOid](
            input=SimulateInput[GitOid](
                simulation=SimulationSpec(start=date(2026, 1, 1), horizon="1d"),
                dynamical_model_spec_ref=git_oid(1),
            )
        ),
        DataDiffRequest[GitOid](
            input=DataDiffInput[GitOid](
                left_ref=(DataRef[GitOid, int | None](revision=git_oid(1), replicate_index=0),),
                right_ref=(DataRef[GitOid, int | None](revision=git_oid(2), replicate_index=None),),
            )
        ),
        ModelDiffRequest[GitOid](
            input=ModelDiffInput[GitOid](before_ref=git_oid(1), after_ref=git_oid(2))
        ),
    ],
    ids=lambda action_request: action_request.action,
)
def test_action_reasoning_round_trips_without_changing_call_identity(action_request):
    intent = "Check the revised assumptions.\nGoal: decide which model to fit."
    explained = action_request.revised(reasoning=intent)
    adapter = TypeAdapter(
        ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    )
    assert action_request.reasoning is None
    assert adapter.validate_json(explained.model_dump_json(round_trip=True)) == explained
    assert explained.reasoning == intent
    assert call_identity(action_request) == call_identity(explained)
    assert call_identity(explained.revised(reasoning="A new explanation")) == call_identity(
        action_request
    )


def test_call_contract_reuses_successful_running_and_failed_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 10, 4, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    request = EditQuestionRequest(
        reasoning="Define the outcome to guide model construction.",
        input=EditQuestionInput(
            question=QuestionSpec(
                text="Does exercise improve sleep?", outcome=fixture_entity_id("construct", "sleep")
            )
        ),
    )
    repository = StudyRepository("CALLS")

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        executions = 0

        async def execute(name, payload, **_options):
            nonlocal executions
            name = name if isinstance(name, str) else name.__name__
            if name == "read_inputs_activity":
                return await read_inputs_activity(payload)
            if name == "edit_question_activity":
                executions += 1
                entered.set()
                await release.wait()
                return await edit_question_activity(payload)
            if name == "complete_result_activity":
                return await complete_result_activity(payload)
            if name == "journal_activity":
                return await journal_activity(payload)
            if name == "collect_completed_runs_activity":
                return None
            raise AssertionError(name)

        monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
        machine = study_workflow.StudyWorkflow(StudyInit(workspace_id="CALLS"))
        first = asyncio.create_task(
            _execute_action(machine, ActionRequest(request=request, attempt_id=UUID(int=1)))
        )
        await entered.wait()
        duplicate = await _execute_action(
            machine,
            ActionRequest(
                request=request.revised(reasoning="Polling progress"), attempt_id=UUID(int=2)
            ),
        )
        assert isinstance(duplicate, RunningCall)
        assert duplicate.attempt_id == UUID(int=1)
        assert duplicate.request.reasoning == request.reasoning
        assert executions == 1
        assert repository.attempts() == []
        release.set()
        applied = await first
        assert isinstance(applied, CompletedCall)
        assert isinstance(applied.attempt.outcome, Applied)
        assert (
            await _execute_action(machine, ActionRequest(request=request, attempt_id=UUID(int=3)))
            == applied
        )
        assert executions == 1
        assert len(repository.attempts()) == 1
        restarted = study_workflow.StudyWorkflow(StudyInit(workspace_id="CALLS", initial_seq=1))
        restored = await _execute_action(
            restarted,
            ActionRequest(request=request.revised(reasoning=None), attempt_id=UUID(int=4)),
        )
        assert isinstance(restored, CompletedCall)
        assert restored.commit_id == applied.commit_id
        assert restored.attempt == applied.attempt
        assert executions == 1
        assert len(repository.attempts()) == 1
        failure = EditQuestionRequest(
            reasoning="Reframe the study goal.",
            input=EditQuestionInput(
                question=QuestionSpec(
                    text="A different question", outcome=fixture_entity_id("construct", "sleep")
                )
            ),
        )
        for attempt_id in (5, 6):
            rejected = await _execute_action(
                restarted, ActionRequest(request=failure, attempt_id=UUID(int=attempt_id))
            )
            assert isinstance(rejected, CompletedCall)
            assert isinstance(rejected.attempt.outcome, Rejected)
            assert rejected.attempt.request == failure
        assert len(repository.attempts()) == 2
        assert [entry.record.attempt.outcome.status for entry in repository.attempts()] == [
            "applied",
            "rejected",
        ]

    asyncio.run(scenario())
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    client = TestClient(tool_server.app)
    first = client.post("/api/studies/CALLS/edit_question", json=request.model_dump(mode="json"))
    second = client.post(
        "/api/studies/CALLS/edit_question",
        json=request.revised(reasoning=None).model_dump(mode="json"),
    )
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    from grassmoth_server.public.contracts import ActionResult

    assert ActionResult.model_validate({"payload": first.content}).decode().status == "success"
    result = msgpack.unpackb(first.content)
    assert set(result) == {"call_id", "action", "status", "commit_id", "body", "messages"}
    assert result["status"] == "success"
    assert result["call_id"] == call_identity(request)
    assert result["body"]["question"] == request.input.question.model_dump(mode="json")
    assert set(result["body"]) == {"question"}
    assert (
        client.get(f"/api/studies/CALLS/edit_question/{result['call_id']}").content == first.content
    )
    failure = repository.attempts()[-1].record.attempt.request
    assert failure is not None
    cached_failure = client.post(
        "/api/studies/CALLS/edit_question", json=failure.model_dump(mode="json")
    )
    failed = msgpack.unpackb(cached_failure.content)
    assert failed["status"] == "failed"
    assert failed["body"] is None
    assert failed["commit_id"] is not None
    assert any(message["kind"] == "failure" for message in failed["messages"])
    assert (
        client.get(f"/api/studies/CALLS/edit_question/{failed['call_id']}").content
        == cached_failure.content
    )
    assert client.get(f"/api/studies/CALLS/fit/{result['call_id']}").status_code == 404
    assert client.get(f"/api/studies/MISSING/edit_question/{result['call_id']}").status_code == 404
    timeline = client.get("/api/studies/CALLS/timeline").json()
    assert timeline["attempts"][0]["record"]["attempt"]["request"]["reasoning"] == request.reasoning
    assert len(repository.attempts()) == 2
    assert (
        client.post(
            "/api/studies/CALLS/edit_question",
            json={
                "action": "edit_question",
                "input": {"question": {"text": "Unsaved", "outcome": "construct:sleep"}},
            },
        ).status_code
        == 403
    )


def test_get_reads_running_call_without_starting_workflow_or_work(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    request = EditQuestionRequest(
        input=EditQuestionInput(
            question=QuestionSpec(
                text="A question", outcome=fixture_entity_id("construct", "sleep")
            )
        )
    )
    identity = call_identity(request)
    calls = []

    async def query(name, argument, **_options):
        calls.append((name, argument))
        return (
            RunningCall(attempt_id=UUID(int=1), seq=1, request=request)
            if argument == identity
            else None
        )

    def get_workflow_handle(_identity):
        return SimpleNamespace(query=query)

    async def get(_provider):
        return SimpleNamespace(get_workflow_handle=get_workflow_handle)

    monkeypatch.setattr(study_api.TemporalClientProvider, "get", get)
    with TestClient(tool_server.app) as client:
        for payload in (
            {"action": "edit_question", "input": {"question": {"text": "Missing outcome"}}},
            {
                "action": "edit_question",
                "input": {"question": {"text": "Null outcome", "outcome": None}},
            },
            {"input": {"question": {"text": "Missing action", "outcome": "construct:sleep"}}},
            {
                "action": "edit_question",
                "question": {"text": "Missing input", "outcome": "construct:sleep"},
            },
            {
                "action": "fit",
                "input": {"question": {"text": "Wrong action", "outcome": "construct:sleep"}},
            },
        ):
            assert (
                client.post("/api/studies/RUNNING/edit_question", json=payload).status_code == 422
            )
        response = client.get(f"/api/studies/RUNNING/edit_question/{identity}")
        assert response.status_code == 200
        assert msgpack.unpackb(response.content) == {
            "call_id": identity,
            "action": "edit_question",
            "status": "running",
            "commit_id": None,
            "body": None,
            "messages": [],
        }
        assert client.get(f"/api/studies/RUNNING/fit/{identity}").status_code == 404
        assert client.get(f"/api/studies/RUNNING/edit_question/call:{'0' * 64}").status_code == 404
    assert len(calls) == 3
    assert not (tmp_path / "RUNNING" / "study" / "history.git").exists()
