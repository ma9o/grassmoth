"""Journal-owned durable LLM traces."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth_server.actions.call_logs import (
    collect_call_log,
    collect_call_snapshot,
    emit_message,
    read_call_log,
)
from grassmoth_server.actions.contracts.call_state import RunningCall
from grassmoth_server.actions.contracts.inputs import EditModelInput
from grassmoth_server.actions.contracts.progress import StepEvent
from grassmoth_server.actions.contracts.requests import EditModelRequest, call_identity
from grassmoth_server.actions.progress import emit_event
from grassmoth_server.actions.temporal.study.activities import (
    journal_activity,
    read_inputs_activity,
)
from grassmoth_server.actions.temporal.study.messages import AttemptPublication, ReadInputsInput
from grassmoth_server.llm.traces import LLMTrace, TraceMessage, TraceUsage
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.storage.traces import collect_run_traces
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.records import ActionMessage, AttemptRecord, EditAttempt, Raised
from grassmoth_test_support.values import revise_fixture

if TYPE_CHECKING:
    from grassmoth.json_types import JsonObject

pytestmark = pytest.mark.contract


@pytest.fixture
def data_root(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setattr(data_module, "_DATA_URI", str(root))
    return root


def _trace(model: str = "openrouter/test-model") -> LLMTrace:
    return LLMTrace(
        messages=[TraceMessage(role="assistant", content="Done.")],
        model=model,
        usage=TraceUsage(input_tokens=3, output_tokens=5),
        total_time_seconds=0.25,
    )


def _scratch_trace(workspace_id: str, call_id: str, subroutine_id: str) -> str:
    path = storage.join(
        data_module.call_dir(workspace_id, call_id),
        "llm",
        subroutine_id,
        "trace.json",
    )
    storage.write_text(path, _trace().model_dump_json())
    return path


def test_collects_finalized_traces_and_validates_trace_identities(data_root):
    del data_root
    _scratch_trace("ws-trace", "call-test", "zeta")
    _scratch_trace("ws-trace", "call-test", "alpha")
    logs = collect_run_traces("ws-trace", "call-test")
    assert list(logs) == ["alpha", "zeta"]
    assert logs["alpha"] == _trace()
    _scratch_trace("ws-trace", "call-test", "invalid trace")
    with pytest.raises(ValueError, match="Invalid attempt trace subroutine id"):
        collect_run_traces("ws-trace", "call-test")


def test_collected_snapshot_retains_all_sources_after_scratch_removal(data_root):
    del data_root
    workspace_id, identity = "ws-snapshot", "call-snapshot"
    _scratch_trace(workspace_id, identity, "stage")
    emit_event(workspace_id, identity, StepEvent(attempt_id=UUID(int=3), status="running"))
    message = ActionMessage(
        timestamp=datetime(2026, 10, 1, tzinfo=UTC),
        severity="info",
        code="EXECUTION_ESTIMATE",
        subject="fit",
        detail="Collected once before dispatch.",
    )
    emit_message(workspace_id, identity, message)
    snapshot = collect_call_snapshot(workspace_id, identity)
    storage.rm_tree(data_module.call_dir(workspace_id, identity))
    assert snapshot.execution_messages == (message,)
    assert len(snapshot.events) == 1
    event = snapshot.events[0]
    assert isinstance(event, StepEvent)
    assert event.status == "running"
    assert event.cursor
    assert snapshot.traces == {"stage": _trace()}


def test_raised_attempt_discovers_trace_and_retry_no_longer_needs_scratch(data_root):
    del data_root
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server_test_support.helpers import initial_model_record, write_question

    store = ArtifactStore("ws-trace")
    question = write_question(store)
    initial = initial_model_record(store, question)
    previous_state = StudyRepository("ws-trace").state()
    base = asyncio.run(
        read_inputs_activity(
            ReadInputsInput(
                workspace_id="ws-trace",
                request=EditModelRequest[GitOid](
                    input=EditModelInput[GitOid](
                        parent_ref=initial.revision, dynamical_model_spec=DynamicalModelSpec()
                    )
                ),
            )
        )
    )
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=initial.revision, dynamical_model_spec=DynamicalModelSpec()
        )
    )
    _scratch_trace("ws-trace", call_identity(request), "latent-structure")
    attempt_id = UUID(int=1)
    messages = (
        ActionMessage(
            timestamp=datetime(2026, 10, 1, tzinfo=UTC),
            severity="info",
            code="EDIT_MODEL_STARTED",
            subject="edit_model",
            detail="Started.",
        ),
    )
    emit_event(
        "ws-trace", call_identity(request), StepEvent(attempt_id=attempt_id, status="running")
    )
    execution_message = ActionMessage(
        timestamp=datetime(2026, 10, 1, 0, 0, 1, tzinfo=UTC),
        severity="info",
        code="EXECUTION_ESTIMATE",
        subject="edit_model",
        detail="An execution message retained before the action finishes.",
    )
    emit_message("ws-trace", call_identity(request), execution_message)
    emit_message(
        "ws-trace", "another-call", revise_fixture(execution_message, detail="Another call.")
    )
    live = collect_call_log(
        "ws-trace",
        RunningCall(attempt_id=attempt_id, seq=1, request=request, messages=messages),
        at=datetime.now(UTC),
    )
    assert {message.kind for message in live.messages} == {"log", "progress", "trace"}
    assert [message for message in live.messages if message.kind == "log"] == [
        *messages,
        execution_message,
    ]
    publication_input = AttemptPublication(
        workspace_id="ws-trace",
        messages=messages,
        record=AttemptRecord(
            seq=1,
            attempt_id=attempt_id,
            ts=datetime.now(UTC).isoformat(),
            attempt=EditAttempt(
                action="edit_model",
                request=request,
                outcome=Raised(error_type="LLMSubroutineError", error_message="validation failed"),
            ),
        ),
    )

    publication = asyncio.run(journal_activity(publication_input))
    assert base.state.current["question"] == question
    assert StudyRepository("ws-trace").state() == previous_state
    storage.rm_tree(data_module.call_dir("ws-trace", call_identity(request)))
    asyncio.run(journal_activity(publication_input))

    record = StudyRepository("ws-trace").call(call_identity(request))
    assert record is not None
    assert record.record.attempt.outcome.status == "raised"
    repository = StudyRepository("ws-trace")
    assert repository.call(call_identity(request)) == record
    saved_metadata = repository.saved_record(publication.commit_id).model_dump()
    assert "messages" not in saved_metadata
    assert "trace_ids" not in saved_metadata
    log = read_call_log(repository, publication.commit_id)
    assert {message.kind for message in log.messages} == {"log", "progress", "trace", "failure"}
    recorded_messages = [message for message in log.messages if message.kind == "log"]
    assert recorded_messages[:-1] == [
        *messages,
        execution_message,
    ]
    assert recorded_messages[-1].code == "ACTION_FAILED_LLM_SUBROUTINE_ERROR"
    assert recorded_messages[-1].detail == "validation failed"
    assert (
        next(message for message in log.messages if message.kind == "progress").progress
        == next(message for message in live.messages if message.kind == "progress").progress
    )
    assert next(message for message in log.messages if message.kind == "trace").trace == _trace()
    assert (
        next(message for message in log.messages if message.kind == "failure").failure
        == record.record.attempt.outcome
    )


def test_subroutine_retains_conversation_before_a_tool_failure(data_root, monkeypatch):
    from grassmoth_server.actions.temporal.llm import workflow as subroutine_workflow
    from grassmoth_server.actions.temporal.llm.activities import (
        finalize_llm_subroutine_trace_activity,
    )
    from grassmoth_server.actions.temporal.llm.messages import (
        LLMSubroutineInput,
        LLMSubroutineRef,
        LLMToolSpec,
    )
    from grassmoth_server.actions.temporal.llm.storage import (
        subroutine_root,
        write_subroutine_json,
    )
    from grassmoth_server.llm.specs import EmbeddedLLMSpec

    subroutine = LLMSubroutineRef(
        workspace_id="ws-trace",
        run_id="call-test",
        subroutine_id="extraction",
        context_ref="unused-context.json",
    )
    root = subroutine_root("ws-trace", subroutine.run_id, subroutine.subroutine_id)
    conversation = storage.join(root, "conversation.json")
    messages: list[JsonObject] = [{"role": "user", "content": "Read the uploaded data."}]
    write_subroutine_json(conversation, {"messages": messages})
    start = SimpleNamespace(
        conversation_ref=conversation,
        user_message_count=1,
        call_ref_base=storage.join(root, "calls"),
        tools=[LLMToolSpec(name="read_file", description="Read the upload", parameters={})],
        tool_execution_ref_base=storage.join(root, "tools"),
        result_ref_base=storage.join(root, "results"),
    )

    async def execute(name, payload, **_options):
        if name == "start_llm_subroutine_activity":
            return start
        if name == "append_llm_user_message_activity":
            return SimpleNamespace(conversation_ref=conversation)
        if name == "finalize_llm_subroutine_trace_activity":
            return await finalize_llm_subroutine_trace_activity(payload)
        assert name == "execute_llm_tool_calls_activity"
        # A live poll already sees the complete assistant tool invocation.
        retained = LLMTrace.model_validate_json(storage.read_text(storage.join(root, "trace.json")))
        assert retained.messages[-1].tool_calls is not None
        assert retained.messages[-1].tool_calls[0]["name"] == "read_file"
        raise RuntimeError("tool service unavailable")

    async def provider(*_args):
        messages.append(
            {
                "role": "assistant",
                "content": "Inspect the source.",
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"file":"data.csv"}'},
                    }
                ],
            }
        )
        write_subroutine_json(conversation, {"messages": messages})
        return SimpleNamespace(
            conversation_ref=conversation,
            assistant_ref="unused-assistant.json",
            tool_calls=["tool-1"],
        ), 1

    monkeypatch.setattr(subroutine_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(subroutine_workflow, "_execute_openrouter_call", provider)
    with pytest.raises(RuntimeError, match="tool service unavailable"):
        asyncio.run(
            subroutine_workflow.LLMSubroutineWorkflow().run(
                LLMSubroutineInput(
                    subroutine=subroutine, llm=EmbeddedLLMSpec(model="test"), max_tool_turns=2
                )
            )
        )
    trace = collect_run_traces("ws-trace", "call-test")["extraction"]
    assert [message.role for message in trace.messages] == ["user", "assistant"]
    assert trace.messages[-1].tool_calls is not None
    assert trace.messages[-1].tool_calls[0]["arguments"] == '{"file":"data.csv"}'
    assert not list(data_root.rglob("*.partial"))


def test_failed_extraction_chunks_become_durable_messages_once(data_root):
    del data_root
    from grassmoth_server.actions.contracts.inputs import PrepareDataInput
    from grassmoth_server.actions.contracts.preparation import FileSourceRef
    from grassmoth_server.actions.contracts.progress import ExtractionWorkerEvent
    from grassmoth_server.actions.contracts.requests import PrepareDataRequest
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.records import PrepareAttempt
    from grassmoth_server_test_support.storage import seed_record
    from grassmoth_test_support.helpers import make_model

    store = ArtifactStore("ws-extraction-errors")
    model = make_model(["outcome"])
    parent = seed_record(
        store,
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    request = PrepareDataRequest[GitOid, FileSourceRef](
        input=PrepareDataInput(
            dynamical_model_spec_ref=parent.revision,
            source=FileSourceRef(files=("source.csv",)),
            extraction={
                model.indicators[0].observation.id: {
                    "kind": "semantic",
                    "how_to_measure": "Read it",
                }
            },
        )
    )
    attempt_id = UUID(int=9)
    identity = call_identity(request)
    errors = {7: "Provider request exhausted retries", 8: "No usable extraction"}
    for worker_id in (7, 8, 7):
        emit_event(
            store.workspace_id,
            identity,
            ExtractionWorkerEvent(
                attempt_id=attempt_id,
                worker_id=worker_id,
                state="failed",
                n_windows=1,
                n_extractions=0,
                error=errors[worker_id],
            ),
        )
    emit_event(
        store.workspace_id,
        identity,
        ExtractionWorkerEvent(
            attempt_id=attempt_id,
            worker_id=9,
            state="completed",
            n_windows=1,
            n_extractions=2,
        ),
    )
    publication_input = AttemptPublication(
        workspace_id=store.workspace_id,
        record=AttemptRecord(
            seq=2,
            attempt_id=attempt_id,
            ts="2026-11-03T00:00:00Z",
            attempt=PrepareAttempt(
                action="prepare_data",
                request=request,
                outcome=Raised(error_type="ValueError", error_message="Panel assembly failed"),
            ),
        ),
    )
    publication = asyncio.run(journal_activity(publication_input))
    messages = tuple(
        message
        for message in read_call_log(
            StudyRepository(store.workspace_id), publication.commit_id
        ).messages
        if message.kind == "log"
    )
    assert [(message.subject, message.severity, message.detail) for message in messages[:-1]] == [
        (f"extraction_worker:{worker_id}", "error", detail) for worker_id, detail in errors.items()
    ]
    assert all(message.code == "EXTRACTION_CHUNK_FAILED" for message in messages[:-1])
    assert messages[-1].code == "ACTION_FAILED_VALUE_ERROR"
    assert messages[-1].detail == "Panel assembly failed"
    storage.rm_tree(data_module.call_dir(store.workspace_id, identity))
    assert asyncio.run(journal_activity(publication_input)) == publication
    saved = read_call_log(StudyRepository(store.workspace_id), publication.commit_id)
    assert tuple(message for message in saved.messages if message.kind == "log") == messages
