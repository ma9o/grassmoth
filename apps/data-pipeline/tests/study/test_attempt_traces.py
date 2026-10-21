"""Journal-owned durable LLM traces."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from nof1_causal_lab.actions.call_logs import collect_call_log, read_call_log
from nof1_causal_lab.actions.contracts.bodies import EditModelInput
from nof1_causal_lab.actions.contracts.progress import StepEvent
from nof1_causal_lab.actions.contracts.requests import EditModelRequest, call_identity
from nof1_causal_lab.actions.progress import emit_event
from nof1_causal_lab.actions.temporal.study.activities import journal_activity, read_inputs_activity
from nof1_causal_lab.actions.temporal.study.messages import AttemptPublication, ReadInputsInput
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.llm.traces import LLMTrace, TraceMessage, TraceUsage
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.storage.traces import collect_run_traces, trace_log_path
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.records import ActionMessage, AttemptRecord, EditAttempt, Raised

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject

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


def _scratch_trace(workspace_id: str, seq: int, subroutine_id: str) -> str:
    path = storage.join(
        data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"),
        "llm",
        subroutine_id,
        "trace.json",
    )
    storage.write_text(path, _trace().model_dump_json())
    return path


def test_collects_finalized_traces_and_validates_commit_paths(data_root):
    del data_root
    _scratch_trace("ws-trace", 1, "zeta")
    _scratch_trace("ws-trace", 1, "alpha")
    logs = collect_run_traces("ws-trace", 1)
    assert list(logs) == ["traces/alpha.json", "traces/zeta.json"]
    with pytest.raises(ValueError, match="Invalid attempt trace subroutine id"):
        trace_log_path("../scratch")


def test_raised_attempt_discovers_trace_and_retry_no_longer_needs_scratch(data_root):
    del data_root
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from tests.support.helpers import write_question

    question = write_question(ArtifactStore("ws-trace"))
    previous_state = StudyRepository("ws-trace").state()
    base = asyncio.run(
        read_inputs_activity(
            ReadInputsInput(
                workspace_id="ws-trace",
                request=EditModelRequest[GitOid](
                    input=EditModelInput[GitOid](
                        parent_ref=question.revision, dynamical_model_spec=DynamicalModelSpec()
                    )
                ),
            )
        )
    )
    _scratch_trace("ws-trace", 1, "latent-structure")
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=question.revision, dynamical_model_spec=DynamicalModelSpec()
        )
    )
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
    emit_event("ws-trace", StepEvent(attempt_id=attempt_id, status="running"))
    live = collect_call_log(
        "ws-trace", seq=1, attempt_id=attempt_id, messages=messages, at=datetime.now(UTC)
    )
    assert {message.kind for message in live.messages} == {"log", "progress", "trace"}
    publication_input = AttemptPublication(
        workspace_id="ws-trace",
        record=AttemptRecord(
            seq=1,
            attempt_id=attempt_id,
            messages=messages,
            ts="2026-10-01T00:00:00Z",
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
    storage.rm_tree(data_module.scratch_run_dir("ws-trace", "seq-000001"))
    storage.rm_tree(data_module.scratch_events_dir("ws-trace"))
    asyncio.run(journal_activity(publication_input))

    record = StudyRepository("ws-trace").read_attempt(1)
    assert record is not None
    assert record.record.attempt.outcome.status == "raised"
    assert record.record.trace_ids == ("latent-structure",)
    repository = StudyRepository("ws-trace")
    assert repository.call(call_identity(request)) == record
    log = read_call_log(repository, publication.commit_id)
    assert {message.kind for message in log.messages} == {"log", "progress", "trace", "failure"}
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
    from nof1_causal_lab.actions.temporal.llm import workflow as subroutine_workflow
    from nof1_causal_lab.actions.temporal.llm.activities import (
        finalize_llm_subroutine_trace_activity,
    )
    from nof1_causal_lab.actions.temporal.llm.messages import (
        LLMSubroutineInput,
        LLMSubroutineRef,
        LLMToolSpec,
    )
    from nof1_causal_lab.actions.temporal.llm.storage import subroutine_root, write_subroutine_json
    from nof1_causal_lab.llm.specs import EmbeddedLLMSpec

    subroutine = LLMSubroutineRef(
        workspace_id="ws-trace",
        run_id="seq-000001",
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
    trace = LLMTrace.model_validate_json(
        collect_run_traces("ws-trace", 1)["traces/extraction.json"]
    )
    assert [message.role for message in trace.messages] == ["user", "assistant"]
    assert trace.messages[-1].tool_calls is not None
    assert trace.messages[-1].tool_calls[0]["arguments"] == '{"file":"data.csv"}'
    assert not list(data_root.rglob("*.partial"))
