import json
import uuid
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth_server.actions.prepare_data.worker_output import ExtractionRow, WorkerOutput
from grassmoth_server.actions.temporal.extraction.activities import OpenRouterActivities
from grassmoth_server.actions.temporal.extraction.messages import (
    ExtractionChunkWorkflowInput,
    MeasurementChunkRef,
    ProgressEventInput,
)
from grassmoth_server.actions.temporal.extraction.workflow import ExtractionChunkWorkflow
from grassmoth_server.actions.temporal.llm.activities import (
    append_llm_repair_message_activity,
    execute_llm_tool_calls_activity,
)
from grassmoth_server.actions.temporal.llm.messages import (
    AppendLLMRepairMessageInput,
    LLMSubroutineRef,
    LLMToolExecutionInput,
    LLMToolSpec,
    OpenRouterCallInput,
    StoredConversation,
)
from grassmoth_server.actions.temporal.llm.storage import read_subroutine_json
from grassmoth_server.llm.openrouter import create_openrouter_client
from grassmoth_server.llm.specs import EmbeddedLLMSpec
from grassmoth_server.storage import backend as storage
from grassmoth_server_test_support.helpers import run_async

pytestmark = pytest.mark.timeout(240)


@pytest.fixture(autouse=True)
def isolated_measurement_store(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path / "data"))


@pytest.mark.contract
@pytest.mark.parametrize(
    ("payload", "required_fields"),
    [
        (
            {"event": "grassmoth.step", "status": "running"},
            ("status",),
        ),
        ({"event": "grassmoth.extraction.plan", "total_workers": 2}, ("total_workers",)),
        (
            {
                "event": "grassmoth.extraction.worker",
                "worker_id": 0,
                "state": "running",
                "n_windows": 2,
            },
            ("worker_id", "state", "n_windows"),
        ),
        (
            {
                "event": "grassmoth.extraction.snapshot",
                "total_workers": 2,
                "pending_workers": 1,
                "running_workers": 1,
                "completed_workers": 0,
                "failed_workers": 0,
            },
            ("total_workers", "failed_workers"),
        ),
    ],
)
def test_progress_event_variants_round_trip_through_temporal(payload, required_fields):
    from temporalio.contrib.pydantic import pydantic_data_converter

    adapter = TypeAdapter(ProgressEventInput)
    event = {"attempt_id": str(uuid.uuid4()), **payload}
    message = adapter.validate_python({"workspace_id": "test", "event": event})

    async def round_trip():
        encoded = await pydantic_data_converter.encode([message])
        return await pydantic_data_converter.decode(encoded, [ProgressEventInput])

    assert run_async(round_trip()) == [message]
    for field in (*required_fields, "attempt_id"):
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "workspace_id": "test",
                    "event": {name: value for name, value in event.items() if name != field},
                }
            )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"workspace_id": "test", "event": {**event, "model": "unrelated field"}}
        )


@pytest.mark.contract
def test_call_openrouter_activity_reuses_persisted_call_result(monkeypatch, tmp_path):
    import grassmoth_server.llm.openrouter as openrouter_client

    calls: list[list[dict[str, Any]]] = []
    conversation_ref = str(tmp_path / "conversation.json")
    next_conversation_ref = str(tmp_path / "conversation-next.json")
    call_ref = str(tmp_path / "call.json")
    assistant_ref = str(tmp_path / "assistant.json")
    storage.write_text(
        conversation_ref,
        json.dumps({"messages": [{"role": "user", "content": "extract"}]}),
    )

    async def fake_call_model(
        model_name, messages, *, client, tools=None, config=None, log_label=None
    ):
        del client, tools, config, log_label
        calls.append(messages)
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": []},
            "completion": "",
            "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
            "model": model_name,
            "time": 0.25,
            "stop_reason": "tool_calls",
        }

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)

    activity_input = OpenRouterCallInput(
        conversation_ref=conversation_ref,
        next_conversation_ref=next_conversation_ref,
        call_ref=call_ref,
        assistant_ref=assistant_ref,
        llm=EmbeddedLLMSpec(model="openrouter/mock", timeout=120),
        tools=[
            LLMToolSpec(
                name="validate_extractions",
                description="Validate output.",
                parameters={
                    "type": "object",
                    "properties": {
                        "output_json": {"type": "string", "description": "Worker output JSON."}
                    },
                    "required": ["output_json"],
                    "additionalProperties": False,
                },
            )
        ],
        log_label="test",
    )

    activities = OpenRouterActivities(cast("Any", object()))
    first = run_async(activities.call_openrouter_activity(activity_input))
    second = run_async(activities.call_openrouter_activity(activity_input))

    assert len(calls) == 1
    assert first == second
    assert first.model == "openrouter/mock"
    assert storage.exists(call_ref)


@pytest.mark.contract
def test_execute_llm_tool_calls_activity_dispatches_by_tool_name(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")

    output_json = json.dumps(
        {
            "extractions": [
                {
                    "window_start": "2026-01-01T00:00:00",
                    "indicator_id": "indicator:steps",
                    "value": 1000,
                }
            ]
        }
    )
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "submit_extractions",
            "arguments": json.dumps({"output_json": output_json}),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "question": "does exercise improve sleep?",
                "window_text": "2026-01-01T00:00:00: steps were 1000",
                "window_starts": ["2026-01-01T00:00:00"],
                "measurement_structure": {
                    "source": {"files": ["source.csv"]},
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "observation": {
                                "id": "indicator:steps",
                                "name": "steps",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "extraction": {"kind": "semantic", "how_to_measure": "Measure steps"},
                        }
                    ],
                },
            }
        ),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "extract"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                subroutine=LLMSubroutineRef(
                    workspace_id="ws-test",
                    run_id="seq-000001",
                    subroutine_id="measurement-extraction",
                    context_ref=context_ref,
                ),
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="validate_extractions",
                        description="Validate worker extraction output JSON.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "output_json": {
                                    "type": "string",
                                    "description": "The JSON string containing the worker output.",
                                }
                            },
                            "required": ["output_json"],
                            "additionalProperties": False,
                        },
                    ),
                    LLMToolSpec(
                        name="submit_extractions",
                        description="Submit worker extraction output JSON.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "output_json": {
                                    "type": "string",
                                    "description": "The JSON string containing the worker output.",
                                }
                            },
                            "required": ["output_json"],
                            "additionalProperties": False,
                        },
                    ),
                ],
            )
        )
    )

    assert result.terminal_success is True
    assert result.result_ref == result_ref
    assert result.tool_calls_fired == ["submit_extractions"]
    assert (
        WorkerOutput.model_validate_json(json.dumps(read_subroutine_json(result_ref)))
        .extractions[0]
        .value
        == 1000
    )


@pytest.mark.contract
def test_execute_llm_tool_calls_activity_returns_recoverable_tool_exception(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    tool_call = {
        "id": "call-submit",
        "type": "function",
        "function": {
            "name": "validate_extractions",
            "arguments": json.dumps({"output_json": "{}"}),
        },
    }
    storage.write_text(context_ref, json.dumps({"extract_dir": str(tmp_path)}))
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "extract"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                subroutine=LLMSubroutineRef(
                    workspace_id="ws-test",
                    run_id="seq-000001",
                    subroutine_id="raw-data",
                    context_ref=context_ref,
                ),
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="validate_extractions",
                        description="Submit table.",
                        parameters={
                            "type": "object",
                            "properties": {"output_json": {"type": "string"}},
                            "required": ["output_json"],
                            "additionalProperties": False,
                        },
                    )
                ],
            )
        )
    )

    assert result.terminal_success is False
    assert result.result_ref is None
    assert "Tool execution failed:" in json.dumps(
        read_subroutine_json(result.conversation_ref, StoredConversation)["messages"][-1]
    )


@pytest.mark.contract
def test_append_llm_repair_message_activity_persists_repair_turn(tmp_path):
    conversation_ref = str(tmp_path / "conversation.json")
    next_conversation_ref = str(tmp_path / "conversation-repair.json")
    storage.write_text(
        conversation_ref,
        json.dumps({"messages": [{"role": "user", "content": "try tool"}]}),
    )

    result = run_async(
        append_llm_repair_message_activity(
            AppendLLMRepairMessageInput(
                subroutine=LLMSubroutineRef(
                    workspace_id="ws-test",
                    run_id="seq-000001",
                    subroutine_id="repair",
                    context_ref=str(tmp_path / "context.json"),
                ),
                conversation_ref=conversation_ref,
                next_conversation_ref=next_conversation_ref,
                error_text="provider rejected malformed tool context",
                tools=[
                    LLMToolSpec(
                        name="validate_extractions",
                        description="Submit table.",
                        parameters={
                            "type": "object",
                            "properties": {"output_json": {"type": "string"}},
                            "required": ["output_json"],
                            "additionalProperties": False,
                        },
                    )
                ],
            )
        )
    )

    assert result.conversation_ref == next_conversation_ref
    messages = read_subroutine_json(next_conversation_ref, StoredConversation)["messages"]
    assert messages[-1]["role"] == "user"
    content = messages[-1]["content"]
    assert isinstance(content, str)
    assert "Your previous response could not be processed" in content
    assert "validate_extractions" in content


@pytest.mark.contract
def test_execute_llm_tool_calls_activity_requires_terminal_acknowledgement(tmp_path, monkeypatch):
    from grassmoth_server.actions.temporal.llm import activities as llm_subroutine_activities

    async def incomplete_validation(**_kwargs):
        return "Validation still pending", None

    monkeypatch.setattr(llm_subroutine_activities, "execute_subroutine_tool", incomplete_validation)
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    (tmp_path / "observations.csv").write_text("timestamp,steps\n2026-01-01T08:00:00,1000\n")

    tool_call = {
        "id": "call-list",
        "type": "function",
        "function": {
            "name": "validate_extractions",
            "arguments": json.dumps({"path": "."}),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "extract_dir": str(tmp_path),
                "dataframe_ref": str(tmp_path / "latest-dataframe.ipc"),
            }
        ),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "extract"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                subroutine=LLMSubroutineRef(
                    workspace_id="ws-test",
                    run_id="seq-000001",
                    subroutine_id="raw-data",
                    context_ref=context_ref,
                ),
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="validate_extractions",
                        description="Validate extraction.",
                        kind="terminal",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": [],
                            "additionalProperties": False,
                        },
                    )
                ],
            )
        )
    )

    assert result.terminal_success is False
    assert result.result_ref is None
    assert storage.exists(result_ref) is False


@pytest.mark.workflow
@pytest.mark.parametrize("reused", [False, True])
def test_extraction_chunk_workflow_runs_shared_llm_subroutine(monkeypatch, tmp_path, reused):
    from temporalio.testing import WorkflowEnvironment

    import grassmoth_server.llm.openrouter as openrouter_client
    from grassmoth_server.actions.temporal.client import pydantic_data_converter
    from grassmoth_server.actions.temporal.worker import build_openrouter_worker, build_worker
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    output_json = json.dumps(
        {
            "extractions": [
                {
                    "window_start": "2026-01-01T00:00:00",
                    "indicator_id": "indicator:steps",
                    "value": 1000,
                }
            ]
        }
    )

    async def fake_call_model(
        model_name, messages, *, client, tools=None, config=None, log_label=None
    ):
        assert not reused, "A retained cache hit must skip the LLM subroutine"
        del messages, tools, config, log_label
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-extract",
                        "type": "function",
                        "function": {
                            "name": "validate_extractions",
                            "arguments": json.dumps({"output_json": output_json}),
                        },
                    }
                ],
            },
            "completion": "",
            "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
            "model": model_name,
            "time": 0.25,
            "stop_reason": "tool_calls",
        }

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)

    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    cached_result_ref = str(tmp_path / "cached-result.json") if reused else None
    if cached_result_ref is not None:
        storage.write_text(cached_result_ref, output_json)
    spec_ref = str(tmp_path / "chunk.json")
    storage.write_text(
        spec_ref,
        json.dumps(
            {
                "worker_id": 0,
                "cache_ref": str(tmp_path / "extraction-cache.json"),
                "question": "does exercise improve sleep?",
                "window_text": "2026-01-01T00:00:00: steps were 1000",
                "window_starts": ["2026-01-01T00:00:00"],
                "measurement_structure": {
                    "source": {"files": ["source.csv"]},
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "observation": {
                                "id": "indicator:steps",
                                "name": "steps",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "extraction": {"kind": "semantic", "how_to_measure": "Measure steps"},
                        }
                    ],
                },
            }
        ),
    )

    async def scenario():
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                create_openrouter_client() as transport,
                build_worker(env.client, task_queue="test-studies"),
                build_openrouter_worker(env.client, transport),
            ):
                result = await env.client.execute_workflow(
                    ExtractionChunkWorkflow.run,
                    ExtractionChunkWorkflowInput(
                        workspace_id=workspace_id,
                        run_id="seq-000001",
                        chunk=MeasurementChunkRef(
                            worker_id=0,
                            n_windows=1,
                            spec_ref=spec_ref,
                            cached_result_ref=cached_result_ref,
                        ),
                        llm=EmbeddedLLMSpec(
                            model="openrouter/mock-extraction",
                            timeout=120,
                        ),
                        max_tool_turns=3,
                    ),
                    id=f"extract-chunk-{workspace_id}",
                    task_queue="test-studies",
                )
        finally:
            await env.shutdown()
        return result

    result = run_async(scenario())

    assert result.status == "completed"
    assert result.n_extractions == 1
    assert result.n_llm_calls == (0 if reused else 1)
    assert result.reused is reused
    assert result.result_ref is not None
    assert (
        TypeAdapter[list[ExtractionRow]](list[ExtractionRow]).validate_python(
            storage.read_json(result.result_ref)["dataframe"]
        )[0]["value"]
        == "1000"
    )
    from grassmoth_server.llm.traces import LLMTrace

    trace_path = storage.join(
        data_module.scratch_run_dir(workspace_id, "seq-000001"),
        "llm",
        "measurement-chunk-000000-attempt-001",
        "trace.json",
    )
    if reused:
        assert not storage.exists(trace_path)
    else:
        trace = LLMTrace.model_validate(storage.read_json(trace_path))
        assert trace.model == "openrouter/mock-extraction"
        assert trace.usage.input_tokens == 3
