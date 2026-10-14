"""Shared activities for workflow-driven LLM subroutines."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, assert_never
from uuid import uuid4

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from pydantic import BaseModel, TypeAdapter, ValidationError
from temporalio import activity
from typing_extensions import TypedDict

from nof1_causal_lab.actions.temporal.llm.context import subroutine_context_messages
from nof1_causal_lab.actions.temporal.llm.messages import (
    AppendLLMRepairMessageInput,
    AppendLLMRepairMessageResult,
    AppendLLMUserMessageInput,
    AppendLLMUserMessageResult,
    HarnessToolExecutionResult,
    HarnessToolRequest,
    HarnessTurnInput,
    HarnessTurnResult,
    LLMSubroutineRef,
    LLMSubroutineStart,
    LLMSubroutineTraceInput,
    LLMSubroutineTraceResult,
    LLMToolExecutionInput,
    LLMToolExecutionResult,
    LLMToolSpec,
    OpenRouterCallResult,
    StoredConversation,
)
from nof1_causal_lab.actions.temporal.llm.storage import (
    read_subroutine_json,
    subroutine_conversation_path,
    subroutine_root,
    write_subroutine_json,
    write_subroutine_trace,
)
from nof1_causal_lab.actions.temporal.llm.tools import execute_subroutine_tool
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.storage import backend as storage

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from nof1_causal_lab.llm.openrouter import Tool
    from nof1_causal_lab.llm.sessions import TurnResult
    from nof1_causal_lab.llm.traces import TraceMessage


def _trace_message(message: JsonObject) -> TraceMessage:
    from nof1_causal_lab.llm.traces import TraceMessage

    content = str(message.get("content", ""))
    raw_calls = message.get("tool_calls")
    calls = (
        None
        if raw_calls is None
        else TypeAdapter(tuple[ChatCompletionMessageFunctionToolCallParam, ...]).validate_python(
            raw_calls
        )
    )
    return TraceMessage.model_validate(
        {
            "role": message["role"],
            "content": content,
            "reasoning": message.get("reasoning"),
            "tool_calls": None
            if calls is None
            else tuple(
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in calls
            ),
            "tool_call_id": message.get("tool_call_id"),
            "tool_name": message.get("name"),
            "tool_result": content if message["role"] == "tool" else None,
            "tool_is_error": message.get("error") is not None,
        }
    )


_RECOVERABLE_TOOL_EXECUTION_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValidationError,
    ValueError,
)


class SubroutineToolMessage(TypedDict):
    """A tool response linked to the assistant call, with any execution error retained."""

    role: str
    content: str
    tool_call_id: str  # noqa: V107 -- OpenAI reads this required wire key when the tool message is sent.
    name: str
    error: str | None


class HarnessState(BaseModel):
    """Mutable harness session state retained between turns, including raw trace events."""

    raw_events: list[JsonObject]
    turn_index: int
    session_id: str | None
    session_jsonl: str | None = None


def _subroutine_tool_message(
    tool_call: JsonObject, content: str, error: str | None = None
) -> SubroutineToolMessage:
    invocation = TypeAdapter(ChatCompletionMessageFunctionToolCallParam).validate_python(tool_call)
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": invocation["id"],
        "name": invocation["function"]["name"],
        "error": error,
    }


def _terminal_tool_succeeded(tool: LLMToolSpec, output: str, error: str | None) -> bool:
    if tool.kind != "terminal" or error is not None:
        return False
    result_text = output.strip()
    return result_text == "VALID"


def _tool_execution_failed(exc: BaseException) -> str:
    return f"Tool execution failed: {exc}"


@activity.defn
async def start_llm_subroutine_activity(
    activity_input: LLMSubroutineRef,
) -> LLMSubroutineStart:
    """Persist the initial extraction conversation and allocate paths for subsequent turns."""
    system_prompt, user_messages, tools = subroutine_context_messages(activity_input.context_ref)
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    conversation_ref = subroutine_conversation_path(
        activity_input.workspace_id,
        activity_input.run_id,
        activity_input.subroutine_id,
        "turn-000-system.json",
    )
    root = subroutine_root(
        activity_input.workspace_id, activity_input.run_id, activity_input.subroutine_id
    )
    write_subroutine_json(
        conversation_ref,
        {
            "messages": messages,
            "context_ref": activity_input.context_ref,
            "user_messages": user_messages,
        },
    )

    return LLMSubroutineStart(
        conversation_ref=conversation_ref,
        conversation_ref_base=storage.join(root, "conversation"),
        user_message_count=len(user_messages),
        tools=tools,
        call_ref_base=storage.join(root, "calls"),
        assistant_ref_base=storage.join(root, "assistants"),
        tool_execution_ref_base=storage.join(root, "tool-executions"),
        harness_state_ref=storage.join(root, "harness-state.json"),
        harness_tool_ref_base=storage.join(root, "harness-tools"),
        result_ref_base=storage.join(root, "results"),
    )


@activity.defn
async def append_llm_user_message_activity(
    activity_input: AppendLLMUserMessageInput,
) -> AppendLLMUserMessageResult:
    """Append the selected context prompt as a new persisted conversation.

    Args:
        activity_input: Subroutine context, prior conversation, and zero-based prompt index.

    Returns:
        Reference to the conversation containing the additional user message.

    Raises:
        IndexError: The requested prompt index is past the context's user messages.
    """
    system_prompt, user_messages, _tool = subroutine_context_messages(
        activity_input.subroutine.context_ref
    )
    del system_prompt
    if activity_input.user_message_index >= len(user_messages):
        raise IndexError(f"user message index {activity_input.user_message_index} out of range")

    conversation = read_subroutine_json(activity_input.conversation_ref, StoredConversation)
    messages = [
        *conversation["messages"],
        {"role": "user", "content": user_messages[activity_input.user_message_index]},
    ]
    conversation_ref = subroutine_conversation_path(
        activity_input.subroutine.workspace_id,
        activity_input.subroutine.run_id,
        activity_input.subroutine.subroutine_id,
        f"user-{activity_input.user_message_index + 1:03d}.json",
    )
    write_subroutine_json(conversation_ref, {"messages": messages})
    return AppendLLMUserMessageResult(conversation_ref=conversation_ref)


@activity.defn
async def append_llm_repair_message_activity(
    activity_input: AppendLLMRepairMessageInput,
) -> AppendLLMRepairMessageResult:
    """Persist tool-repair feedback, reusing the next conversation if it already exists."""
    if storage.exists(activity_input.next_conversation_ref):
        return AppendLLMRepairMessageResult(conversation_ref=activity_input.next_conversation_ref)

    from nof1_causal_lab.llm.traces import _tool_retry_message

    conversation = read_subroutine_json(activity_input.conversation_ref, StoredConversation)
    repair_message = _tool_retry_message(activity_input.error_text, activity_input.tools)
    write_subroutine_json(
        activity_input.next_conversation_ref,
        {"messages": [*conversation["messages"], repair_message]},
    )
    return AppendLLMRepairMessageResult(conversation_ref=activity_input.next_conversation_ref)


@activity.defn
async def execute_llm_tool_calls_activity(
    activity_input: LLMToolExecutionInput,
) -> LLMToolExecutionResult:
    """Execute the stored assistant's tool requests and persist responses for retry reuse."""
    if storage.exists(activity_input.execution_ref):
        return LLMToolExecutionResult.model_validate(
            read_subroutine_json(activity_input.execution_ref)["result"]
        )

    assistant_output = read_subroutine_json(activity_input.assistant_ref)
    assistant_message = TypeAdapter(JsonObject).validate_python(assistant_output["message"])
    conversation = read_subroutine_json(activity_input.conversation_ref, StoredConversation)
    messages = list(conversation["messages"])
    tool_messages: list[SubroutineToolMessage] = []
    tool_calls_fired: list[str] = []
    terminal_success = False
    captured_result_ref: str | None = None
    tool_by_name = {tool.name: tool for tool in activity_input.tools}

    for _tool_index, tool_call in enumerate(
        TypeAdapter(list[JsonObject]).validate_python(assistant_message.get("tool_calls") or [])
    ):
        fn = TypeAdapter(JsonObject).validate_python(tool_call.get("function") or {})
        tool_name = str(fn.get("name") or tool_call.get("name", ""))
        tool_calls_fired.append(tool_name)
        tool = tool_by_name.get(tool_name)
        if tool is None:
            result_text = f"Unknown tool: {tool_name}"
            tool_messages.append(
                _subroutine_tool_message(tool_call, result_text, error=result_text)
            )
            continue
        try:
            raw_args = str(fn.get("arguments") or tool_call.get("arguments", "{}") or "{}")
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must decode to a JSON object")
            result_text, tool_result_ref = await execute_subroutine_tool(
                activity_input=activity_input,
                args=args,
                result_ref=activity_input.result_ref,
            )
        except json.JSONDecodeError as exc:
            result_text = f"JSON parse error: {exc}"
            error_text = None
        except _RECOVERABLE_TOOL_EXECUTION_ERRORS as exc:
            result_text = _tool_execution_failed(exc)
            error_text = str(exc)
        else:
            error_text = None
            if tool_result_ref is not None:
                captured_result_ref = tool_result_ref
        tool_messages.append(_subroutine_tool_message(tool_call, result_text, error=error_text))
        if _terminal_tool_succeeded(tool, result_text, error_text):
            terminal_success = True

    next_conversation_ref = subroutine_conversation_path(
        activity_input.subroutine.workspace_id,
        activity_input.subroutine.run_id,
        activity_input.subroutine.subroutine_id,
        f"tool-execution-{Path(activity_input.execution_ref).stem}.json",
    )
    next_messages = [*messages, *tool_messages]
    write_subroutine_json(next_conversation_ref, {"messages": next_messages})

    result = LLMToolExecutionResult(
        conversation_ref=next_conversation_ref,
        terminal_success=terminal_success,
        result_ref=captured_result_ref,
        tool_calls_fired=tool_calls_fired,
    )
    write_subroutine_json(activity_input.execution_ref, {"result": result.model_dump(mode="json")})
    return result


def _harness_tool_request_path(base: str, folder: str, request_id: str) -> str:
    return storage.join(base, folder, f"{request_id}.json")


def _build_harness_bridge_tools(activity_input: HarnessTurnInput) -> list[Tool]:
    from nof1_causal_lab.llm.openrouter import Tool

    if not activity_input.tools:
        raise ValueError("harness tool requested for a no-tool LLM subroutine")

    def _build_one(tool: LLMToolSpec) -> Tool:
        async def _execute(**kwargs: str) -> str:
            request_id = uuid4().hex
            response_ref = _harness_tool_request_path(
                activity_input.harness_tool_ref_base,
                "responses",
                request_id,
            )
            request_ref = _harness_tool_request_path(
                activity_input.harness_tool_ref_base,
                "requests",
                request_id,
            )
            request = HarnessToolRequest(
                request_id=request_id,
                subroutine=activity_input.subroutine,
                result_ref=activity_input.result_ref,
                tool=tool,
                tool_name=tool.name,
                arguments=dict(kwargs),
                request_ref=request_ref,
                response_ref=response_ref,
            )
            write_subroutine_json(request_ref, request.model_dump(mode="json"))
            from nof1_causal_lab.actions.temporal.client import connect_client

            client = await connect_client()
            handle = client.get_workflow_handle(
                activity_input.workflow_id,
                run_id=activity_input.workflow_run_id,
            )
            await handle.signal("harness_tool_requested", request)
            while not storage.exists(response_ref):
                await asyncio.sleep(0.2)
            response = HarnessToolExecutionResult.model_validate(read_subroutine_json(response_ref))
            return response.output

        return Tool(
            name=tool.name,
            description=tool.description,
            parameters=dict(tool.parameters),
            execute=_execute,
            stop_on_success=tool.kind == "terminal",
            success_output="VALID" if tool.kind == "terminal" else None,
        )

    return [_build_one(tool) for tool in activity_input.tools]


async def _await_harness_turn(turn: Awaitable[TurnResult], subroutine_id: str) -> TurnResult:
    """Await a harness turn while heartbeating from the activity task.

    Harness MCP callbacks execute in the server's task context, where Temporal's
    activity context is unavailable. The owning activity task carries the
    heartbeat instead while the harness and any bridged tool request run.
    """
    task = asyncio.ensure_future(turn)
    while not task.done():
        activity.heartbeat({"waiting_for_harness_turn": subroutine_id})
        await asyncio.sleep(1)
    return await task


@activity.defn
async def execute_harness_tool_request_activity(
    activity_input: HarnessToolRequest,
) -> HarnessToolExecutionResult:
    """Execute one harness tool request and persist its correlated response, including failures."""
    if storage.exists(activity_input.response_ref):
        return HarnessToolExecutionResult.model_validate(
            read_subroutine_json(activity_input.response_ref)
        )

    output = ""
    captured_result_ref: str | None = None
    success = False
    if activity_input.tool_name != activity_input.tool.name:
        output = f"Unknown tool: {activity_input.tool_name}"
    else:
        try:
            output, captured_result_ref = await execute_subroutine_tool(
                activity_input=activity_input,
                args=activity_input.arguments,
                result_ref=activity_input.result_ref,
            )
        except json.JSONDecodeError as exc:
            output = f"JSON parse error: {exc}"
        except _RECOVERABLE_TOOL_EXECUTION_ERRORS as exc:
            output = _tool_execution_failed(exc)
        else:
            success = _terminal_tool_succeeded(activity_input.tool, output, None)

    result = HarnessToolExecutionResult(
        request_id=activity_input.request_id,
        tool_name=activity_input.tool_name,
        output=output,
        result_ref=captured_result_ref,
        success=success,
    )
    write_subroutine_json(activity_input.response_ref, result.model_dump(mode="json"))
    return result


@activity.defn
async def run_harness_turn_activity(activity_input: HarnessTurnInput) -> HarnessTurnResult:
    """Resume a harness session for the next extraction prompt and retain its turn trace."""
    from nof1_causal_lab.llm.harness.claude import open_claude_harness_session
    from nof1_causal_lab.llm.harness.codex import open_codex_harness_session
    from nof1_causal_lab.llm.harness.pi import open_pi_harness_session

    _system_prompt, user_messages, _tools = subroutine_context_messages(
        activity_input.subroutine.context_ref
    )
    if activity_input.user_message_index >= len(user_messages):
        raise IndexError(f"user message index {activity_input.user_message_index} out of range")

    state = HarnessState.model_validate(
        read_subroutine_json(activity_input.harness_state_ref)
        if storage.exists(activity_input.harness_state_ref)
        else {"raw_events": [], "turn_index": 0, "session_id": None},
    )
    raw_events = state.raw_events
    turn_index = state.turn_index
    session_id = state.session_id
    tools = [] if not activity_input.tools else _build_harness_bridge_tools(activity_input)
    user_message = user_messages[activity_input.user_message_index]

    if activity_input.llm.harness == "claude-code":
        async with open_claude_harness_session(
            tools=tools,
            system_prompt=_system_prompt if turn_index == 0 else None,
            model=activity_input.llm.model,
            executable=activity_input.llm.bin or "claude",
            effort=activity_input.llm.effort,
            max_turns=activity_input.llm.max_turns,
            max_budget_usd=activity_input.llm.max_budget_usd,
            fallback_model=activity_input.llm.fallback_model,
            timeout_seconds=900.0,
            log_label=activity_input.log_label,
            session_id=session_id,
            initial_events=raw_events,
            turn_index=turn_index,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                activity_input.subroutine.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": session.session_id,
            }

    elif activity_input.llm.harness == "codex":
        async with open_codex_harness_session(
            tools=tools,
            system_prompt=_system_prompt if turn_index == 0 else None,
            model=activity_input.llm.model,
            executable=activity_input.llm.bin or "codex",
            reasoning_effort=activity_input.llm.reasoning_effort,
            service_tier=activity_input.llm.service_tier,
            timeout_seconds=float(activity_input.llm.timeout or 1800),
            log_label=activity_input.log_label,
            initial_events=raw_events,
            turn_index=turn_index,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                activity_input.subroutine.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": None,
            }
    elif activity_input.llm.harness == "pi":
        async with open_pi_harness_session(
            tools=tools,
            system_prompt=_system_prompt,
            provider=activity_input.llm.provider or "openai-codex",
            model=activity_input.llm.model,
            thinking=activity_input.llm.thinking or "high",
            executable=activity_input.llm.bin or "pi",
            timeout_seconds=float(activity_input.llm.timeout or 1800),
            log_label=activity_input.log_label,
            initial_events=raw_events,
            initial_session_jsonl=state.session_jsonl,
            session_id=session_id,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                activity_input.subroutine.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": session.session_id,
                "session_jsonl": session.session_jsonl,
            }
    else:
        assert_never(activity_input.llm)

    write_subroutine_json(activity_input.harness_state_ref, next_state)
    trace_ref = storage.join(
        subroutine_root(
            activity_input.subroutine.workspace_id,
            activity_input.subroutine.run_id,
            activity_input.subroutine.subroutine_id,
        ),
        "traces",
        f"harness-turn-{activity_input.user_message_index + 1:03d}.json",
    )
    write_subroutine_json(trace_ref, result.trace.model_dump(mode="json"))
    return HarnessTurnResult(
        harness_state_ref=activity_input.harness_state_ref,
        trace_ref=trace_ref,
        result_ref=activity_input.result_ref if storage.exists(activity_input.result_ref) else None,
        terminal_tool_name=turn.terminal_tool_name,
        tool_calls_fired=turn.tool_calls_fired,
    )


@activity.defn
async def finalize_llm_subroutine_trace_activity(
    activity_input: LLMSubroutineTraceInput,
) -> LLMSubroutineTraceResult:
    """Combine provider calls or harness traces into the subroutine's retained conversation trace."""
    from nof1_causal_lab.llm.traces import LLMTrace, TraceUsage, _merge_trace

    root = subroutine_root(
        activity_input.subroutine.workspace_id,
        activity_input.subroutine.run_id,
        activity_input.subroutine.subroutine_id,
    )
    trace_path = storage.join(root, "trace.json")

    if activity_input.harness_trace_refs:
        trace = LLMTrace()
        for harness_trace_ref in activity_input.harness_trace_refs:
            trace = _merge_trace(
                trace, LLMTrace.model_validate(read_subroutine_json(harness_trace_ref))
            )
        write_subroutine_trace(trace_path, trace)
        return LLMSubroutineTraceResult(trace_ref=trace_path)

    conversation = read_subroutine_json(activity_input.conversation_ref, StoredConversation)
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    has_reasoning_tokens = False
    total_time = 0.0
    model = ""
    for entry in sorted(storage.listdir(activity_input.call_ref_base)):
        if not entry.endswith(".json"):
            continue
        call = OpenRouterCallResult.model_validate(read_subroutine_json(entry)["result"])
        model = call.model or model
        total_time += float(call.time or 0.0)
        usage = call.usage or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        reasoning = usage.get("reasoning_tokens")
        if reasoning is not None:
            has_reasoning_tokens = True
            reasoning_tokens += reasoning

    trace = LLMTrace(
        messages=tuple(_trace_message(message) for message in conversation["messages"]),
        model=model,
        total_time_seconds=total_time,
        usage=TraceUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens if has_reasoning_tokens else None,
        ),
    )
    write_subroutine_trace(trace_path, trace)
    return LLMSubroutineTraceResult(trace_ref=trace_path)


LLM_SUBROUTINE_ACTIVITIES = [
    start_llm_subroutine_activity,
    append_llm_user_message_activity,
    append_llm_repair_message_activity,
    execute_llm_tool_calls_activity,
    execute_harness_tool_request_activity,
    run_harness_turn_activity,
    finalize_llm_subroutine_trace_activity,
]
