"""Shared activities for workflow-driven LLM subroutines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from pydantic import TypeAdapter, ValidationError
from temporalio import activity
from typing_extensions import TypedDict

from nof1_causal_lab.actions.temporal.llm.context import subroutine_context_messages
from nof1_causal_lab.actions.temporal.llm.messages import (
    AppendLLMRepairMessageInput,
    AppendLLMRepairMessageResult,
    AppendLLMUserMessageInput,
    AppendLLMUserMessageResult,
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


@activity.defn
async def finalize_llm_subroutine_trace_activity(
    activity_input: LLMSubroutineTraceInput,
) -> LLMSubroutineTraceResult:
    """Combine provider calls into the subroutine's retained conversation trace."""
    from nof1_causal_lab.llm.traces import LLMTrace, TraceUsage

    root = subroutine_root(
        activity_input.subroutine.workspace_id,
        activity_input.subroutine.run_id,
        activity_input.subroutine.subroutine_id,
    )
    trace_path = storage.join(root, "trace.json")
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
    finalize_llm_subroutine_trace_activity,
]
