"""Pi event parsing and trace assembly."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nof1_causal_lab.llm.harness.events.common import (
    SessionStreamRuntime,
    _extract_usage,
    _format_usage,
    event_object,
)
from nof1_causal_lab.llm.traces import TraceMessage, TraceToolCall, TraceUsage

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject, JsonValue


def _pi_content_text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _pi_tool_calls(content: JsonValue) -> list[TraceToolCall]:
    if not isinstance(content, list):
        return []
    calls: list[TraceToolCall] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            continue
        arguments = block.get("arguments") or block.get("input") or {}
        calls.append(
            {
                "id": str(block.get("id") or block.get("toolCallId") or ""),
                "name": str(block.get("name") or block.get("toolName") or ""),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            }
        )
    return calls


def _pi_result_text(result: JsonValue) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return json.dumps(result) if result is not None else ""
    content = result.get("content")
    text = _pi_content_text(content)
    return text if text else json.dumps(result)


def _add_usage(total: TraceUsage, usage_raw: JsonValue) -> TraceUsage:
    usage = _extract_usage(usage_raw)
    return TraceUsage(
        input_tokens=total.input_tokens + usage.input_tokens,
        output_tokens=total.output_tokens + usage.output_tokens,
        reasoning_tokens=((total.reasoning_tokens or 0) + (usage.reasoning_tokens or 0)) or None,
    )


def apply_pi_event(state: SessionStreamRuntime, event: JsonObject) -> None:
    """Fold one Pi JSON event into ``state``."""
    state.raw_events.append(event)
    etype = event.get("type")

    if etype == "session":
        session_id = event.get("id")
        if isinstance(session_id, str):
            state.session_id = session_id
        return

    if etype == "nof1.turn_timing":
        duration = event.get("duration_seconds")
        if isinstance(duration, (int, float)):
            state.total_time_seconds += float(duration)
        return

    if etype == "message_end":
        message = event_object(event.get("message"))
        role = message.get("role")
        if role == "user":
            state.messages.append(
                TraceMessage(role="user", content=_pi_content_text(message.get("content")))
            )
            return
        if role != "assistant":
            return
        content = message.get("content")
        text = _pi_content_text(content)
        tool_calls = _pi_tool_calls(content)
        state.messages.append(
            TraceMessage(
                role="assistant", content=text, tool_calls=tuple(tool_calls) if tool_calls else None
            )
        )
        if text:
            state.final_text = text
        model = message.get("model")
        if isinstance(model, str):
            state.model = model
        reason = message.get("stopReason")
        if isinstance(reason, str):
            state.stop_reason = reason
        state.usage = _add_usage(state.usage, message.get("usage"))
        return

    if etype == "tool_execution_end":
        result = _pi_result_text(event.get("result"))
        state.messages.append(
            TraceMessage(
                role="tool",
                content=result,
                tool_call_id=str(event.get("toolCallId") or ""),
                tool_result=result,
                tool_is_error=bool(event.get("isError", False)),
            )
        )


def format_pi_event_for_log(event: JsonObject) -> str | None:
    """Return a concise live-log line for a Pi JSON event."""
    etype = event.get("type")
    if etype == "session":
        return f"pi session started ({event.get('id') or '?'})"
    if etype == "tool_execution_start":
        args = event.get("args") or {}
        return f"pi tool call: {event.get('toolName') or '?'}({json.dumps(args, default=str)})"
    if etype == "tool_execution_end":
        error = " [error]" if event.get("isError") else ""
        return (
            f"pi tool result: {event.get('toolName') or '?'}{error} -> "
            f"{_pi_result_text(event.get('result'))}"
        )
    if etype == "message_end":
        message = event_object(event.get("message"))
        if message.get("role") != "assistant":
            return None
        text = _pi_content_text(message.get("content"))
        usage = _format_usage(message.get("usage"))
        details = f" [{usage}]" if usage else ""
        return f"pi message: {text}{details}" if text else None
    if etype == "agent_end":
        return "pi turn completed"
    return None
