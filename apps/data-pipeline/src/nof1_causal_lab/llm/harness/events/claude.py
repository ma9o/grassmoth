"""Claude event parsing and trace assembly."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nof1_causal_lab.llm.harness.events.common import (
    SessionStreamRuntime,
    _extract_usage,
    _format_usage,
    _log_text,
    event_object,
)
from nof1_causal_lab.llm.traces import TraceMessage, TraceToolCall, TraceUsage

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject, JsonValue


def _coerce_content_text(content: JsonValue) -> str:
    """Flatten Anthropic ``content`` blocks into a plain text string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _claude_assistant_message(message: JsonObject) -> TraceMessage:
    """Build a TraceMessage from a Claude assistant stream-json message."""
    content = message.get("content", [])
    text = _coerce_content_text(content)
    tool_calls: list[TraceToolCall] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}),
                    }
                )
    return TraceMessage(
        role="assistant",
        content=text,
        tool_calls=tuple(tool_calls) if tool_calls else None,
    )


def _claude_tool_result_messages(message: JsonObject) -> list[TraceMessage]:
    """Extract tool-result blocks from a Claude user stream-json message."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    out: list[TraceMessage] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        result_content = block.get("content")
        if isinstance(result_content, list):
            text = _coerce_content_text(result_content)
        elif isinstance(result_content, str):
            text = result_content
        else:
            text = json.dumps(result_content) if result_content is not None else ""
        out.append(
            TraceMessage(
                role="tool",
                content=text,
                tool_call_id=str(block.get("tool_use_id", "")),
                tool_result=text,
                tool_is_error=bool(block.get("is_error", False)),
            )
        )
    return out


def _claude_user_prompt_message(message: JsonObject) -> TraceMessage | None:
    """Handle a pure user prompt (string content, not tool_result blocks)."""
    content = message.get("content")
    if isinstance(content, str):
        return TraceMessage(role="user", content=content)
    if isinstance(content, list):
        # If no tool_result blocks present, treat remaining text as a user prompt.
        has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
        if not has_tool_result:
            return TraceMessage(role="user", content=_coerce_content_text(content))
    return None


def apply_claude_event(state: SessionStreamRuntime, event: JsonObject) -> None:
    """Fold one Claude stream-json event into the accumulator."""
    state.raw_events.append(event)
    etype = event.get("type")

    if etype == "system" and event.get("subtype") == "init":
        session_id = event.get("session_id")
        model = event.get("model")
        if isinstance(session_id, str):
            state.session_id = session_id
        if isinstance(model, str):
            state.model = model
        return

    if etype == "user":
        message = event_object(event.get("message"))
        prompt = _claude_user_prompt_message(message)
        if prompt is not None:
            state.messages.append(prompt)
        state.messages.extend(_claude_tool_result_messages(message))
        return

    if etype == "assistant":
        message = event_object(event.get("message"))
        state.messages.append(_claude_assistant_message(message))
        usage = _extract_usage(message.get("usage"))
        state.usage = TraceUsage(
            input_tokens=state.usage.input_tokens + usage.input_tokens,
            output_tokens=state.usage.output_tokens + usage.output_tokens,
            reasoning_tokens=((state.usage.reasoning_tokens or 0) + (usage.reasoning_tokens or 0))
            or None,
        )
        stop_reason = message.get("stop_reason")
        if isinstance(stop_reason, str):
            state.stop_reason = stop_reason
        return

    if etype == "result":
        result_text = event.get("result")
        if isinstance(result_text, str):
            state.final_text = result_text
        duration_ms = event.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            state.total_time_seconds = float(duration_ms) / 1000.0
        usage = _extract_usage(event.get("usage"))
        if usage.input_tokens or usage.output_tokens:
            state.usage = usage
        subtype = event.get("subtype")
        if isinstance(subtype, str):
            state.stop_reason = state.stop_reason or subtype
        return


def format_claude_event_for_log(event: JsonObject) -> str | None:
    """Return a single human-readable line for one claude-code event."""
    etype = event.get("type")
    if etype == "system" and event.get("subtype") == "init":
        model = event.get("model") or "?"
        session = event.get("session_id") or "?"
        return f"claude session started ({model}, {session})"
    if etype == "assistant":
        message = event_object(event.get("message"))
        content = message.get("content") or []
        lines: list[str] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        lines.append(f"message: {_log_text(text)}")
                elif btype == "tool_use":
                    name = block.get("name") or "?"
                    args = block.get("input") or {}
                    args_preview = (
                        json.dumps(args, default=str) if isinstance(args, dict) else str(args)
                    )
                    lines.append(f"tool call: {name}({_log_text(args_preview)})")
                elif btype == "thinking":
                    thinking = block.get("thinking") or block.get("text") or ""
                    lines.append(f"reasoning: {_log_text(thinking)}")
        usage_str = _format_usage(message.get("usage"))
        if usage_str:
            lines.append(f"[{usage_str}]")
        if not lines:
            return None
        return "claude " + " | ".join(lines)
    if etype == "user":
        message = event_object(event.get("message"))
        content = message.get("content") or []
        previews: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = block.get("content") or ""
                    if isinstance(output, list):
                        output = _coerce_content_text(output)
                    err = " [error]" if block.get("is_error") else ""
                    previews.append(f"tool result{err} -> {_log_text(output)}")
        if not previews:
            return None
        return "claude " + " | ".join(previews)
    if etype == "result":
        usage_str = _format_usage(event.get("usage"))
        duration_ms = event.get("duration_ms")
        duration = (
            f" in {float(duration_ms) / 1000.0:.1f}s"
            if isinstance(duration_ms, (int, float))
            else ""
        )
        subtype = event.get("subtype") or ""
        subtype_str = f" ({subtype})" if subtype else ""
        usage_str = f" [{usage_str}]" if usage_str else ""
        return f"claude turn completed{subtype_str}{duration}{usage_str}"
    return None
