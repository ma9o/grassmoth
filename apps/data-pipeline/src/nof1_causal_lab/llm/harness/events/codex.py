"""Codex event parsing and trace assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nof1_causal_lab.llm.traces import LLMTrace, TraceMessage, TraceUsage

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageFunctionToolCallParam

    from nof1_causal_lab.json_types import JsonObject


from nof1_causal_lab.llm.harness.events.claude import _coerce_content_text
from nof1_causal_lab.llm.harness.events.common import (
    _extract_usage,
    _format_usage,
    _log_text,
    _materialize_trace,
    event_object,
)


def format_codex_event_for_log(event: JsonObject) -> str | None:
    """Return a single human-readable line for one codex ``--json`` event.

    Returns ``None`` for events that are not useful to surface live —
    mostly low-level deltas and bookkeeping. The string is meant to be
    emitted at ``INFO`` so it shows up in Prefect's flow-run logs next to
    the surrounding stage messages.
    """
    etype = event.get("type")
    if etype == "thread.started":
        tid = event.get("thread_id") or "?"
        return f"codex thread started ({tid})"
    if etype == "item.started":
        item = event_object(event.get("item"))
        item_type = item.get("type") or item.get("item_type") or "item"
        # Surface shell command starts so live logs show *what* is being run
        # before the (often long) execution completes. Other item starts
        # (reasoning, messages) are too noisy to log at info level.
        if item_type == "command_execution":
            command = item.get("command") or item.get("cmd") or ""
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            return f"codex shell start: {_log_text(command)}"
        return None
    if etype == "item.completed":
        item = event_object(event.get("item"))
        item_type = item.get("type") or item.get("item_type") or "item"
        if item_type == "reasoning":
            summary = item.get("summary") or item.get("text") or item.get("content")
            if isinstance(summary, list):
                summary = " ".join(str(part) for part in summary if part)
            preview = _log_text(summary)
            return f"codex reasoning: {preview}" if preview else "codex reasoning (empty)"
        if item_type in {"agent_message", "message"}:
            text = item.get("text")
            if not isinstance(text, str):
                text = _coerce_content_text(item.get("content"))
            return f"codex message: {_log_text(text)}"
        if item_type in {"tool_call", "mcp_tool_call"}:
            name = item.get("name") or item.get("tool") or "?"
            args = item.get("arguments") or item.get("input") or {}
            if isinstance(args, dict):
                args_preview = json.dumps(args, default=str)
            else:
                args_preview = str(args)
            status = item.get("status") or ""
            output = item.get("output") or item.get("result") or item.get("text") or ""
            error = item.get("error") or (item.get("is_error") and output) or ""
            status_str = f" [{status}]" if status else ""
            if output and not isinstance(output, str):
                output = json.dumps(output, default=str)
            if error:
                error_str = json.dumps(error, default=str) if not isinstance(error, str) else error
                return (
                    f"codex tool call: {name}{status_str}({_log_text(args_preview)}) "
                    f"→ error: {_log_text(error_str)}"
                )
            output_str = f" → {_log_text(output)}" if output else ""
            return f"codex tool call: {name}{status_str}({_log_text(args_preview)}){output_str}"
        if item_type in {"tool_result", "mcp_tool_result"}:
            name = item.get("name") or item.get("tool") or "?"
            output = item.get("output") or item.get("result") or item.get("text") or ""
            if not isinstance(output, str):
                output = json.dumps(output, default=str)
            err = " [error]" if item.get("is_error") else ""
            return f"codex tool result: {name}{err} -> {_log_text(output)}"
        if item_type == "command_execution":
            command = item.get("command") or item.get("cmd") or ""
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            status = item.get("status") or ""
            exit_code = item.get("exit_code")
            output = (
                item.get("aggregated_output")
                or item.get("output")
                or item.get("formatted_output")
                or item.get("stdout")
                or ""
            )
            stderr = item.get("stderr") or ""
            if not isinstance(output, str):
                output = json.dumps(output, default=str)
            if not isinstance(stderr, str):
                stderr = json.dumps(stderr, default=str)
            header = f"codex shell: {_log_text(command)}"
            suffix: list[str] = []
            if status:
                suffix.append(f"status={status}")
            if exit_code is not None:
                suffix.append(f"exit={exit_code}")
            suffix_str = f" [{' '.join(suffix)}]" if suffix else ""
            tail: list[str] = []
            if output:
                tail.append(f"out: {_log_text(output)}")
            if stderr:
                tail.append(f"err: {_log_text(stderr)}")
            tail_str = f" → {' | '.join(tail)}" if tail else ""
            return f"{header}{suffix_str}{tail_str}"
        # Unknown item type: dump the item as JSON so we can see its shape
        # the next time the formatter falls through here instead of silently
        # emitting a detail-free "codex {item_type}" line.
        payload = json.dumps(item, default=str)
        return f"codex {item_type}: {_log_text(payload)}"
    if etype in {"tool_call", "mcp_tool_call"}:
        name = event.get("name") or event.get("tool") or "?"
        args = event.get("arguments") or event.get("input") or {}
        if isinstance(args, dict):
            args_preview = json.dumps(args, default=str)
        else:
            args_preview = str(args)
        return f"codex tool call: {name}({_log_text(args_preview)})"
    if etype in {"tool_result", "mcp_tool_result"}:
        name = event.get("name") or event.get("tool") or "?"
        output = event.get("output") or event.get("result") or event.get("text") or ""
        if not isinstance(output, str):
            output = json.dumps(output, default=str)
        err = " [error]" if event.get("is_error") else ""
        return f"codex tool result: {name}{err} -> {_log_text(output)}"
    if etype in {"agent_message", "message"}:
        message = event_object(event["message"]) if event.get("message") else event
        text = message.get("text")
        if not isinstance(text, str):
            text = _coerce_content_text(message.get("content"))
        return f"codex message: {_log_text(text)}"
    if etype in {"thread.completed", "turn.completed", "done", "result"}:
        usage_str = _format_usage(event.get("usage"))
        duration_ms = event.get("duration_ms")
        duration = (
            f" in {float(duration_ms) / 1000.0:.1f}s"
            if isinstance(duration_ms, (int, float))
            else ""
        )
        reason = event.get("stop_reason") or event.get("subtype") or ""
        reason_str = f" ({reason})" if reason else ""
        usage_str = f" [{usage_str}]" if usage_str else ""
        return f"codex turn completed{reason_str}{duration}{usage_str}"
    if etype in {"turn.failed", "error"}:
        err = event.get("error") or event.get("message") or ""
        if isinstance(err, dict):
            err = json.dumps(err, default=str)
        return f"codex error: {_log_text(err)}"
    return None


@dataclass
class CodexStreamState:
    """Accumulator for :func:`parse_codex_stream`."""

    thread_id: str | None = None
    messages: list[TraceMessage] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)
    total_time_seconds: float = 0.0
    stop_reason: str | None = None
    final_text: str = ""
    raw_events: list[JsonObject] = field(default_factory=list)
    _open_tool_calls: dict[str, ChatCompletionMessageFunctionToolCallParam] = field(
        default_factory=dict
    )


def apply_codex_event(state: CodexStreamState, event: JsonObject) -> None:
    """Fold one Codex ``--json`` event into the accumulator.

    Codex's event schema is less stable than Claude's; this handler
    extracts the commonly-seen shapes (thread start, agent messages,
    tool calls) and records everything in ``raw_events`` so downstream
    code can reach back to anything we don't yet model.
    """
    state.raw_events.append(event)
    etype = event.get("type")

    if etype == "thread.started":
        tid = event.get("thread_id")
        if isinstance(tid, str):
            state.thread_id = tid
        return

    # Codex 0.121 emits `item.completed` with a nested ``item`` carrying
    # the actual type (``agent_message``, ``reasoning``, ``mcp_tool_call``,
    # etc.). Earlier/prototype schemas used top-level ``agent_message``
    # events; both shapes are handled.
    if etype == "item.completed":
        item = event_object(event.get("item"))
        apply_codex_event(state, {**item, "_from_item_completed": True})
        return

    if etype in {"agent_message", "message"}:
        # The item may or may not carry an explicit role; codex's
        # agent_message items are always assistant output.
        message = event_object(event["message"]) if event.get("message") else event
        role = message.get("role") or "assistant"
        text = message.get("text")
        if not isinstance(text, str):
            content = message.get("content") or ""
            text = _coerce_content_text(content) if isinstance(content, list) else str(content)
        if role == "assistant":
            state.messages.append(TraceMessage(role="assistant", content=text))
            if text:
                state.final_text = text
        elif role == "user":
            state.messages.append(TraceMessage(role="user", content=text))
        return

    if etype in {"tool_call", "mcp_tool_call"}:
        call_id = str(event.get("call_id") or event.get("id") or "")
        tool_name = str(event.get("name") or event.get("tool") or "")
        arguments = event.get("arguments") or event.get("input") or {}
        if isinstance(arguments, dict):
            arguments_json = json.dumps(arguments)
        else:
            arguments_json = str(arguments)
        tool_call_entry: ChatCompletionMessageFunctionToolCallParam = {
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": arguments_json},
        }
        state.messages.append(
            TraceMessage(
                role="assistant",
                content="",
                tool_calls=(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": arguments_json,
                    },
                ),
            )
        )
        state._open_tool_calls[call_id] = tool_call_entry
        return

    if etype in {"tool_result", "mcp_tool_result"}:
        call_id = str(event.get("call_id") or event.get("id") or "")
        result = event.get("output") or event.get("result") or event.get("text") or ""
        if not isinstance(result, str):
            result = json.dumps(result)
        state.messages.append(
            TraceMessage(
                role="tool",
                content=result,
                tool_call_id=call_id,
                tool_result=result,
                tool_is_error=bool(event.get("is_error", False)),
            )
        )
        state._open_tool_calls.pop(call_id, None)
        return

    if etype in {"thread.completed", "turn.completed", "done", "result"}:
        usage = _extract_usage(event.get("usage"))
        if usage.input_tokens or usage.output_tokens:
            state.usage = usage
        duration_ms = event.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            state.total_time_seconds = float(duration_ms) / 1000.0
        reason = event.get("stop_reason") or event.get("subtype")
        if isinstance(reason, str):
            state.stop_reason = reason
        return

    if etype in {"turn.failed", "error"}:
        err = event.get("error") or event.get("message") or ""
        if isinstance(err, dict):
            err = json.dumps(err)
        state.stop_reason = "error"
        state.final_text = state.final_text or str(err)
        return


def finalize_codex_trace(state: CodexStreamState, model: str) -> LLMTrace:
    """Materialize an :class:`LLMTrace` from a Codex accumulator."""
    return _materialize_trace(state, model)
