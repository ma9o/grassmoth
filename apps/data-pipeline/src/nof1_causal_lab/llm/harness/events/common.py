"""Common event parsing and trace assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import TypeAdapter

from nof1_causal_lab.json_types import JsonObject, JsonValue
from nof1_causal_lab.llm.traces import LLMTrace, TraceMessage, TraceUsage

_EVENT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def parse_stream_event(event: str | bytes | JsonObject) -> JsonObject:
    """Decode a foreign event once, at its transport boundary."""
    return (
        _EVENT_ADAPTER.validate_json(event)
        if isinstance(event, (str, bytes))
        else _EVENT_ADAPTER.validate_python(event)
    )


def event_object(value: JsonValue) -> JsonObject:
    """Read an optional object in a provider event; reject malformed field shapes."""
    if value is None:
        return dict[str, JsonValue]()
    if not isinstance(value, Mapping):
        raise ValueError("Expected an object in a harness event")
    fields: JsonObject = value
    return fields


class _TraceAccumulator(Protocol):
    messages: list[TraceMessage]
    total_time_seconds: float
    usage: TraceUsage


def _materialize_trace(state: _TraceAccumulator, model: str) -> LLMTrace:
    return LLMTrace(
        messages=tuple(state.messages),
        model=model,
        total_time_seconds=state.total_time_seconds,
        usage=state.usage,
    )


@dataclass
class SessionStreamRuntime:
    """Accumulator for Claude Code and Pi session event streams."""

    session_id: str | None = None
    model: str = ""
    messages: list[TraceMessage] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)
    total_time_seconds: float = 0.0
    stop_reason: str | None = None
    final_text: str = ""
    raw_events: list[JsonObject] = field(default_factory=list)


def _extract_usage(usage_raw: JsonValue) -> TraceUsage:
    if not isinstance(usage_raw, dict):
        return TraceUsage()
    return TraceUsage.model_validate(
        {
            "input_tokens": usage_raw.get("input_tokens")
            or usage_raw.get("prompt_tokens")
            or usage_raw.get("input")
            or 0,
            "output_tokens": usage_raw.get("output_tokens")
            or usage_raw.get("completion_tokens")
            or usage_raw.get("output")
            or 0,
            "reasoning_tokens": usage_raw.get("reasoning_tokens")
            or usage_raw.get("reasoning")
            or event_object(usage_raw.get("completion_tokens_details")).get("reasoning_tokens"),
        }
    )


def _log_text(text: JsonValue) -> str:
    """Render a streamed value for live logging without truncation."""
    if isinstance(text, str):
        return text
    return "" if text is None else str(text)


def _format_usage(usage: JsonValue) -> str:
    """Render the ``usage`` dict as ``in=.. out=.. reasoning=..`` for log lines."""
    parsed = _extract_usage(usage)
    parts: list[str] = []
    if parsed.input_tokens:
        parts.append(f"in={parsed.input_tokens}")
    if parsed.output_tokens:
        parts.append(f"out={parsed.output_tokens}")
    if parsed.reasoning_tokens:
        parts.append(f"reasoning={parsed.reasoning_tokens}")
    return " ".join(parts)


def finalize_trace(state: SessionStreamRuntime) -> LLMTrace:
    """Materialize an :class:`LLMTrace` from an accumulator."""
    return _materialize_trace(state, state.model)
