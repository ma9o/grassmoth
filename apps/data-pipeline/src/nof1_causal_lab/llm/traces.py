"""Trace contracts and small shared LLM boundary helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import Field
from typing_extensions import TypedDict

from nof1_causal_lab.artifacts.base import Value

if TYPE_CHECKING:
    from collections.abc import Sequence


MAX_TOOL_REPAIR_ERROR_CHARS = 1200


class NamedTool(Protocol):
    """Minimal tool surface needed when constructing retry guidance."""

    @property
    def name(self) -> str:
        """Name exposed to the model for selecting this tool."""
        ...


class TraceToolCall(TypedDict):
    """A function invocation with the call identity used to match its result."""

    id: str
    name: str
    arguments: str


class TraceMessage(Value):
    """One retained conversation step, including any reasoning or tool interaction."""

    role: str
    content: str
    reasoning: str | None = None
    tool_calls: tuple[TraceToolCall, ...] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result: str | None = None
    tool_is_error: bool = False


class TraceUsage(Value):
    """Trace usage records the input, output, and reasoning tokens consumed by a conversation."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None


class LLMTrace(Value):
    """An LLM trace records a conversation, its model, elapsed time, and token usage."""

    messages: tuple[TraceMessage, ...] = Field(default_factory=tuple)
    model: str = ""
    total_time_seconds: float = 0.0
    usage: TraceUsage = Field(default_factory=TraceUsage)


def _merge_trace(existing: LLMTrace, new_trace: LLMTrace) -> LLMTrace:
    """Append a trace segment onto an existing stage-local trace."""
    return LLMTrace(
        messages=(*existing.messages, *new_trace.messages),
        model=new_trace.model or existing.model,
        total_time_seconds=existing.total_time_seconds + new_trace.total_time_seconds,
        usage=TraceUsage(
            input_tokens=existing.usage.input_tokens + new_trace.usage.input_tokens,
            output_tokens=existing.usage.output_tokens + new_trace.usage.output_tokens,
            reasoning_tokens=(
                (existing.usage.reasoning_tokens or 0) + (new_trace.usage.reasoning_tokens or 0)
            )
            or None,
        ),
    )


def _truncate_tool_error(error_text: str, limit: int = MAX_TOOL_REPAIR_ERROR_CHARS) -> str:
    """Trim provider error payloads before echoing them back to the model."""
    if len(error_text) <= limit:
        return error_text
    return error_text[:limit] + "\n...[truncated]"


def _tool_retry_message(error_text: str, tools: Sequence[NamedTool] | None) -> dict[str, str]:
    """Build a repair instruction after a malformed or failed tool response."""
    guidance = (
        "Retry the same step. If you need a tool, emit a valid tool call with a JSON object "
        f"for arguments and use only these tools: {', '.join(tool.name for tool in tools)}."
        if tools
        else "Retry the same step in plain text only. No tools are available on this turn, "
        "so do not emit any tool calls."
    )
    return {
        "role": "user",
        "content": (
            "Your previous response could not be processed.\n\n"
            "Error:\n"
            f"{_truncate_tool_error(error_text)}\n\n"
            f"{guidance}"
        ),
    }
