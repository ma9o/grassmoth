"""Minimal OpenRouter runtime helpers.

Runtime orchestration uses plain OpenAI-style message dicts and a small local
tool abstraction. This module owns the transport to OpenRouter directly.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol, TypedDict

from openai import NOT_GIVEN, AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionTokensDetails
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from nof1_causal_lab.public import JsonObject, JsonValue
from nof1_causal_service.config import get_secret

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from pydantic.json_schema import JsonSchemaValue

    from nof1_causal_service.llm.specs import EmbeddedReasoningEffort


logger = logging.getLogger(__name__)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_PREFIX = "openrouter/"


# ---------------------------------------------------------------------------
# RPM (requests-per-minute) rate limiter
# ---------------------------------------------------------------------------


class RpmLimiter:
    """Thread-safe request limiter using an asynchronous sliding-window wait.

    Each acquired slot records a monotonic timestamp. Callers wait asynchronously
    while the window is full, and a lock protects shared state across threads.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        """Initialize a request budget over a rolling time window.

        Args:
            max_requests: Maximum number of acquisitions allowed in one window.
            window_seconds: Window duration in seconds; sixty implements an RPM limit.
        """
        self.max_requests = max_requests
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._timestamps and self._timestamps[0] <= now - self._window:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        """Wait until a request slot is available within the window."""
        while True:
            with self._lock:
                now = monotonic()
                self._purge(now)
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return
                # Calculate how long until the oldest entry expires
                wait_for = self._timestamps[0] + self._window - now
            await asyncio.sleep(min(wait_for + 0.05, 1.0))


_limiters: dict[str, RpmLimiter] = {}


async def acquire_limiter(name: str) -> None:
    """Acquire a slot from the named limiter (no-op if not registered)."""
    limiter = _limiters.get(name)
    if limiter is not None:
        await limiter.acquire()


def create_openrouter_client() -> AsyncOpenAI:
    """Create the client whose connection pool is owned by its worker's lifetime."""
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        # The SDK requires a string up front; missing credentials still
        # surface as a normal authentication error on the first request.
        api_key=get_secret("OPENROUTER_API_KEY") or "missing",
    )


def normalize_openrouter_model_name(model_name: str) -> str:
    """Translate repo-local model IDs to the upstream OpenRouter format."""
    normalized = model_name.strip()
    if normalized.startswith(OPENROUTER_MODEL_PREFIX):
        return normalized[len(OPENROUTER_MODEL_PREFIX) :]
    return normalized


@dataclass(frozen=True)
class GenerateConfig:
    """Generation settings shared across model calls."""

    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: EmbeddedReasoningEffort | None = None


@dataclass
class Tool:
    """Callable tool with model-facing JSON schema."""

    name: str
    description: str
    parameters: JsonSchemaValue
    execute: Callable[..., Awaitable[str]]

    async def __call__(self, *args: object, **kwargs: object) -> str:
        """Await the underlying tool implementation with the supplied positional and keyword arguments."""
        return await self.execute(*args, **kwargs)


class _ResponseValue(BaseModel):
    """Validate the OpenRouter transport once, including its SDK extension fields."""

    model_config = ConfigDict(from_attributes=True)


class _ContentPart(_ResponseValue):
    type: str
    text: str | None = None
    reasoning: str | None = None


class _FunctionCall(_ResponseValue):
    name: str
    arguments: str | JsonObject


class _ToolCall(_ResponseValue):
    id: str
    function: _FunctionCall


class _AssistantResponse(_ResponseValue):
    content: str | list[_ContentPart] | None = None
    tool_calls: list[_ToolCall] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: JsonValue = None


class _ResponseChoice(_ResponseValue):
    message: _AssistantResponse
    finish_reason: str | None = None


class _ResponseUsage(_ResponseValue):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    completion_tokens_details: CompletionTokensDetails | None = None
    reasoning_tokens: int | None = None


class _OpenRouterResponse(_ResponseValue):
    model: str
    choices: list[_ResponseChoice]
    usage: _ResponseUsage | None = None


class AssistantMessage(TypedDict):
    """Normalized assistant message passed to logs and the next conversation turn."""

    role: Literal["assistant"]  # noqa: FIELD002 -- OpenAI's assistant-message wire protocol requires this role.
    content: str
    tool_calls: NotRequired[list[ChatCompletionMessageFunctionToolCallParam]]
    reasoning: NotRequired[str]
    reasoning_details: NotRequired[JsonValue]


def _parse_arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Extract argument descriptions from a Google-style docstring."""
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    in_args = False
    current_name: str | None = None

    for raw_line in inspect.cleandoc(docstring).splitlines():
        stripped = raw_line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if not stripped:
            current_name = None
            continue
        if not raw_line.startswith(" "):
            break

        name, _, description = stripped.partition(":")
        if _ and name.replace("_", "").isalnum():
            descriptions[name.strip()] = description.strip()
            current_name = name.strip()
            continue
        if current_name is not None:
            descriptions[current_name] = f"{descriptions[current_name]} {stripped}".strip()

    return descriptions


def _parameter_schema(handler: Callable[..., Awaitable[str]], name: str) -> JsonSchemaValue:
    """Build a JSON schema from a tool handler signature."""
    signature = inspect.signature(handler)
    descriptions = _parse_arg_descriptions(inspect.getdoc(handler))
    fields: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny] -- create_model types field definitions as Any keyword arguments beside its own __config__/__base__ options.

    for parameter_name, param in signature.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Signature.empty else Any
        default = param.default if param.default is not inspect.Signature.empty else ...
        if default is ...:
            field = Field(..., description=descriptions.get(parameter_name))
        else:
            field = Field(default, description=descriptions.get(parameter_name))
        fields[parameter_name] = (annotation, field)

    if not fields:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    model = create_model(
        f"{name.title()}ToolParams",
        **fields,
    )
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return schema


class ToolFactory[**P](Protocol):
    """Named factory that binds tool configuration into an asynchronous text-returning callable."""

    @property
    def __name__(self) -> str:
        """Factory name used when deriving the exposed tool name."""
        ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Callable[..., Awaitable[str]]:
        """Bind factory arguments and return the asynchronous tool implementation."""
        ...


def tool[**P](factory: ToolFactory[P]) -> Callable[P, Tool]:
    """Decorator that converts a tool factory into a Tool object factory."""

    @wraps(factory)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Tool:
        handler = factory(*args, **kwargs)
        return Tool(
            name=factory.__name__,
            description=inspect.getdoc(factory) or "",
            parameters=_parameter_schema(handler, factory.__name__),
            execute=handler,
        )

    return wrapper


class _ReasoningFields(BaseModel):
    reasoning: str | None = None
    reasoning_details: JsonValue = None


class ReasoningAssistantMessage(ChatCompletionAssistantMessageParam):
    """Provider assistant message extended with optional reasoning text and structured details."""

    reasoning: NotRequired[str]
    reasoning_details: NotRequired[JsonValue]


_MESSAGE_ADAPTER: TypeAdapter[ChatCompletionMessageParam] = TypeAdapter(ChatCompletionMessageParam)


def normalize_message(message: object) -> ChatCompletionMessageParam:
    """Decode a stored chat message, including OpenRouter's reasoning extension."""
    normalized = _MESSAGE_ADAPTER.validate_python(message)
    if normalized["role"] == "assistant":
        reasoning = _ReasoningFields.model_validate(message)
        extended: ReasoningAssistantMessage = {**normalized}
        if reasoning.reasoning is not None:
            extended["reasoning"] = reasoning.reasoning
        if reasoning.reasoning_details is not None:
            extended["reasoning_details"] = reasoning.reasoning_details
        return extended
    return normalized


def _tool_schema(tool_obj: Tool) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters,
        },
    }


def _message_content_parts(content: str | list[_ContentPart] | None) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content, None
    if content is None:
        return "", None

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content:
        part_type = part.type
        text = part.text
        reasoning = part.reasoning
        if part_type in {"text", "output_text"} and text:
            text_parts.append(str(text))
        elif part_type in {"reasoning", "thinking"} and reasoning:
            reasoning_parts.append(str(reasoning))
        elif text:
            text_parts.append(str(text))

    joined_reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return "\n".join(text_parts), joined_reasoning


def _assistant_message(message: _AssistantResponse) -> AssistantMessage:
    content_text, content_reasoning = _message_content_parts(message.content)
    assistant_message: AssistantMessage = {
        "role": "assistant",
        "content": content_text,
    }

    tool_calls_raw = message.tool_calls or []
    tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []
    for tool_call in tool_calls_raw:
        function = tool_call.function
        arguments = function.arguments
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": function.name,
                    "arguments": arguments or "{}",
                },
            }
        )
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    reasoning = message.reasoning
    if reasoning is None:
        reasoning = message.reasoning_content
    if reasoning is None:
        reasoning = content_reasoning
    if reasoning:
        assistant_message["reasoning"] = reasoning

    reasoning_details = message.reasoning_details
    if reasoning_details is not None:
        assistant_message["reasoning_details"] = reasoning_details

    return assistant_message


def _usage_from_response(response: _OpenRouterResponse) -> dict[str, int | None] | None:
    usage = response.usage
    if usage is None:
        return None

    details = usage.completion_tokens_details
    reasoning_tokens = details.reasoning_tokens if details is not None else None
    if reasoning_tokens is None:
        reasoning_tokens = usage.reasoning_tokens

    return {
        "input_tokens": usage.prompt_tokens or 0,
        "output_tokens": usage.completion_tokens or 0,
        "reasoning_tokens": reasoning_tokens,
    }


def _log_response_details(
    *,
    log_label: str | None,
    message: AssistantMessage,
    completion_text: str,
) -> None:
    """Log raw assistant outputs (completion, tool calls, reasoning)."""
    prefix = f"[{log_label}] " if log_label else ""

    logger.info(
        "%scall_model completion:\n%s",
        prefix,
        completion_text or "<empty>",
    )

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        logger.info(
            "%scall_model tool_calls:\n%s",
            prefix,
            json.dumps(tool_calls, indent=2, sort_keys=True),
        )

    reasoning = message.get("reasoning")
    if reasoning:
        logger.info(
            "%scall_model reasoning:\n%s",
            prefix,
            reasoning,
        )
    reasoning_details = message.get("reasoning_details")
    if reasoning_details is not None:
        logger.info(
            "%scall_model reasoning_details:\n%s",
            prefix,
            json.dumps(reasoning_details, indent=2, sort_keys=True, default=str),
        )


class ModelCallResult(TypedDict):
    """One provider response with its assistant message, usage, model, timing, and stop reason."""

    message: AssistantMessage
    usage: dict[str, int | None] | None
    model: str
    time: float
    stop_reason: str | None


async def call_model(
    model_name: str,
    messages: Sequence[object],
    *,
    client: AsyncOpenAI,
    tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    log_label: str | None = None,
) -> ModelCallResult:
    """Call OpenRouter and normalize the first choice into a plain dict."""
    request = config or GenerateConfig()
    normalized_model_name = normalize_openrouter_model_name(model_name)

    await acquire_limiter("llm")

    extra_body: JsonObject = {
        "provider": {
            "sort": "throughput",
        }
    }
    if request.reasoning_effort is not None:
        extra_body["reasoning"] = {
            "effort": request.reasoning_effort,
        }

    if log_label:
        logger.info(
            "[%s] call_model request: model=%s messages=%d tools=%d timeout=%s max_tokens=%s",
            log_label,
            normalized_model_name,
            len(messages),
            len(tools or []),
            request.timeout,
            request.max_tokens,
        )

    started_at = perf_counter()
    request_coro = client.chat.completions.create(
        model=normalized_model_name,
        messages=[normalize_message(message) for message in messages],
        max_tokens=request.max_tokens if request.max_tokens is not None else omit,
        timeout=request.timeout if request.timeout is not None else NOT_GIVEN,
        extra_body=extra_body,
        tools=[_tool_schema(tool_obj) for tool_obj in tools] if tools else omit,
    )
    if request.timeout is not None:
        response = await asyncio.wait_for(request_coro, timeout=request.timeout)
    else:
        response = await request_coro
    elapsed = perf_counter() - started_at

    parsed = _OpenRouterResponse.model_validate(response)
    choices = parsed.choices
    if not choices:
        raise ValueError("OpenRouter returned no choices")

    choice = choices[0]
    message = _assistant_message(choice.message)
    completion_text = message["content"]
    tool_call_count = len(message.get("tool_calls") or [])
    stop_reason = choice.finish_reason

    if log_label:
        logger.info(
            "[%s] call_model response: model=%s stop=%s time=%.1fs tool_calls=%d completion_chars=%d",
            log_label,
            parsed.model,
            stop_reason or "end_turn",
            elapsed,
            tool_call_count,
            len(completion_text),
        )
    _log_response_details(
        log_label=log_label,
        message=message,
        completion_text=completion_text,
    )

    return {
        "message": message,
        "usage": _usage_from_response(parsed),
        "model": parsed.model,
        "time": elapsed,
        "stop_reason": stop_reason,
    }
