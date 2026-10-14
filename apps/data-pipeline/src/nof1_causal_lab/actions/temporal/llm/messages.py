"""LLM workflow requests, tool invocations, and conversation records."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.llm.specs import EmbeddedLLMSpec, HarnessLLMSpec, LLMProfileSpec


class LLMToolSpec(BaseModel):
    """An LLM-visible tool schema and its role in completing a subroutine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: JsonObject
    kind: Literal["read_only", "checkpoint", "terminal"] = "terminal"


class LLMSubroutineRef(BaseModel):
    """Execution identity and stored context shared by a subroutine's messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_ref: str


class LLMSubroutineInput(BaseModel):
    """Subroutine context, model profile, and tool-turn budget for an LLM workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    llm: LLMProfileSpec
    max_tool_turns: int


class LLMSubroutineStart(BaseModel):
    """Initial conversation and storage locations allocated for one LLM subroutine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    conversation_ref_base: str
    user_message_count: int
    tools: list[LLMToolSpec] = Field(default_factory=list)
    call_ref_base: str
    assistant_ref_base: str
    tool_execution_ref_base: str
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref_base: str


class AppendLLMUserMessageInput(BaseModel):
    """A context prompt index and the conversation to which it should be appended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    user_message_index: int


class AppendLLMUserMessageResult(BaseModel):
    """Reference to the persisted conversation after adding a user prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class AppendLLMRepairMessageInput(BaseModel):
    """Tool failure feedback and the destination for a repaired conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    next_conversation_ref: str
    error_text: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class AppendLLMRepairMessageResult(BaseModel):
    """Reference to the persisted conversation containing repair feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class LLMToolExecutionInput(BaseModel):
    """Stored assistant response, allowed tools, and destinations for execution results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    assistant_ref: str
    execution_ref: str
    result_ref: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class LLMToolExecutionResult(BaseModel):
    """Conversation after tool execution and any result accepted by a terminal tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    terminal_success: bool
    result_ref: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class HarnessTurnInput(BaseModel):
    """Harness session inputs and Temporal routing identities for one model turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    workflow_run_id: str
    subroutine: LLMSubroutineRef
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref: str
    llm: HarnessLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    user_message_index: int
    log_label: str


class HarnessToolRequest(BaseModel):
    """Harness tool invocation with the request and response paths used for rendezvous."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    subroutine: LLMSubroutineRef
    result_ref: str
    tool: LLMToolSpec
    tool_name: str
    arguments: JsonObject
    request_ref: str
    response_ref: str


class HarnessToolExecutionResult(BaseModel):
    """Correlated harness tool response and any successfully retained scientific output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tool_name: str
    output: str
    result_ref: str | None = None
    success: bool = False


class HarnessTurnResult(BaseModel):
    """Saved harness session and trace, with any terminal output produced by the turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_state_ref: str
    trace_ref: str
    result_ref: str | None = None
    terminal_tool_name: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class LLMSubroutineResult(BaseModel):
    """Completed extraction output and conversation provenance, including model call counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_ref: str
    conversation_ref: str
    trace_ref: str
    n_llm_calls: int = 0
    n_harness_turns: int = 0


class LLMSubroutineTraceInput(BaseModel):
    """Conversation and call-trace references to assemble into a subroutine trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    call_ref_base: str
    harness_trace_refs: list[str] = Field(default_factory=list)


class LLMSubroutineTraceResult(BaseModel):
    """Reference to the assembled and persisted subroutine trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_ref: str


class OpenRouterCallInput(BaseModel):
    """Persisted conversation, model settings, and output paths for one OpenRouter call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    next_conversation_ref: str
    call_ref: str
    assistant_ref: str
    llm: EmbeddedLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    log_label: str


class ToolCallSummary(BaseModel):
    """Tool call identity and position within an assistant response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    id: str
    name: str


class OpenRouterCallResult(BaseModel):
    """Stored provider response references, token usage, duration, and requested tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    assistant_ref: str
    model: str
    stop_reason: str | None = None
    time: float
    usage: dict[str, int | None] | None = None
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)


class StoredConversation(TypedDict):
    """Ordered provider messages persisted between LLM activity calls."""

    messages: list[JsonObject]
