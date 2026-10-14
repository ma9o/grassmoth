"""Backend-specific LLM definitions shared by configuration and orchestration."""

from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

type HarnessName = Literal["none", "claude-code", "codex", "pi"]
type EmbeddedReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type HarnessEffort = Literal["low", "medium", "high", "xhigh", "max"]
type PiThinking = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class EmbeddedLLMSpec(Value):
    """OpenRouter model and its optional generation controls."""

    harness: Literal["none"] = "none"
    model: str
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: EmbeddedReasoningEffort | None = None


class ClaudeCodeLLMSpec(Value):
    """Claude Code model and subprocess controls."""

    harness: Literal["claude-code"] = "claude-code"
    model: str
    bin: str | None = None
    effort: HarnessEffort | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None


class CodexLLMSpec(Value):
    """Codex model and subprocess controls."""

    harness: Literal["codex"] = "codex"
    model: str
    bin: str | None = None
    timeout: int | None = None
    reasoning_effort: HarnessEffort | None = None
    service_tier: str | None = None


class PiLLMSpec(Value):
    """Pi provider, model, and subprocess controls."""

    harness: Literal["pi"] = "pi"
    model: str
    bin: str | None = None
    timeout: int | None = None
    provider: str | None = None
    thinking: PiThinking | None = None


type HarnessLLMSpec = Annotated[
    ClaudeCodeLLMSpec | CodexLLMSpec | PiLLMSpec, Field(discriminator="harness")
]
type LLMProfileSpec = Annotated[
    EmbeddedLLMSpec | ClaudeCodeLLMSpec | CodexLLMSpec | PiLLMSpec,
    Field(discriminator="harness"),
]
