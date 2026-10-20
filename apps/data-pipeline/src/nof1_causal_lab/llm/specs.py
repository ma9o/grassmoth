"""OpenRouter model definitions shared by configuration and orchestration."""

from typing import Literal

from nof1_causal_lab.artifacts.base import Value

type EmbeddedReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class EmbeddedLLMSpec(Value):
    """OpenRouter model and its optional generation controls."""

    model: str
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: EmbeddedReasoningEffort | None = None
