"""Effective LLM request identities, independent of study and run locations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.actions.temporal.llm.context import subroutine_context_messages
from nof1_causal_lab.storage.cache import cache_path, content_key

if TYPE_CHECKING:
    from nof1_causal_lab.llm.specs import LLMProfileSpec

# Bump the version when extraction prompts, tools, schemas, or validation change.
EXTRACTION_POLICY_VERSION = "extraction-v4"


def preparation_cache_path(
    context_ref: str,
    llm: LLMProfileSpec,
    max_tool_turns: int,
    inputs: object,
) -> str:
    """Derive an extraction cache path from prompts, tools, model settings, and input content."""
    system, users, tools = subroutine_context_messages(context_ref)
    key = content_key(
        {
            "policy": EXTRACTION_POLICY_VERSION,
            "system": system,
            "users": users,
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "llm": llm.model_dump(mode="json"),
            "max_tool_turns": max_tool_turns,
            "inputs": inputs,
        }
    )
    return cache_path("measurement_extraction", key)
