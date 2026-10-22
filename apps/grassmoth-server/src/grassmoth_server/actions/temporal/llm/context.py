"""Context adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

from grassmoth_server.actions.temporal.extraction.messages import MeasurementChunkContext
from grassmoth_server.actions.temporal.llm.messages import LLMToolSpec
from grassmoth_server.actions.temporal.llm.storage import read_subroutine_json


def subroutine_context_messages(
    context_ref: str,
) -> tuple[str | None, list[str], list[LLMToolSpec]]:
    """Build the extraction prompts and validation tool from a saved chunk context.

    Args:
        context_ref: Stored measurement chunk specification to render.

    Returns:
        A tuple of the optional system prompt, ordered user prompts, and available
        tool specifications. The terminal tool validates the worker's extraction JSON.
    """
    from grassmoth_server.actions.prepare_data.messages import WorkerMessages

    spec = read_subroutine_json(context_ref, MeasurementChunkContext)
    messages = WorkerMessages(
        question=spec.question,
        measurement_structure=spec.measurement_structure,
        window_text=spec.window_text,
        n_windows=len(spec.window_starts),
    ).extraction_messages()
    system_prompt = None
    user_messages: list[str] = []
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            user_messages.append(message["content"])
    return (
        system_prompt,
        user_messages,
        [
            LLMToolSpec(
                name="validate_extractions",
                description="Validate worker extraction output JSON.",
                parameters={
                    "type": "object",
                    "properties": {
                        "output_json": {
                            "type": "string",
                            "description": "The JSON string containing the worker output.",
                        }
                    },
                    "required": ["output_json"],
                    "additionalProperties": False,
                },
            )
        ],
    )
