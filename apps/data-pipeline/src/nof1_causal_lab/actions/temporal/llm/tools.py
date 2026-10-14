"""Tool adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.prepare_data.tool_contracts import ValidateExtractionsInput
from nof1_causal_lab.actions.temporal.extraction.messages import MeasurementChunkContext
from nof1_causal_lab.actions.temporal.llm.storage import read_subroutine_json, write_subroutine_json

if TYPE_CHECKING:
    from nof1_causal_lab.actions.temporal.llm.messages import (
        HarnessToolRequest,
        LLMToolExecutionInput,
    )
    from nof1_causal_lab.json_types import JsonObject


def _validate_measurement_payload(
    *,
    context_ref: str,
    data: object,
) -> tuple[JsonObject | None, str]:
    from nof1_causal_lab.actions.prepare_data.worker_output import validate_worker_output

    spec = read_subroutine_json(context_ref, MeasurementChunkContext)
    output, errors = validate_worker_output(
        data,
        spec.measurement_structure,
        spec.window_starts,
    )
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
    if output is None:
        return None, "VALIDATION ERRORS:\n- validator returned no output"
    payload: JsonObject = output.model_dump(mode="json")
    return payload, "VALID"


async def execute_subroutine_tool(
    *,
    activity_input: LLMToolExecutionInput | HarnessToolRequest,
    args: object,
    result_ref: str,
) -> tuple[str, str | None]:
    """Validate extraction-tool arguments and persist accepted worker output.

    Args:
        activity_input: Invocation carrying the subroutine's measurement context.
        args: Tool arguments containing the worker's output JSON.
        result_ref: Storage destination for output that passes measurement validation.

    Returns:
        Validation feedback and the stored result reference, or ``None`` for the
        reference when the worker output is rejected.
    """
    data = json.loads(ValidateExtractionsInput.model_validate(args).output_json)
    context_output, feedback = _validate_measurement_payload(
        context_ref=activity_input.subroutine.context_ref,
        data=data,
    )
    if context_output is not None:
        write_subroutine_json(result_ref, context_output)
        return feedback, result_ref
    return feedback, None
