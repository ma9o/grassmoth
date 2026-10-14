"""Input of the extraction worker's validation tool."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class ValidateExtractionsInput(Value):
    """JSON worker output submitted to the extraction validation tool."""

    output_json: str = Field(
        description="The JSON string containing the worker output to validate."
    )
