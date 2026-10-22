"""Production message builders for measurement-extraction subroutines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_service.actions.prepare_data.prompts import SYSTEM, USER

if TYPE_CHECKING:
    from nof1_causal_service.actions.prepare_data.context import MeasurementContext


def _format_indicators(measurement_structure: MeasurementContext) -> str:
    """Format indicators and their observation semantics for a worker prompt."""
    lines = []
    for indicator in measurement_structure.indicators:
        name = indicator.observation.name
        how_to_measure = indicator.extraction.how_to_measure
        dtype = indicator.observation.measurement_dtype
        support_kind = indicator.observation.support_kind.value
        summary_operator = indicator.observation.summary_operator.value
        window = measurement_structure.window(indicator)
        details = [
            dtype,
            f"operator={summary_operator}",
            f"support={support_kind}",
            f"window={window}",
        ]
        levels = (
            indicator.observation.ordinal_levels
            if dtype == "ordinal"
            else indicator.observation.categorical_levels
        )
        if dtype in {"ordinal", "categorical"} and levels:
            codebook = ", ".join(f"{index}={level}" for index, level in enumerate(levels))
            details.append(f"{dtype}_codes={codebook}")

        lines.append(
            f"- {name} [{indicator.observation.id}] ({', '.join(details)}): {how_to_measure}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class WorkerMessages:
    """Build the prompt messages for one measurement-extraction chunk."""

    question: str
    measurement_structure: MeasurementContext
    window_text: str
    n_windows: int

    def extraction_messages(self) -> list[dict[str, str]]:
        """Render the extraction system prompt and user request for this chunk's indicators and windows."""
        indicators_text = _format_indicators(self.measurement_structure)
        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=self.question,
                    indicators=indicators_text,
                    n_windows=self.n_windows,
                    window_text=self.window_text,
                ),
            },
        ]


__all__ = ["WorkerMessages"]
