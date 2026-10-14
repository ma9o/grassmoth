"""Schemas for worker LLM outputs."""

import math
from typing import assert_never

import polars as pl
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.observations.dtypes import MeasurementDtype
from nof1_causal_lab.artifacts.observations.semantics import normalize_level_label


class ExtractionRow(TypedDict):
    """Pre-annotation dataframe row emitted by both extraction engines."""

    indicator_id: str
    value: str | None
    timestamp: str


EXTRACTION_ROW_SCHEMA = {"indicator_id": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8}


class WindowExtraction(Value):
    """A single extracted observation for an indicator within a support window."""

    window_start: str = Field(
        description="The support-window start time (e.g. '2024-01-15T00:00:00')"
    )
    indicator_id: IndicatorId = Field(description="Persistent identity of the indicator")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )


class WorkerOutput(Value):
    """Complete output from a worker processing a chunk of support windows."""

    extractions: tuple[WindowExtraction, ...] = Field(
        default_factory=tuple,
        description="Extracted observations for indicators (one per support window per indicator)",
    )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert extractions to a Polars DataFrame.

        Returns:
            DataFrame with columns: indicator (Utf8), value (Utf8), timestamp (Utf8).
            The timestamp column contains the support-window start time.
            Value column is stored as string for downstream encoding.
        """
        if not self.extractions:
            return pl.DataFrame(schema=EXTRACTION_ROW_SCHEMA)

        rows = []
        for e in self.extractions:
            rows.append(
                {
                    "indicator_id": e.indicator_id,
                    "value": str(e.value) if e.value is not None else None,
                    "timestamp": e.window_start,
                }
            )

        return pl.DataFrame(rows, schema=EXTRACTION_ROW_SCHEMA)


def _check_dtype_match(value: object, expected_dtype: MeasurementDtype) -> bool:
    """Check if a value matches the expected measurement_dtype."""
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return False
    if value is None:
        return True  # None is always acceptable

    match expected_dtype:
        case "continuous":
            return isinstance(value, (int, float))
        case "binary":
            return isinstance(value, bool) or value in (
                0,
                1,
                "0",
                "1",
                "true",
                "false",
                "True",
                "False",
            )
        case "count":
            return isinstance(value, (int, float)) and value >= 0 and value == int(value)
        case "ordinal":
            return not isinstance(value, bool) and (
                isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            )
        case "categorical":
            return isinstance(value, str)
        case _:
            assert_never(expected_dtype)


def validate_worker_output(
    data: object,
    measurement_structure: MeasurementContext,
    expected_window_starts: list[str] | None = None,
) -> tuple[WorkerOutput | None, list[str]]:
    """Validate worker output dict, collecting ALL errors instead of failing on first.

    Args:
        data: Dictionary to validate as WorkerOutput
        measurement_structure: The owned observation definitions to validate against
        expected_window_starts: If provided, validate that extractions only
            reference these support-window starts.

    Returns:
        Tuple of (validated output or None, list of error messages)
    """
    errors = []

    # Basic structure checks
    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    extractions = data.get("extractions", [])

    if not isinstance(extractions, list):
        errors.append("'extractions' must be a list")
        extractions = []

    # Keep the pinned observation definitions indexed by identity.
    indicator_info = {
        indicator.observation.id: indicator for indicator in measurement_structure.indicators
    }
    expected_window_start_set = set(expected_window_starts) if expected_window_starts else None

    # Validate each extraction
    valid_extractions: list[WindowExtraction] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, ext_data in enumerate(extractions):
        if not isinstance(ext_data, dict):
            errors.append(f"extractions[{i}]: must be a dictionary")
            continue

        try:
            ext = WindowExtraction.model_validate(
                {
                    "window_start": ext_data.get("window_start", "<missing>"),
                    "indicator_id": ext_data.get("indicator_id", "<missing>"),
                    "value": ext_data.get("value"),
                },
                strict=True,
            )
        except ValidationError as exc:
            errors.append(f"extractions[{i}]: {exc}")
            continue
        window_start = ext.window_start
        ind_name = ext.indicator_id
        value = ext.value

        # Check support window is valid
        if expected_window_start_set is not None and window_start not in expected_window_start_set:
            errors.append(
                f"extractions[{i}]: window_start '{window_start}' not in expected support windows"
            )
            continue

        # Check indicator exists
        if ind_name not in indicator_info:
            valid_ind_names = ", ".join(sorted(indicator_info.keys()))
            errors.append(
                f"extractions[{i}]: indicator '{ind_name}' not in indicators. "
                f"Valid indicators: {valid_ind_names}"
            )
            continue

        # Check no duplicate (window_start, indicator) pairs
        pair = (window_start, ind_name)
        if pair in seen_pairs:
            errors.append(
                f"extractions[{i}]: duplicate (window_start, indicator) pair: "
                f"({window_start}, {ind_name})"
            )
            continue
        seen_pairs.add(pair)

        # Check dtype match
        expected_dtype = indicator_info[ind_name].observation.measurement_dtype
        if not _check_dtype_match(value, expected_dtype):
            errors.append(
                f"extractions[{i}]: value {value!r} for '{ind_name}' doesn't match "
                f"expected dtype '{expected_dtype}'"
            )
            continue

        if expected_dtype == "categorical" and isinstance(value, str):
            levels = indicator_info[ind_name].observation.categorical_levels
            if levels is not None and normalize_level_label(value) not in {
                normalize_level_label(label) for label in levels
            }:
                errors.append(
                    f"extractions[{i}]: categorical value {value!r} is outside the codebook"
                )
                continue

        if expected_dtype == "ordinal" and value is not None:
            ordinal_levels = indicator_info[ind_name].observation.ordinal_levels
            ordinal_code = int(value)
            if ordinal_code < 0:
                errors.append(
                    f"extractions[{i}]: ordinal value {value!r} for '{ind_name}' must be >= 0"
                )
                continue
            if ordinal_levels is not None and ordinal_code >= len(ordinal_levels):
                errors.append(
                    f"extractions[{i}]: ordinal value {value!r} for '{ind_name}' "
                    f"must be in 0..{len(ordinal_levels) - 1}"
                )
                continue
            value = ordinal_code

        valid_extractions.append(
            WindowExtraction(
                window_start=ext.window_start, indicator_id=ext.indicator_id, value=value
            )
        )

    if expected_window_start_set is not None:
        expected_pairs = {
            (window, indicator)
            for window in expected_window_start_set
            for indicator in indicator_info
        }
        if missing := expected_pairs - seen_pairs:
            errors.append(
                f"Missing extractions (use null for unavailable values): {sorted(missing)}"
            )

    # If no errors, build and return the output
    if not errors:
        return WorkerOutput(extractions=tuple(valid_extractions)), []

    return None, errors
