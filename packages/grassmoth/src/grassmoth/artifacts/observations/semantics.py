"""Shared indicator observation semantics.

This module owns level-label matching and how an indicator's aggregation
maps to downstream measurement semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from grassmoth.artifacts.observations.expressions import WindowExpression


class SupportKind(StrEnum):
    """Whether an observation describes a point in time or summarizes an interval."""

    POINT = "point"
    INTERVAL = "interval"


class AnchorPolicy(StrEnum):
    """This policy selects which boundary of a measurement window receives its timestamp."""

    SUPPORT_START = "support_start"
    SUPPORT_END = "support_end"


class SummaryOperator(StrEnum):
    """How measurements within a support window are reduced to one observation."""

    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    COUNT = "count"
    MEAN = "mean"
    STD = "std"


@dataclass(frozen=True)
class IndicatorObservationSemantics:
    """Canonical semantics derived from one indicator definition."""

    support_kind: SupportKind
    summary_operator: SummaryOperator
    anchor_policy: AnchorPolicy


def supported_summary_operators_text() -> str:
    """Return a stable human-readable list of supported operators."""
    return ", ".join(SummaryOperator)


def validate_indicator_observation_semantics(
    aggregation: SummaryOperator,
    measurement_dtype: str,
) -> str | None:
    """Return a user-facing validation error for unsupported semantics."""
    if measurement_dtype == "ordinal" and aggregation not in {
        SummaryOperator.FIRST,
        SummaryOperator.LAST,
    }:
        return "ordinal indicators currently support only first/last point measurements."

    if aggregation == SummaryOperator.COUNT and measurement_dtype != "count":
        return "aggregation 'count' requires measurement.dtype='count'."

    if (
        aggregation in {SummaryOperator.MEAN, SummaryOperator.STD}
        and measurement_dtype != "continuous"
    ):
        return f"aggregation '{aggregation}' requires measurement.dtype='continuous'."

    if aggregation == SummaryOperator.SUM and measurement_dtype not in {
        "continuous",
        "count",
    }:
        return "aggregation 'sum' requires measurement.dtype='continuous' or 'count'."

    return None


def derive_indicator_observation_semantics(
    aggregation: SummaryOperator,
    measurement_dtype: str,
    computed_rule: WindowExpression | None = None,
) -> IndicatorObservationSemantics:
    """Derive semantics from the computation, or the declared semantic aggregation."""
    if computed_rule is not None:
        summary = computed_rule.summary_operator
        if aggregation != summary:
            raise ValueError(
                f"computed_rule produces '{summary}' but aggregation is '{aggregation}'"
            )
        aggregation = summary
    error = validate_indicator_observation_semantics(aggregation, measurement_dtype)
    if error is not None:
        raise ValueError(error)

    match aggregation:
        case SummaryOperator.FIRST | SummaryOperator.LAST:
            support_kind = SupportKind.POINT
        case (
            SummaryOperator.SUM | SummaryOperator.COUNT | SummaryOperator.MEAN | SummaryOperator.STD
        ):
            support_kind = SupportKind.INTERVAL
        case _:
            assert_never(aggregation)

    anchor_policy = (
        AnchorPolicy.SUPPORT_START
        if aggregation == SummaryOperator.FIRST
        else AnchorPolicy.SUPPORT_END
    )
    return IndicatorObservationSemantics(
        support_kind=support_kind,
        summary_operator=aggregation,
        anchor_policy=anchor_policy,
    )
