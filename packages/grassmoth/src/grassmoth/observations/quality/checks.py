"""validation low-level validation checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import polars as pl

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.identity import IndicatorRef
from grassmoth.artifacts.observations.time import ModelTime, ObservationInstant
from grassmoth.artifacts.reports.checks import Evaluated

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime, timedelta

    from grassmoth.artifacts.identity import IndicatorId
    from grassmoth.artifacts.model.construct import ConstructSpec
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.artifacts.reports.data_quality import DataFinding

MIN_OBSERVATIONS = 10
MIN_COVERAGE_PERIODS = 10
MAX_GAP_MULTIPLIER = 5
OUTLIER_IQR_MULTIPLIER = 3.0
MIN_ALIGNED_FOR_CFA = 10
HALLUCINATION_DUPLICATE_THRESHOLD = 0.5


def check_continuous_range(
    values: pl.Series,
    ind_name: IndicatorId,
) -> tuple[list[DataFinding], int]:
    """Check continuous readings for outliers beyond the interquartile range.

    Args:
        values: Usable numeric measurements for one indicator.
        ind_name: Indicator identity to attach to reported issues.

    Returns:
        The issues and number of values violating the selected check.
    """
    issues: list[DataFinding] = []
    violation_count = 0

    if len(values) >= MIN_OBSERVATIONS:
        # The caller supplies non-null Float64 values; Polars does not carry
        # a Series' dtype through its scalar return annotations.
        q1 = cast("float", values.quantile(0.25))
        q3 = cast("float", values.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outliers = values.filter((values < lower) | (values > upper))
            violation_count = len(outliers)
            if violation_count > 0:
                issues.append(
                    Evaluated(
                        code="dtype_violation",
                        subject=IndicatorRef(id=ind_name),
                        outcome="failed",
                        evidence=f"{violation_count} outlier(s) outside [{lower:.2f}, {upper:.2f}]",
                    )
                )

    return issues, violation_count


def check_time_coverage(
    parsed_ts: pl.Series,
    model_clock_hours: float,
    ind_name: IndicatorId,
) -> tuple[list[DataFinding], float | None]:
    """Assess whether recorded times span enough model-clock periods.

    Args:
        parsed_ts: Parsed measurement timestamps.
        model_clock_hours: Duration of one model-clock period in hours.
        ind_name: Indicator identity for any coverage warning.

    Returns:
        Coverage issues and the fraction of the required span covered, capped at
        one; the fraction is ``None`` when coverage cannot be assessed.
    """
    issues: list[DataFinding] = []
    if len(parsed_ts) < 2:
        return issues, None

    ts_max = cast("datetime", parsed_ts.max())
    ts_min = cast("datetime", parsed_ts.min())
    time_span_hours = (ts_max - ts_min).total_seconds() / 3600
    min_hours = MIN_COVERAGE_PERIODS * model_clock_hours
    coverage_ratio = min(time_span_hours / min_hours, 1.0) if min_hours > 0 else None

    if time_span_hours < min_hours:
        issues.append(
            Evaluated(
                code="insufficient_coverage",
                subject=IndicatorRef(id=ind_name),
                outcome="failed",
                evidence=f"Time span {time_span_hours:.0f}h < required {min_hours}h "
                f"({MIN_COVERAGE_PERIODS} x {model_clock_hours}h)",
            )
        )

    return issues, coverage_ratio


def check_timestamp_gaps(
    parsed_ts: pl.Series,
    model_clock_hours: float,
    ind_name: IndicatorId,
) -> tuple[list[DataFinding], float | None]:
    """Assess the largest consecutive gap relative to the model clock.

    Args:
        parsed_ts: Parsed measurement timestamps, sorted internally.
        model_clock_hours: Duration of one model-clock period in hours.
        ind_name: Indicator identity for any gap warning.

    Returns:
        Gap issues and the largest gap divided by the allowed gap threshold, or
        ``None`` for the ratio when there are too few timestamps.
    """
    issues: list[DataFinding] = []
    if len(parsed_ts) < 3:
        return issues, None

    diffs = parsed_ts.sort().diff().drop_nulls()
    max_gap = cast("timedelta", diffs.max())
    max_gap_hours = max_gap.total_seconds() / 3600
    threshold = MAX_GAP_MULTIPLIER * model_clock_hours
    max_gap_ratio = max_gap_hours / threshold if threshold > 0 else None

    if max_gap_hours > threshold:
        issues.append(
            Evaluated(
                code="large_timestamp_gap",
                subject=IndicatorRef(id=ind_name),
                outcome="failed",
                evidence=f"Max consecutive gap {max_gap_hours:.0f}h > "
                f"{MAX_GAP_MULTIPLIER}x {model_clock_hours}h ({threshold}h)",
            )
        )

    return issues, max_gap_ratio


def check_hallucination_signals(
    values: pl.Series, dtype: str, ind_name: IndicatorId
) -> tuple[list[DataFinding], float, bool]:
    """Flag suspicious repetition and arithmetic patterns in extracted values.

    Args:
        values: Usable measurements from one indicator.
        dtype: Declared data kind used to interpret repeated values.
        ind_name: Indicator identity to attach to warnings.

    Returns:
        Issues, the fraction occupied by the most frequent value, and whether
        distinct sorted values form a nonconstant arithmetic sequence. These are
        heuristic signals, not proof of fabricated measurements.
    """
    issues: list[DataFinding] = []
    n = len(values)
    duplicate_pct = 0.0
    arithmetic_sequence_detected = False

    if n < 2:
        return issues, duplicate_pct, arithmetic_sequence_detected

    vc = values.value_counts()
    max_count = cast("int", vc["count"].max())
    duplicate_pct = max_count / n if n > 0 else 0.0

    if dtype not in ("binary", "count"):
        variance = cast("float", values.var())
        if variance > 0 and max_count > n * HALLUCINATION_DUPLICATE_THRESHOLD:
            most_common = vc.sort("count", descending=True).row(0)[0]
            issues.append(
                Evaluated(
                    code="suspicious_pattern",
                    subject=IndicatorRef(id=ind_name),
                    outcome="failed",
                    evidence=f">{HALLUCINATION_DUPLICATE_THRESHOLD * 100:.0f}% of values "
                    f"are {most_common} ({max_count}/{n})",
                )
            )

    if n >= 5:
        diffs = values.sort().diff().drop_nulls()
        if diffs.n_unique() == 1:
            step = diffs[0]
            if step != 0:
                arithmetic_sequence_detected = True
                issues.append(
                    Evaluated(
                        code="suspicious_pattern",
                        subject=IndicatorRef(id=ind_name),
                        outcome="failed",
                        evidence=f"Values form arithmetic sequence with step {step}",
                    )
                )

    return issues, duplicate_pct, arithmetic_sequence_detected


def check_construct_correlations(
    observations: ObservationDataset,
    construct_lookup: Mapping[IndicatorId, ConstructSpec],
) -> list[DataFinding]:
    """Warn when sufficiently aligned indicators of the same construct correlate negatively."""
    issues: list[DataFinding] = []

    construct_indicators: dict[str, list[IndicatorId]] = {}
    for indicator_id, construct in construct_lookup.items():
        construct_indicators.setdefault(construct.id, []).append(indicator_id)

    for construct_id, indicator_names in construct_indicators.items():
        if len(indicator_names) < 2:
            continue
        for i, name_a in enumerate(indicator_names):
            for name_b in indicator_names[i + 1 :]:
                if name_a not in observations.series or name_b not in observations.series:
                    continue
                origin = ObservationInstant(observations.time_origin)
                data_a, data_b = (
                    pl.DataFrame(
                        {
                            "day": [
                                ModelTime(float(day)).at(origin).value.date()
                                for day in observations.series[identity].history.times.values
                            ],
                            label: pl.Series(
                                vector_values(observations.series[identity].history.values)
                            ).fill_nan(None),
                        }
                    )
                    .drop_nulls()
                    .group_by("day")
                    .agg(pl.col(label).mean())
                    for identity, label in ((name_a, "value_a"), (name_b, "value_b"))
                )
                aligned = data_a.join(data_b, on="day", how="inner")
                if len(aligned) < MIN_ALIGNED_FOR_CFA:
                    continue
                corr = aligned.select(pl.corr("value_a", "value_b")).item()
                if corr is not None and not math.isnan(corr) and corr < 0:
                    issues.append(
                        Evaluated(
                            code="low_construct_correlation",
                            subject="dataset",
                            outcome="failed",
                            evidence=f"Indicators {name_a} and {name_b} of {construct_id} have negative "
                            f"daily correlation (r={corr:.3f}), violating reflective "
                            f"measurement assumption",
                        )
                    )

    return issues
