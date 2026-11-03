"""Pure data-quality findings and empirical profiles for recorded indicators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.identity import IndicatorId, IndicatorRef
from grassmoth.artifacts.observations.semantics import SupportKind
from grassmoth.artifacts.observations.time import ModelTime, ObservationInstant
from grassmoth.artifacts.reports.checks import CheckOutcome, Evaluated, NotEvaluated
from grassmoth.artifacts.reports.data_quality import (
    DataFinding,
    DataProfileReport,
    IndicatorEmpiricalProfile,
)
from grassmoth.observations.quality.checks import (
    MIN_OBSERVATIONS,
    check_continuous_range,
    check_hallucination_signals,
    check_time_coverage,
    check_timestamp_gaps,
)

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.artifacts.observations.dataset import ObservationSeries


@dataclass(frozen=True)
class IndicatorContext:
    """Parsed observations shared by the indicator's data-quality checks."""

    name: IndicatorId
    values: pl.Series
    n_obs: int
    variance: float | None
    dtype: str | None
    model_clock_hours: float | None
    parsed_ts: pl.Series


def no_data_validation_result() -> DataProfileReport:
    """Retain the evaluated absence of extracted data."""
    return DataProfileReport(
        indicators={},
        findings=(
            Evaluated(
                code="no_data", subject="dataset", outcome="failed", evidence="No data extracted."
            ),
        ),
    )


def indicator_findings(
    series: ObservationSeries,
    time_origin: datetime,
) -> tuple[DataFinding, ...]:
    """Evaluate data quality from an admitted series, without reparsing readings."""
    ctx = _build_indicator_context(series, time_origin)
    subject = IndicatorRef(id=series.observation.id)
    if ctx is None:
        empty = series.history.times.layout[0][0] == 0
        return (
            Evaluated(
                code="missing" if empty else "no_numeric",
                subject=subject,
                outcome="failed",
                evidence="No observations were extracted."
                if empty
                else "No numeric observations were retained.",
            ),
        )
    findings: list[DataFinding] = []

    def measured(code: str, evidence: str, *, outcome: CheckOutcome = "passed") -> None:
        findings.append(Evaluated(code=code, subject=subject, outcome=outcome, evidence=evidence))

    measured("data_availability", f"{ctx.n_obs} usable observations.")
    measured(
        "sample_size",
        f"{ctx.n_obs} observations; recommended minimum {MIN_OBSERVATIONS}.",
        outcome="failed" if ctx.n_obs < MIN_OBSERVATIONS else "passed",
    )
    if ctx.variance is None:
        findings.append(
            NotEvaluated(
                code="variance",
                subject=subject,
                reason="INSUFFICIENT_OBSERVATIONS",
                detail="Variance requires at least two observations.",
            )
        )
    else:
        measured(
            "variance",
            f"Observed variance: {ctx.variance:g}.",
            outcome="failed" if ctx.variance == 0 else "passed",
        )
    if ctx.dtype == "continuous":
        problems, violations = check_continuous_range(ctx.values, ctx.name)
        if problems:
            findings.extend(problems)
        else:
            measured("dtype_violation", f"{violations} values violate the declared data kind.")
    problems, duplicate_fraction, arithmetic = check_hallucination_signals(
        ctx.values, ctx.dtype or "continuous", ctx.name
    )
    if problems:
        findings.extend(problems)
    else:
        measured(
            "suspicious_pattern",
            f"Largest repeated-value fraction: {duplicate_fraction:g}; arithmetic sequence: {arithmetic}.",
        )
    if ctx.model_clock_hours is not None:
        for code, check in (
            ("insufficient_coverage", check_time_coverage),
            ("large_timestamp_gap", check_timestamp_gaps),
        ):
            problems, ratio = check(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
            if problems:
                findings.extend(problems)
            elif ratio is None:
                findings.append(
                    NotEvaluated(
                        code=code,
                        subject=subject,
                        reason="INSUFFICIENT_TIMES",
                        detail="Too few observation times to assess this criterion.",
                    )
                )
            else:
                measured(code, f"Observed-to-threshold ratio: {ratio:g}.")
    return tuple(findings)


def _build_indicator_context(
    series: ObservationSeries, time_origin: datetime
) -> IndicatorContext | None:
    history, observation = series.history, series.observation
    numeric = vector_values(history.values)
    observed = ~np.isnan(numeric)
    values = pl.Series(numeric[observed])
    if not len(values):
        return None
    origin = ObservationInstant(time_origin)
    instants = pl.Series(
        [
            ModelTime(float(day)).at(origin).value.replace(tzinfo=None)
            for day in history.times.values
        ],
        dtype=pl.Datetime("us"),
    )
    widths = vector_values(history.support_end) - vector_values(history.support_start)
    model_clock_hours = float(np.median(widths[observed]) * 24)
    if observation.support_kind == SupportKind.POINT:
        model_clock_hours = (
            instants.to_frame("anchor")
            .select(
                (
                    (
                        pl.col("anchor").dt.offset_by(
                            observation.observation_window.polars_interval
                        )
                        - pl.col("anchor")
                    ).dt.total_seconds()
                    / 3600
                ).median()
            )
            .item()
        )
    return IndicatorContext(
        name=observation.id,
        values=values,
        n_obs=len(values),
        variance=cast("float | None", values.var()),
        dtype=observation.measurement.dtype,
        model_clock_hours=model_clock_hours,
        parsed_ts=instants.filter(pl.Series(observed)),
    )


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return None if math.isnan(value) else value


def compute_empirical_profile(series: ObservationSeries) -> IndicatorEmpiricalProfile | None:
    """Summarize the retained values of one indicator without model assumptions."""
    numeric = vector_values(series.history.values)
    values = pl.Series(numeric[~np.isnan(numeric)])
    n_obs = len(values)
    if n_obs == 0:
        return None
    return IndicatorEmpiricalProfile(
        n_obs=n_obs,
        min=_float_or_none(cast("float | None", values.min())),
        q25=_float_or_none(values.quantile(0.25)),
        q50=_float_or_none(values.quantile(0.50)),
        q75=_float_or_none(values.quantile(0.75)),
        max=_float_or_none(cast("float | None", values.max())),
        mean=_float_or_none(cast("float | None", values.mean())),
    )
