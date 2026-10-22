"""Pure data-quality findings and empirical profiles for recorded indicators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from grassmoth.artifacts.identity import IndicatorId, IndicatorRef
from grassmoth.artifacts.observations.semantics import SupportKind
from grassmoth.artifacts.reports.checks import Evaluated, NotEvaluated
from grassmoth.artifacts.reports.data_quality import (
    DataFinding,
    DataProfileReport,
    IndicatorEmpiricalProfile,
)
from grassmoth.observations.quality.checks import (
    MIN_OBSERVATIONS,
    OBSERVATION_TIME_COLUMN,
    check_dtype_range,
    check_hallucination_signals,
    check_time_coverage,
    check_timestamp_gaps,
    parse_timestamp_series,
    timestamp_issue_specs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.artifacts.model.construct import ConstructSpec
    from grassmoth.artifacts.observations.spec import (
        AuthoredObservationSpec,
        ResolvedObservationSpec,
    )


@dataclass(frozen=True)
class IndicatorContext:
    """Parsed observations shared by the indicator's data-quality checks."""

    name: IndicatorId
    ind_data: pl.DataFrame
    values: pl.Series
    n_obs: int
    variance: float | None
    dtype: str | None
    is_time_invariant: bool
    model_clock_hours: float | None
    parsed_ts: pl.Series
    n_unparseable: int
    n_total_ts: int


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
    indicator_id: IndicatorId,
    data: pl.DataFrame,
    definitions: Mapping[IndicatorId, ResolvedObservationSpec],
) -> tuple[DataFinding, ...]:
    """Evaluate data quality once, with codes and evidence independent of action severity."""
    ctx = _build_indicator_context(indicator_id, data, definitions, {})
    subject = IndicatorRef(id=indicator_id)
    if ctx is None:
        return (
            Evaluated(
                code="missing" if data.is_empty() else "no_numeric",
                subject=subject,
                outcome="failed",
                evidence="No observations were extracted."
                if data.is_empty()
                else "No numeric observations were retained.",
            ),
        )
    findings: list[DataFinding] = []

    def measured(
        code: str, evidence: str, *, outcome: Literal["passed", "failed"] = "passed"
    ) -> None:
        findings.append(Evaluated(code=code, subject=subject, outcome=outcome, evidence=evidence))

    measured("data_availability", f"{ctx.n_obs} usable observations.")
    timestamp_problems = timestamp_issue_specs(ctx.n_total_ts, ctx.n_unparseable)
    measured(
        "timestamps",
        "; ".join(timestamp_problems)
        if timestamp_problems
        else f"{ctx.n_unparseable}/{ctx.n_total_ts} timestamps are unparseable.",
        outcome="failed" if timestamp_problems else "passed",
    )
    measured(
        "sample_size",
        f"{ctx.n_obs} observations; recommended minimum {MIN_OBSERVATIONS}.",
        outcome="failed" if ctx.n_obs < MIN_OBSERVATIONS else "passed",
    )
    if not ctx.is_time_invariant:
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
    if ctx.dtype is not None:
        problems, violations = check_dtype_range(ctx.values, ctx.dtype, ctx.name)
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
    if not ctx.is_time_invariant and ctx.model_clock_hours is not None:
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
    indicator_id: IndicatorId,
    ind_data: pl.DataFrame,
    indicator_lookup: Mapping[IndicatorId, AuthoredObservationSpec | ResolvedObservationSpec],
    construct_lookup: Mapping[str, ConstructSpec],
) -> IndicatorContext | None:
    values_df = ind_data.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    variance = cast("float | None", values.var())

    indicator_meta = indicator_lookup.get(indicator_id)
    dtype = indicator_meta.measurement_dtype if indicator_meta is not None else None
    model_clock_hours = (
        ind_data.filter(pl.col("value").is_not_null())
        .select(
            ((pl.col("support_end") - pl.col("support_start")).dt.total_seconds() / 3600).median()
        )
        .item()
    )
    construct_meta = construct_lookup.get(indicator_id)
    is_time_invariant = (
        construct_meta is not None and construct_meta.temporal_status == "time_invariant"
    )

    timestamps = ind_data[OBSERVATION_TIME_COLUMN]
    parsed = parse_timestamp_series(timestamps)
    if (
        indicator_meta is not None
        and indicator_meta.support_kind == SupportKind.POINT
        and (window := indicator_meta.observation_window) is not None
    ):
        # A point's support has zero width. Its declared observation window owns
        # cadence; calendar windows retain their actual elapsed length.
        model_clock_hours = (
            parsed.to_frame("anchor")
            .select(
                (
                    (
                        pl.col("anchor").dt.offset_by(window.polars_interval) - pl.col("anchor")
                    ).dt.total_seconds()
                    / 3600
                ).median()
            )
            .item()
        )

    return IndicatorContext(
        name=indicator_id,
        ind_data=ind_data,
        values=values,
        n_obs=n_obs,
        variance=variance,
        dtype=dtype,
        is_time_invariant=is_time_invariant,
        model_clock_hours=model_clock_hours,
        parsed_ts=parsed.filter(
            ind_data["value"].cast(pl.Float64, strict=False).is_not_null()
        ).drop_nulls(),
        n_unparseable=parsed.null_count(),
        n_total_ts=len(timestamps),
    )


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return None if math.isnan(value) else value


def compute_empirical_profile(
    indicator_id: IndicatorId,
    model_data: pl.DataFrame,
) -> IndicatorEmpiricalProfile | None:
    """Summarize the retained values of one indicator without model assumptions."""
    ind_model = model_data.filter(pl.col("indicator_id") == indicator_id)
    values_df = ind_model.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    return IndicatorEmpiricalProfile(
        n_obs=n_obs,
        min=_float_or_none(cast("float | None", values.min())),
        q25=_float_or_none(values.quantile(0.25)),
        q50=_float_or_none(values.quantile(0.50)),
        q75=_float_or_none(values.quantile(0.75)),
        max=_float_or_none(cast("float | None", values.max())),
        mean=_float_or_none(cast("float | None", values.mean())),
    )
