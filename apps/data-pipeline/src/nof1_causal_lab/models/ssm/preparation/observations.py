"""Shared observation metadata helpers for SSM model preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.observations.semantics import AnchorPolicy
from nof1_causal_lab.artifacts.observations.time import ModelTime, ObservationInstant
from nof1_causal_lab.models.ssm.compile.observation_windows import (
    CompiledWindowPlan,
    compile_window_plan,
)
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure
from nof1_causal_lab.observations.rows import observation_window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel


@dataclass(frozen=True)
class ObservationSupportRuntime:
    """Structured support metadata aligned to the prepared wide observation matrix."""

    window_plan: CompiledWindowPlan
    manifest_names: tuple[str, ...]
    support_kinds: tuple[str | None, ...]
    summary_operators: tuple[str | None, ...]
    anchor_policies: tuple[str | None, ...]
    observation_windows: tuple[str | None, ...]
    support_start_times: np.ndarray  # shape (T, n_manifest), NaN when missing
    support_end_times: np.ndarray  # shape (T, n_manifest), NaN when missing

    def __post_init__(self) -> None:
        """Own immutable collections describing the resolved measurement support schedule."""
        freeze_fields(self)

    @classmethod
    def assembled(
        cls,
        *,
        window_plan: CompiledWindowPlan,
        manifest_names: Sequence[str],
        support_kinds: Sequence[str | None],
        summary_operators: Sequence[str | None],
        anchor_policies: Sequence[str | None],
        observation_windows: Sequence[str | None],
        support_start_times: np.ndarray,
        support_end_times: np.ndarray,
    ) -> Self:
        """Own resolved support metadata and detach every native buffer."""
        return cls(
            window_plan=window_plan,
            manifest_names=tuple(manifest_names),
            support_kinds=tuple(support_kinds),
            summary_operators=tuple(summary_operators),
            anchor_policies=tuple(anchor_policies),
            observation_windows=tuple(observation_windows),
            support_start_times=support_start_times,
            support_end_times=support_end_times,
        )

    @property
    def requires_interval_summary_handling(self) -> bool:
        """Whether any manifest requires interval-summary measurement handling."""
        return any(kind == "interval" for kind in self.support_kinds)

    @property
    def interval_summary_manifest_names(self) -> tuple[str, ...]:
        """Manifest names that require interval-summary measurement handling."""
        return tuple(
            name
            for name, kind in zip(self.manifest_names, self.support_kinds, strict=False)
            if kind == "interval"
        )

    @property
    def anchor_times(self) -> np.ndarray:
        """Model grid owned by the canonical measurement plan."""
        return self.window_plan.times


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    """Parse a datetime-like column to a consistent expression."""
    if df.schema.get(column) == pl.Utf8:
        return pl.col(column).str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
    dtype = df.schema.get(column)
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return pl.col(column).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return pl.col(column).cast(pl.Datetime, strict=False)


def _pivot_support_matrix(
    support_df: pl.DataFrame,
    *,
    value_col: str,
    base_times: pl.DataFrame,
    manifest_names: Sequence[str],
) -> np.ndarray:
    """Pivot one support-time column to a dense matrix aligned with wide_data rows."""
    pivoted = (
        support_df.select("time", "indicator", value_col)
        .pivot(on="indicator", index="time", values=value_col, aggregate_function="first")
        .sort("time")
    )
    aligned = base_times.join(pivoted, on="time", how="left")
    return aligned.select(manifest_names).to_numpy()


def compile_observation_support_runtime(
    observation_data: pl.DataFrame,
    wide_data: pl.DataFrame,
    manifest_names: Sequence[str],
    *,
    time_origin: datetime,
) -> ObservationSupportRuntime | ObservationPreflightFailure:
    """Compile long-format observation support metadata into wide aligned arrays."""
    df = observation_data

    df = df.with_columns(
        _datetime_expr(df, "anchor_time").alias("__anchor_dt"),
        _datetime_expr(df, "support_start").alias("__support_start_dt"),
        _datetime_expr(df, "support_end").alias("__support_end_dt"),
    ).drop_nulls(subset=["__anchor_dt"])

    origin = ObservationInstant(time_origin)
    df = df.with_columns(
        ModelTime.bind_column(pl.col("__anchor_dt"), origin).alias("time"),
        ModelTime.bind_column(pl.col("__support_start_dt"), origin).alias("__support_start_time"),
        ModelTime.bind_column(pl.col("__support_end_dt"), origin).alias("__support_end_time"),
    )

    base_times = wide_data.select(pl.col("time").cast(pl.Float64).alias("time"))
    anchor_times = base_times["time"].to_numpy()

    support_start_times = _pivot_support_matrix(
        df,
        value_col="__support_start_time",
        base_times=base_times,
        manifest_names=tuple(manifest_names),
    )
    support_end_times = _pivot_support_matrix(
        df,
        value_col="__support_end_time",
        base_times=base_times,
        manifest_names=tuple(manifest_names),
    )

    kind_window_rows = (
        df.group_by("indicator")
        .agg(
            pl.col("support_kind").drop_nulls().first().alias("support_kind"),
            pl.col("summary_operator").drop_nulls().first().alias("summary_operator"),
            pl.col("anchor_policy").drop_nulls().first().alias("anchor_policy"),
            pl.col("observation_window").drop_nulls().first().alias("observation_window"),
        )
        .iter_rows(named=True)
    )
    kind_window_lookup = {row["indicator"]: row for row in kind_window_rows}
    support_kinds = [kind_window_lookup[name]["support_kind"] for name in manifest_names]
    summary_operators = [kind_window_lookup[name]["summary_operator"] for name in manifest_names]
    anchor_policies = [kind_window_lookup[name]["anchor_policy"] for name in manifest_names]
    observation_windows = [
        kind_window_lookup[name]["observation_window"] for name in manifest_names
    ]
    window_plan = compile_window_plan(
        anchor_times, support_start_times, support_end_times, support_kinds, manifest_names
    )
    if isinstance(window_plan, ObservationPreflightFailure):
        return window_plan

    return ObservationSupportRuntime.assembled(
        window_plan=window_plan,
        manifest_names=tuple(manifest_names),
        support_kinds=tuple(support_kinds),
        summary_operators=tuple(summary_operators),
        anchor_policies=tuple(anchor_policies),
        observation_windows=tuple(observation_windows),
        support_start_times=support_start_times,
        support_end_times=support_end_times,
    )


def augment_wide_data_with_support_boundaries(
    observation_data: pl.DataFrame,
    wide_data: pl.DataFrame,
    *,
    time_origin: datetime,
) -> pl.DataFrame:
    """Add missing support-boundary rows to the wide matrix.

    Interval-summary observations may begin before the earliest anchored
    observation time. The support-aware likelihood needs those boundary times
    on the latent path, even when all manifests are missing there.
    """
    if (
        "time" in wide_data.columns
        and not wide_data.is_empty()
        and wide_data.select((pl.col("time") > 0).all()).item()
    ):
        initial = wide_data.head(1).select(
            pl.lit(0.0).alias("time"),
            *(
                pl.lit(None, dtype=wide_data.schema[name]).alias(name)
                for name in wide_data.columns
                if name != "time"
            ),
        )
        wide_data = pl.concat([initial, wide_data], how="vertical_relaxed")
    df = observation_data.with_columns(
        _datetime_expr(observation_data, "anchor_time").alias("__anchor_dt"),
        _datetime_expr(observation_data, "support_start").alias("__support_start_dt"),
        _datetime_expr(observation_data, "support_end").alias("__support_end_dt"),
        pl.col("support_kind").alias("__support_kind"),
    ).drop_nulls(subset=["__anchor_dt"])
    if df.is_empty():
        return wide_data

    interval_df = df.filter(pl.col("__support_kind") == "interval")
    if interval_df.is_empty():
        return wide_data

    origin = ObservationInstant(time_origin)
    boundary_times = (
        pl.concat(
            [
                interval_df.select(pl.col("__anchor_dt").alias("__boundary_dt")),
                interval_df.select(pl.col("__support_start_dt").alias("__boundary_dt")),
                interval_df.select(pl.col("__support_end_dt").alias("__boundary_dt")),
            ],
            how="vertical_relaxed",
        )
        .drop_nulls(subset=["__boundary_dt"])
        .unique()
        .sort("__boundary_dt")
        .with_columns(ModelTime.bind_column(pl.col("__boundary_dt"), origin).alias("time"))
        .select("time")
    )

    if boundary_times.is_empty():
        return wide_data

    current_times = wide_data.select(pl.col("time").cast(pl.Float64).alias("time"))
    missing_times = boundary_times.join(current_times, on="time", how="anti")
    if missing_times.is_empty():
        return wide_data.sort("time")

    filler_columns = [
        pl.lit(None, dtype=wide_data.schema.get(name, pl.Float64)).alias(name)
        for name in wide_data.columns
        if name != "time"
    ]
    missing_rows = missing_times.with_columns(filler_columns).select(wide_data.columns)
    return pl.concat([wide_data, missing_rows], how="vertical_relaxed").sort("time")


def extract_numeric_column_values(X: pl.DataFrame, column: str) -> np.ndarray:
    """Extract one manifest column as float64, dropping nulls but not infinities."""
    values = X.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().to_numpy()
    return values[~np.isnan(values)]


def validate_observation_support(
    compiled_dynamical_model: CompiledDynamicalModel, X: pl.DataFrame
) -> ObservationPreflightFailure | None:
    """Reject likelihoods whose support is incompatible with observed data."""
    from nof1_causal_lab.artifacts.model.likelihood import (
        BernoulliLogitsLawSpec,
        BernoulliProbsLawSpec,
        BetaLawSpec,
        GammaLawSpec,
        NegativeBinomial2LawSpec,
        PoissonLawSpec,
    )

    issues: list[str] = []
    for observation in compiled_dynamical_model.observations:
        column, law, family = observation.name, observation.law, observation.law.family
        values = extract_numeric_column_values(X, column)
        if values.size == 0:
            continue
        if np.any(~np.isfinite(values)):
            issues.append(
                f"- '{column}' uses {family.value} emission but observed data contain non-finite values"
            )
            continue

        if isinstance(law, GammaLawSpec):
            invalid, description = values <= 0.0, "strictly positive"
        elif isinstance(law, BetaLawSpec):
            invalid, description = (values <= 0.0) | (values >= 1.0), "strictly inside (0, 1)"
        elif isinstance(law, (BernoulliLogitsLawSpec, BernoulliProbsLawSpec)):
            invalid, description = ~np.isin(values, (0.0, 1.0)), "binary 0/1"
        elif isinstance(law, (PoissonLawSpec, NegativeBinomial2LawSpec)):
            invalid, description = (
                (values < 0.0) | ~np.isclose(values, np.rint(values), atol=1e-6),
                "nonnegative integer",
            )
        else:
            continue
        if not np.any(invalid):
            continue

        bad_values = values[invalid]
        issues.append(
            f"- '{column}' uses {family.value} emission but {bad_values.size}/{values.size} "
            f"observations are outside support ({description}; "
            f"min={float(values.min()):.3g}, max={float(values.max()):.3g})"
        )

    if issues:
        return ObservationPreflightFailure.rejected(
            "Observation support check failed:\n" + "\n".join(issues)
        )

    return None


def recorded_observation_support(
    times: np.ndarray,
    variables: tuple[ResolvedObservationSpec, ...],
    starts: np.ndarray,
    ends: np.ndarray,
) -> ObservationSupportRuntime | ObservationPreflightFailure:
    """Parse the production observation coordinates, deriving only execution weights."""
    names = tuple(variable.id for variable in variables)
    kinds = tuple(variable.support_kind.value for variable in variables)
    window_plan = compile_window_plan(times, starts, ends, kinds, names)
    if isinstance(window_plan, ObservationPreflightFailure):
        return window_plan
    return ObservationSupportRuntime.assembled(
        window_plan=window_plan,
        manifest_names=names,
        support_kinds=kinds,
        summary_operators=tuple(variable.summary_operator.value for variable in variables),
        anchor_policies=tuple(variable.anchor_policy.value for variable in variables),
        observation_windows=tuple(str(variable.observation_window) for variable in variables),
        support_start_times=starts,
        support_end_times=ends,
    )


def simulation_observation_support(
    compiled_dynamical_model: CompiledDynamicalModel,
    times: np.ndarray,
    *,
    time_origin: datetime,
) -> ObservationSupportRuntime:
    """Schedule declared indicators on a simulation grid, omitting unavailable prehistory."""
    ordered = compiled_dynamical_model.observations
    names = [indicator.name for indicator in ordered]
    kinds: list[str | None] = [indicator.support.support_kind.value for indicator in ordered]
    windows: list[str | None] = [indicator.observation_window.source for indicator in ordered]
    starts = np.broadcast_to(times[:, None], (len(times), len(ordered))).copy()
    ends = starts.copy()
    origin = ObservationInstant(time_origin)
    instants = pl.DataFrame(
        {"time": [ModelTime(float(time)).at(origin).value.replace(tzinfo=None) for time in times]}
    )
    for i, observation in enumerate(ordered):
        kind = observation.support.support_kind.value
        if kind == "interval":
            start, end = observation_window_bounds(
                pl.col("time"),
                observation.observation_window,
                anchor_policy=AnchorPolicy.SUPPORT_END,
            )
            starts[:, i], ends[:, i] = (
                instants.select(
                    ModelTime.bind_column(start, origin).alias("start"),
                    ModelTime.bind_column(end, origin).alias("end"),
                )
                .to_numpy()
                .T
            )
            absent = starts[:, i] < times[0] - 1e-8
            starts[absent, i] = np.nan
            ends[absent, i] = np.nan
    window_plan = compile_window_plan(times, starts, ends, kinds, names)
    if isinstance(window_plan, ObservationPreflightFailure):
        raise RuntimeError(f"Generated simulation support is inconsistent: {window_plan.message}")
    return ObservationSupportRuntime.assembled(
        window_plan=window_plan,
        manifest_names=tuple(names),
        support_kinds=tuple(kinds),
        summary_operators=tuple(
            [indicator.support.summary_operator.value for indicator in ordered]
        ),
        anchor_policies=tuple([indicator.support.anchor_policy.value for indicator in ordered]),
        observation_windows=tuple(windows),
        support_start_times=starts,
        support_end_times=ends,
    )
