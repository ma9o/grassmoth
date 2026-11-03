"""Shared observation metadata helpers for SSM model preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import polars as pl

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.observations.semantics import AnchorPolicy
from grassmoth.artifacts.observations.time import (
    OBSERVATION_TIME_TOLERANCE,
    ModelTime,
    ObservationInstant,
    model_time_indices,
)
from grassmoth.models.ssm.compile.observation_windows import (
    CompiledWindowPlan,
    compile_window_plan,
)
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from grassmoth.artifacts.observations.dataset import SelectedObservations
    from grassmoth.artifacts.observations.spec import ResolvedObservationSpec
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel


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


def compile_observation_support_runtime(
    observations: SelectedObservations,
    wide_data: pl.DataFrame,
    manifest_names: Sequence[str],
) -> ObservationSupportRuntime | ObservationPreflightFailure:
    """Compile already admitted support onto the selected numerical execution grid."""
    times = wide_data["time"].to_numpy()
    starts = np.full((len(times), len(observations.series)), np.nan)
    ends = np.full_like(starts, np.nan)
    for column, series in enumerate(observations.series):
        indices = model_time_indices(times, series.history.times.values)
        starts[indices, column] = vector_values(series.history.support_start)
        ends[indices, column] = vector_values(series.history.support_end)
    variables = tuple(series.observation for series in observations.series)
    kinds = tuple(variable.support_kind.value for variable in variables)
    plan = compile_window_plan(times, starts, ends, kinds, manifest_names)
    if isinstance(plan, ObservationPreflightFailure):
        return plan
    return ObservationSupportRuntime.assembled(
        window_plan=plan,
        manifest_names=manifest_names,
        support_kinds=kinds,
        summary_operators=tuple(variable.summary_operator.value for variable in variables),
        anchor_policies=tuple(variable.anchor_policy.value for variable in variables),
        observation_windows=tuple(str(variable.observation_window) for variable in variables),
        support_start_times=starts,
        support_end_times=ends,
    )


def extract_numeric_column_values(X: pl.DataFrame, column: str) -> np.ndarray:
    """Extract one manifest column as float64, dropping nulls but not infinities."""
    values = X.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().to_numpy()
    return values[~np.isnan(values)]


def validate_observation_support(
    compiled_dynamical_model: CompiledDynamicalModel, X: pl.DataFrame
) -> ObservationPreflightFailure | None:
    """Reject likelihoods whose support is incompatible with observed data."""
    from grassmoth.artifacts.model.likelihood import (
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
            start, end = ObservationInstant.support_bounds(
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
            absent = starts[:, i] < times[0] - OBSERVATION_TIME_TOLERANCE
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
