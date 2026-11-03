"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from grassmoth.artifacts.observations.dataset import ObservationDataset, SelectedObservations
from grassmoth.artifacts.observations.time import ObservationInstant, model_time_indices
from grassmoth.artifacts.scenarios import StateAssignment
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.counterfactual.orchestration import ResolvedIntervention
from grassmoth.models.ssm.preparation.observations import (
    compile_observation_support_runtime,
    validate_observation_support,
)
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth.observations.rows import observation_grid, pivot_to_wide

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from grassmoth.artifacts.identity import IndicatorId
    from grassmoth.compute.programs import ProgramCompiler
    from grassmoth.models.ssm.compile.inputs import (
        CompiledDynamicalModel,
    )
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime


@dataclass(frozen=True)
class BoundPanel:
    """Immutable observations and exogenous bindings on one compiled model's axes."""

    compiled_dynamical_model: CompiledDynamicalModel
    time_origin: datetime
    values: jax.Array
    times: jax.Array
    observation_support: ObservationSupportRuntime
    input_values: jax.Array
    input_events: tuple[ResolvedIntervention, ...]

    @property
    def indicator_ids(self) -> tuple[IndicatorId, ...]:
        """Observation identities in the column order used by this bound panel."""
        return tuple(observation.id for observation in self.compiled_dynamical_model.observations)

    @property
    def observation_mask(self) -> jax.Array:
        """Boolean mask of recorded panel entries, with NaN values treated as missing."""
        return ~jnp.isnan(self.values)

    @property
    def observations(self) -> jax.Array:
        """Only endogenous measurements contribute to the fitted likelihood."""
        input_columns = jnp.asarray(
            tuple(
                self.compiled_dynamical_model.states[observation.state_index].is_input
                for observation in self.compiled_dynamical_model.observations
            )
        )
        return jnp.where(input_columns[None, :], jnp.nan, self.values)


def reading_level(
    value: float, support_start: float, support_end: float, summary_operator: str
) -> float:
    """An input reading's constant level: its value, or a count or sum per model day."""
    if summary_operator in {"count", "sum"}:
        return value / (support_end - support_start)
    return value


def input_trajectory_events(
    compiled_dynamical_model: CompiledDynamicalModel,
    *,
    time_origin: datetime,
    start: float,
    end: float,
) -> tuple[ResolvedIntervention, ...] | ObservationPreflightFailure:
    """Resolve model-owned step functions on the execution clock, holding their final value."""
    events = []
    for trajectory in compiled_dynamical_model.input_trajectories:
        origin = trajectory.layout.time_origin
        offset = (
            ObservationInstant(origin).relative_to(ObservationInstant(time_origin)).days
            if not isinstance(origin, str)
            else 0.0
        )
        times = tuple(time + offset for time in trajectory.layout.time_points)
        index = int(np.searchsorted(times, start, side="right")) - 1
        if index < 0:
            return ObservationPreflightFailure.rejected(
                f"Input {compiled_dynamical_model.states[trajectory.index].name!r} has no value at the start of the requested history"
            )
        points = (
            (start, trajectory.values[index]),
            *(
                (time, value)
                for time, value in zip(times, trajectory.values, strict=True)
                if start < time <= end
            ),
        )
        events.extend(
            ResolvedIntervention(
                trajectory.index,
                StateAssignment(target=trajectory.construct_id, time=time, value=value),
            )
            for time, value in points
        )
    return tuple(sorted(events, key=lambda event: (event.spec.time, event.index)))


def input_trajectory_values(
    compiled_dynamical_model: CompiledDynamicalModel,
    times: np.ndarray | jax.Array | Sequence[float],
    events: tuple[ResolvedIntervention, ...],
) -> jnp.ndarray:
    """Expand deterministic input paths on the execution grid; endogenous states stay unknown."""
    grid = np.asarray(times)
    values = np.full(
        (len(grid), numeric.n_states(compiled_dynamical_model)), np.nan, dtype=grid.dtype
    )
    for event in events:
        values[grid >= np.asarray(event.spec.time, dtype=grid.dtype), event.index] = (
            event.spec.value
        )
    return jnp.asarray(values)


def _standardize_manifest_columns(
    wide_data: pl.DataFrame,
    manifest_cols: Sequence[str],
    manifest_standardized: Sequence[bool],
) -> pl.DataFrame:
    """Apply deterministic standardization to manifest columns marked standardized.

    Flagged columns become (y - mean) / sd so their link-scale spread is exactly 1,
    matching the standardized-latent convention the priors are authored under. When
    sd is 0 or undefined every centered value is already 0, so any divisor yields
    identical data; 1 is the canonical completion, not a fallback.
    """
    if not any(manifest_standardized):
        return wide_data

    standardized_exprs = []
    for manifest_name, standardized in zip(manifest_cols, manifest_standardized, strict=False):
        base_expr = pl.col(manifest_name).cast(pl.Float64, strict=False)
        if standardized:
            centered_expr = base_expr - base_expr.mean()
            scale_expr = base_expr.std()
            standardized_exprs.append(
                (
                    centered_expr
                    / pl.when(scale_expr > 0.0).then(scale_expr).otherwise(pl.lit(1.0))
                ).alias(manifest_name)
            )
        else:
            standardized_exprs.append(base_expr.alias(manifest_name))
    passthrough = [col_name for col_name in wide_data.columns if col_name not in set(manifest_cols)]
    return wide_data.select(*passthrough, *standardized_exprs)


@dataclass(frozen=True)
class PanelPreparationFailure:
    """The supplied history cannot be bound to the compiled model."""

    message: str


def prepare_fit_inputs(
    compiled_dynamical_model: CompiledDynamicalModel,
    wide_data: pl.DataFrame,
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[str, ...], pl.DataFrame]:
    """Extract numerical observations on the explicit bound time grid."""
    manifest_cols = numeric.observation_names(compiled_dynamical_model)
    standardized_data = _standardize_manifest_columns(
        wide_data, manifest_cols, numeric.observation_standardized(compiled_dynamical_model)
    )
    observations = jnp.array(standardized_data.select(manifest_cols).to_numpy(), dtype=jnp.float32)
    times = jnp.array(standardized_data["time"].to_numpy(), dtype=jnp.float32)
    return observations, times, manifest_cols, standardized_data


def bind_panel(
    observations: ObservationDataset,
    *,
    compiled_dynamical_model: CompiledDynamicalModel,
    compiler: ProgramCompiler,
) -> BoundPanel | PanelPreparationFailure:
    """Bind admitted histories to the model and its numerical execution grid."""
    observation_selection = observations.select(
        tuple(item.observation for item in compiled_dynamical_model.observations)
    )
    if not isinstance(observation_selection, SelectedObservations):
        return PanelPreparationFailure(observation_selection.message)
    time_origin = observations.time_origin
    projected = project_observation_data(
        observation_selection, compiled_dynamical_model=compiled_dynamical_model
    )
    if isinstance(projected, ObservationPreflightFailure):
        return PanelPreparationFailure(projected.message)
    wide = projected
    support_failure = validate_observation_support(compiled_dynamical_model, wide)
    if support_failure is not None:
        return PanelPreparationFailure(support_failure.message)
    events = input_trajectory_events(
        compiled_dynamical_model,
        time_origin=time_origin,
        start=float(wide["time"][0]),
        end=float(wide["time"][-1]),
    )
    if isinstance(events, ObservationPreflightFailure):
        return PanelPreparationFailure(events.message)
    if events:
        grid = observation_grid(observation_selection, tuple(event.spec.time for event in events))
        labels = {str(item.id): item.name for item in compiled_dynamical_model.observations}
        wide = pivot_to_wide(observation_selection, grid).rename(labels)
        indices = model_time_indices(grid, np.asarray([event.spec.time for event in events]))
        events = tuple(
            ResolvedIntervention(event.index, event.spec.revised(time=float(grid[index])))
            for event, index in zip(events, indices, strict=True)
        )
    values, times, names, wide = prepare_fit_inputs(compiled_dynamical_model, wide)
    support = compile_observation_support_runtime(observation_selection, wide, names)
    if isinstance(support, ObservationPreflightFailure):
        return PanelPreparationFailure(support.message)
    input_values = input_trajectory_values(compiled_dynamical_model, times, events)
    channels = tuple(
        index
        for index, observation in enumerate(compiled_dynamical_model.observations)
        if compiled_dynamical_model.states[observation.state_index].is_input
    )
    if channels:
        from grassmoth.models.ssm.execution.observation_operator import (
            compile_observation_operator,
        )

        operator = compile_observation_operator(support, held_channels=channels)
        response = input_values[
            :, jnp.asarray([item.state_index for item in compiled_dynamical_model.observations])
        ]
        expected, _ = compiler(operator.project_response_trajectory, response).call(response)
        recorded_values, expected_values = np.asarray(values), np.asarray(expected)
        for channel in channels:
            recorded = recorded_values[:, channel]
            consistent = np.isnan(recorded) | np.isclose(
                recorded, expected_values[:, channel], rtol=1e-6, atol=1e-6
            )
            if not consistent.all():
                return PanelPreparationFailure(
                    f"Exact readings for {compiled_dynamical_model.observations[channel].name!r} conflict with its deterministic trajectory law"
                )
    return BoundPanel(
        compiled_dynamical_model,
        time_origin,
        values,
        times,
        support,
        input_values,
        events,
    )


def project_observation_data(
    observations: SelectedObservations,
    *,
    compiled_dynamical_model: CompiledDynamicalModel,
) -> pl.DataFrame | ObservationPreflightFailure:
    """Bind selected identities to display columns on the tolerance-merged execution grid."""
    missing = [
        series.observation.id
        for series in observations.series
        if series.history.times.layout[0][0] == 0
    ]
    if missing:
        return ObservationPreflightFailure.rejected(
            "Prepared observations are missing model indicators: " + ", ".join(missing)
        )
    grid = observation_grid(observations)
    if not len(grid):
        return ObservationPreflightFailure.rejected("Cannot bind an empty observation panel")
    labels = {str(item.id): item.name for item in compiled_dynamical_model.observations}
    return pivot_to_wide(observations, grid).rename(labels)
