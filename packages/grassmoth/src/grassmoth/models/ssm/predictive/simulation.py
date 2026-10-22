"""Forward histories from current joint laws, with separate reusable measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Self

import jax.numpy as jnp
import jax.random as random
import numpy as np

from grassmoth.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from grassmoth.artifacts.model.expressions import hill_applications
from grassmoth.artifacts.observations.time import ModelTime, ObservationInstant
from grassmoth.artifacts.reports.checks import (
    NotEvaluated,
    PredictiveAssessment,
    PredictiveSubject,
)
from grassmoth.models.diagnostics.history_comparison import measure_predictive_checks
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.counterfactual.orchestration import ResolvedIntervention
from grassmoth.models.ssm.diagnostics.simulation import (
    ConstructSimulationTarget,
    DesignInfo,
    measure_construct_simulation,
)
from grassmoth.models.ssm.predictive.parameters import sample_model_laws
from grassmoth.models.ssm.predictive.registry_runtime import (
    predictive_keys,
    simulate_latent_histories,
    simulate_predictive_draws,
)
from grassmoth.models.ssm.preparation.binding import BoundPanel, input_trajectory_events
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth.observations.rows import observation_window_boundaries

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.artifacts.reports.posterior_diagnostics import PosteriorPredictiveChecks
    from grassmoth.artifacts.scenarios import StateAssignment
    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.predictive.types import PredictiveDraws


SIMULATION_DRAWS = 100
SIMULATION_SEED = 0


@dataclass(frozen=True)
class SimulationBatch:
    """One generated batch and its resolved observation coordinates."""

    times: tuple[float, ...]
    prediction: PredictiveDraws
    observations: jnp.ndarray | None
    measurement_design: DesignInfo
    time_origin: datetime

    @classmethod
    def from_draws(
        cls,
        prediction: PredictiveDraws,
        measurement_design: DesignInfo,
        *,
        time_origin: datetime,
    ) -> Self:
        """Bind existing draws to the measurement design that owns their time coordinates."""
        return cls(
            tuple(float(time) for time in measurement_design.t_grid),
            prediction,
            None,
            measurement_design,
            time_origin,
        )


def _time_grid(
    compiled_dynamical_model: CompiledDynamicalModel,
    start: float,
    end: float,
    assignments: tuple[StateAssignment, ...],
    time_origin: datetime,
) -> np.ndarray:
    times = {
        float(value)
        for value in np.linspace(
            start, end, max(1, ceil((end - start) / compiled_dynamical_model.clock_days)) + 1
        )
    }
    times.update(event.time for event in assignments)
    origin = ObservationInstant(time_origin)
    times.update(
        ObservationInstant(instant).relative_to(origin).days
        for window in {item.observation_window for item in compiled_dynamical_model.observations}
        for instant in observation_window_boundaries(
            window, ModelTime(start).at(origin).value, ModelTime(end).at(origin).value
        )
    )
    return np.asarray(sorted(times))


def _window_input_events(
    input_events: tuple[ResolvedIntervention, ...], start: float
) -> tuple[ResolvedIntervention, ...]:
    """Each input's level held at the window start, then its readings inside the window."""
    held: dict[int, ResolvedIntervention] = {}
    for event in sorted(input_events, key=lambda event: event.spec.time):
        if event.spec.time <= start:
            held[event.index] = event
    return (
        *(
            ResolvedIntervention(index=event.index, spec=event.spec.revised(time=start))
            for event in held.values()
        ),
        *(event for event in input_events if event.spec.time > start),
    )


def generate_simulation_batch(
    source: CompiledDynamicalModel | BoundPanel,
    *,
    compute_budget: ComputeBudget,
    start: float,
    end: float,
    assignments: tuple[StateAssignment, ...] = (),
    times: np.ndarray | jnp.ndarray | None = None,
    draws: int = SIMULATION_DRAWS,
    seed: int = SIMULATION_SEED,
    time_origin: datetime,
) -> SimulationBatch | ObservationPreflightFailure:
    """Sample current laws once and always generate nonlinear stochastic paths and emissions.

    The window and intervention assignments are in model days. Internal check callers
    may supply a prepared observation grid and execution budget; otherwise the grid
    follows the model clock and the assignment boundaries.
    """
    if isinstance(source, BoundPanel):
        time_origin = source.time_origin
        compiled_dynamical_model = source.compiled_dynamical_model
        times = source.times
        observations = source.values
        support = source.observation_support
    else:
        compiled_dynamical_model = source
        observations = None
        support = None
    indicator_ids = tuple(numeric.observation_ids(compiled_dynamical_model))
    state_ids = tuple(numeric.state_ids(compiled_dynamical_model))
    time_points: tuple[float, ...] = next(
        (law.layout.time_points for law in compiled_dynamical_model.laws if law.layout.constructs),
        (),
    )
    if isinstance(source, BoundPanel):
        input_events = source.input_events
    else:
        anchor_index = int(np.searchsorted(time_points, start, side="right")) - 1
        history_start = time_points[anchor_index] if time_points and anchor_index >= 0 else 0.0
        events = input_trajectory_events(
            compiled_dynamical_model, time_origin=time_origin, start=history_start, end=end
        )
        if isinstance(events, ObservationPreflightFailure):
            return events
        input_events = events
    laws = sample_model_laws(
        compiled_dynamical_model, draws=draws, key=predictive_keys(seed).parameters
    )
    grid = (
        _time_grid(compiled_dynamical_model, start, end, assignments, time_origin)
        if times is None
        else np.asarray(times)
    )
    if len(grid) < 2 or grid[0] != start or grid[-1] != end or np.any(np.diff(grid) <= 0):
        raise ValueError("The prepared simulation grid must increase from start through end")
    grid = np.asarray(
        sorted(
            {
                *grid,
                *(event.spec.time for event in input_events if start < event.spec.time < end),
            }
        )
    )
    initial = None
    if laws.latent_paths is None:
        if start < 0:
            raise ValueError("Simulation cannot start before the initial law at model day zero")
        if start > 0:
            history_events = tuple(event for event in input_events if event.spec.time <= start)
            history_grid = sorted(
                {
                    0.0,
                    start,
                    *(event.spec.time for event in history_events if 0 < event.spec.time < start),
                }
            )
            history, _, _ = simulate_latent_histories(
                compiled_dynamical_model,
                laws.parameters,
                jnp.asarray(history_grid),
                random.fold_in(predictive_keys(seed).latents, 1),
                None,
                (),
                history_events,
                compute_budget=compute_budget,
            )
            initial = history[:, -1, :]
    if laws.latent_paths is not None:
        index = int(np.searchsorted(time_points, start, side="right")) - 1
        if index < 0:
            raise ValueError("Simulation cannot start before the model's first retained state")
        initial = jnp.zeros((draws, len(state_ids)), dtype=laws.latent_paths.dtype)
        initial = initial.at[
            :, jnp.asarray([state_ids.index(identity) for identity in laws.state_ids])
        ].set(laws.latent_paths[:, index, :])
        anchor = time_points[index]
        if anchor < start:
            # Advance a jointly drawn state to an unrepresented start time; never interpolate
            # retained paths or condition on a second independently sampled state.
            history_events = tuple(event for event in input_events if event.spec.time <= start)
            history_grid = sorted(
                {
                    anchor,
                    start,
                    *(
                        event.spec.time
                        for event in history_events
                        if anchor < event.spec.time < start
                    ),
                }
            )
            history, _, _ = simulate_latent_histories(
                compiled_dynamical_model,
                laws.parameters,
                jnp.asarray(history_grid),
                random.fold_in(predictive_keys(seed).latents, 1),
                initial,
                (),
                history_events,
                compute_budget=compute_budget,
            )
            initial = history[:, -1, :]
    from grassmoth.models.ssm.preparation.observations import simulation_observation_support

    if support is None:
        support = simulation_observation_support(
            compiled_dynamical_model, grid, time_origin=time_origin
        )
    interventions = []
    for event in assignments:
        if event.target not in state_ids:
            raise ValueError(f"Intervention target is not a model state: {event.target}")
        interventions.append(ResolvedIntervention(index=state_ids.index(event.target), spec=event))
    prediction = simulate_predictive_draws(
        compiled_dynamical_model,
        laws.parameters,
        jnp.asarray(grid),
        compute_budget=compute_budget,
        seed=seed,
        initial_states=initial,
        interventions=tuple(interventions),
        input_events=_window_input_events(input_events, start),
        observation_support=support,
        observation_mask=None if observations is None else ~jnp.isnan(observations),
    )
    indices: dict[str, np.ndarray] = {
        identity: np.flatnonzero(
            np.asarray(prediction.trajectory.observations_mask[:, :, i]).any(axis=0)
        )
        if observations is None
        else np.flatnonzero(np.isfinite(observations[:, i]))
        for i, identity in enumerate(indicator_ids)
    }
    values: dict[str, np.ndarray] = {
        identity: np.asarray([])
        if observations is None
        else np.asarray(observations[indices[identity], i])
        for i, identity in enumerate(indicator_ids)
    }
    return SimulationBatch(
        tuple(float(t) for t in grid),
        prediction,
        observations,
        DesignInfo(
            t_grid=jnp.asarray(grid),
            manifest_ids=indicator_ids,
            obs_index_by_indicator=indices,
            values_by_indicator=values,
            n_draws=draws,
            seed=seed,
            observation_support=support,
        ),
        time_origin=time_origin,
    )


def measure_simulation_batch(
    compiled_dynamical_model: CompiledDynamicalModel,
    batch: SimulationBatch,
    *,
    groups: tuple[str, ...] = ("dynamics", "measurement"),
    edge_contrasts: ComputeBudget | None = None,
) -> tuple[tuple[PredictiveAssessment, ...], PosteriorPredictiveChecks | None]:
    """All selected reducers consume the same generated paths."""
    prediction = batch.prediction
    observations = batch.observations
    measurement_design = batch.measurement_design
    indicator_ids = tuple(numeric.observation_ids(compiled_dynamical_model))
    state_ids = tuple(numeric.state_ids(compiled_dynamical_model))
    targets = {
        **{state.name: ConstructRef(id=state.id) for state in compiled_dynamical_model.states},
        **{
            observation.id: IndicatorRef(id=observation.id)
            for observation in compiled_dynamical_model.observations
        },
        **{
            f"{compiled_dynamical_model.states[source].name}->{state.name}": EdgeRef(id=identity)
            for state in compiled_dynamical_model.states
            for source, identity in state.incoming_edges
        },
    }
    findings: list[PredictiveAssessment] = []
    components = compiled_dynamical_model.dynamics.spec.components if "dynamics" in groups else ()
    for state_index, identity in enumerate(
        state_ids if set(groups) & {"dynamics", "measurement"} else ()
    ):
        if compiled_dynamical_model.states[state_index].is_input:
            continue
        incoming = [
            component
            for component in components
            if component.target == state_index and component.edge_owned
        ]
        parents = sorted(
            {
                source
                for component in incoming
                for source in component.sources
                if source != state_index
            }
        )
        hills = {
            source
            for component in incoming
            if any(hill_applications(component.expression))
            for source in component.sources
            if source != state_index
        }
        target = ConstructSimulationTarget(
            construct=compiled_dynamical_model.states[state_index],
            edge_parents=tuple(compiled_dynamical_model.states[source].name for source in parents),
            hill_parents=tuple(
                compiled_dynamical_model.states[source].name for source in sorted(hills)
            ),
        )
        measured = measure_construct_simulation(
            compiled_dynamical_model,
            prediction,
            measurement_design,
            target,
            dynamics="dynamics" in groups,
            measurement="measurement" in groups,
            edge_contrasts=edge_contrasts,
        )
        findings.extend(result.finding(identity, targets[result.target]) for result in measured)
    modeled_columns = [
        index
        for index, identity in enumerate(indicator_ids)
        if not compiled_dynamical_model.states[
            compiled_dynamical_model.observations[index].state_index
        ].is_input
    ]
    nonfinite = bool(
        np.any(
            np.asarray(prediction.trajectory.observations_mask[:, :, modeled_columns])
            & ~np.isfinite(prediction.trajectory.observations[:, :, modeled_columns])
        )
    )
    checks = (
        None
        if observations is None or "data_comparison" not in groups or nonfinite
        else measure_predictive_checks(
            prediction.trajectory.observations[:, :, modeled_columns],
            observations[:, modeled_columns],
            tuple(indicator_ids[index] for index in modeled_columns),
            times=batch.times,
            time_origin=batch.time_origin,
            standardized=tuple(
                compiled_dynamical_model.observations[index].standardized
                for index in modeled_columns
            ),
        )
    )
    if "data_comparison" in groups and checks is None:
        findings.append(
            NotEvaluated(
                code="data_comparison",
                subject=PredictiveSubject(target="observations"),
                reason="NONFINITE_PATHS" if nonfinite else "COMPARISON_INPUTS_MISSING",
                detail="Predictive observations contain non-finite values."
                if nonfinite
                else "Observed data are required to assess predictive calibration.",
            )
        )
    return tuple(findings), checks
