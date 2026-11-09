"""Compile analytical particle-memory dimensions before remote execution."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import numpyro.distributions as dist

from grassmoth.artifacts.observations.dataset import SelectedObservations
from grassmoth.artifacts.sampling import SamplerSpec
from grassmoth.compute.particle_memory import ParticleMgradMemoryFailure, ParticleMgradMemoryShape
from grassmoth.models.ssm.compile.inputs import CompiledFitInputs, compile_ssm_inputs_from_model
from grassmoth.models.ssm.inference.conditioning import compile_exact_state_constraints
from grassmoth.models.ssm.inference.samplers.sweep_plan import SWEEP_BLOCK_SIZE
from grassmoth.models.ssm.preparation.binding import (
    PanelPreparationFailure,
    input_trajectory_events,
    input_trajectory_values,
    prepare_fit_inputs,
    project_observation_data,
    select_model_observations,
)
from grassmoth.models.ssm.preparation.observations import validate_observation_support
from grassmoth.models.ssm.preparation.preflight import (
    ObservationPreflightFailure,
    validate_sampler_support,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.models.structure import StructuralSelection


def _unconstrained_size(prior: dist.Distribution, shape: tuple[int, ...]) -> int:
    """Use NumPyro's shape transforms for the fully non-centred prior coordinates.

    AutoReparam unwraps transformed laws to their base law; location-scale
    reparameterization preserves shape. ProjectedNormal uses its ambient Normal
    coordinates. No prior draw, density, initialization or target is evaluated.
    """
    if isinstance(prior, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
        return _unconstrained_size(prior.base_dist, shape)
    if isinstance(prior, dist.TransformedDistribution):
        for transform in reversed(prior.transforms):
            shape = transform.inverse_shape(shape)
        return _unconstrained_size(prior.base_dist, shape)
    if isinstance(prior, dist.ProjectedNormal):
        return prod(shape)
    return int(prod(dist.transforms.biject_to(prior.support).inverse_shape(shape)))


def particle_mgrad_memory_shape(
    selection: StructuralSelection,
    observations: ObservationDataset,
) -> ParticleMgradMemoryShape | ParticleMgradMemoryFailure:
    """Reuse model compilation, exact observation grids and fixed-coordinate ownership.

    This compiles structural metadata only: no numerical target, prior sampling,
    IEKS solve, JIT program, GPU probe or fitting action is constructed. Current
    Particle-mGRAD rejects interval windows and replay, so no unaccounted window
    state is silently treated as an ordinary point-observation dimension.
    """
    inputs = compile_ssm_inputs_from_model(selection)
    if not isinstance(inputs, CompiledFitInputs):
        return ParticleMgradMemoryFailure(message=inputs.message)
    compiled = inputs.compiled_dynamical_model
    failure = validate_sampler_support(compiled, SamplerSpec())
    if failure is not None:
        return ParticleMgradMemoryFailure(message=failure.message)
    selected = observations.select(tuple(item.observation for item in compiled.observations))
    if not isinstance(selected, SelectedObservations):
        return ParticleMgradMemoryFailure(message=selected.message)
    selected = select_model_observations(selected, compiled.time)
    if isinstance(selected, PanelPreparationFailure):
        return ParticleMgradMemoryFailure(message=selected.message)
    wide = project_observation_data(selected, compiled_dynamical_model=compiled)
    if isinstance(wide, ObservationPreflightFailure):
        return ParticleMgradMemoryFailure(message=wide.message)
    failure = validate_observation_support(compiled, wide)
    if failure is not None:
        return ParticleMgradMemoryFailure(message=failure.message)
    events = input_trajectory_events(
        compiled,
        time_origin=selected.time_origin,
        start=float(wide["time"][0]),
        end=float(wide["time"][-1]),
    )
    if isinstance(events, ObservationPreflightFailure):
        return ParticleMgradMemoryFailure(message=events.message)
    values, times, _, _ = prepare_fit_inputs(compiled, wide)
    input_values = input_trajectory_values(compiled, times, events)
    constraints = compile_exact_state_constraints(compiled, values, input_values=input_values)
    state_space = compiled.state_space
    free = np.zeros((len(times), state_space.size), dtype=bool)
    free[0, state_space.initial_path_coordinates] = True
    free[1:, state_space.transition] = True
    if constraints is not None:
        free &= np.asarray(constraints.free_mask)
    free_static = sum(
        index in state_space.initial
        and (constraints is None or bool(constraints.free_mask[0, index]))
        for index in state_space.static
    )
    parameters = (
        sum(
            _unconstrained_size(prior, prior.batch_shape + prior.event_shape)
            for prior in inputs.prior_runtime_bundle.priors.values()
        )
        + free_static
    )
    return ParticleMgradMemoryShape(
        time_points=len(times),
        state_dimensions=state_space.size,
        initial_dimensions=len(state_space.initial_path_coordinates),
        transition_dimensions=len(state_space.transition),
        parameter_dimensions=parameters,
        observation_dimensions=len(compiled.observations),
        free_path_coordinates=int(free.sum()),
        initial_covariance=state_space.initial_covariance,
        transition_covariance=state_space.transition_covariance,
        sweep_block_size=SWEEP_BLOCK_SIZE,
    )
