"""Total numerical accessors over resolved compiler-owned facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import ConstructId, IndicatorId
    from grassmoth.artifacts.probability.families import DistributionFamily
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.structure.blocks import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseBlockSpec,
        T0CholBlockSpec,
    )


def state_ids(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[ConstructId, ...]:
    """Return construct IDs in the model's compiled state-axis order."""
    return tuple(state.id for state in compiled_dynamical_model.states)


def state_names(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[str, ...]:
    """Return display labels in the model's compiled state-axis order."""
    return tuple(state.name for state in compiled_dynamical_model.states)


def observation_ids(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[IndicatorId, ...]:
    """Return observation IDs in the model's compiled channel order."""
    return tuple(observation.id for observation in compiled_dynamical_model.observations)


def observation_names(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[str, ...]:
    """Return observation display labels in the model's compiled channel order."""
    return tuple(observation.name for observation in compiled_dynamical_model.observations)


def observation_families(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[DistributionFamily, ...]:
    """Return channel-aligned likelihood families from the compiled observation laws."""
    return tuple(observation.law.family for observation in compiled_dynamical_model.observations)


def observation_level_counts(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[int, ...]:
    """Return channel-aligned category counts, with zero for observations without declared levels."""
    return tuple(len(observation.levels) for observation in compiled_dynamical_model.observations)


def observation_standardized(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[bool, ...]:
    """Return channel-aligned flags identifying observations represented on a standardized scale."""
    return tuple(observation.standardized for observation in compiled_dynamical_model.observations)


def categorical_anchors(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[bool, ...]:
    """Return channel-aligned flags identifying categorical measurement anchors."""
    return tuple(
        observation.categorical_anchor for observation in compiled_dynamical_model.observations
    )


def time_invariant_mask(compiled_dynamical_model: CompiledDynamicalModel) -> np.ndarray:
    """Mark compiled state coordinates with time-invariant dynamics."""
    return np.asarray(
        [state.time_invariant for state in compiled_dynamical_model.states], dtype=bool
    )


def input_mask(compiled_dynamical_model: CompiledDynamicalModel) -> np.ndarray:
    """Mark state coordinates supplied by deterministic input laws."""
    return np.asarray([state.is_input for state in compiled_dynamical_model.states], dtype=bool)


def diffusion_families(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[DistributionFamily, ...]:
    """Return compiled innovation families in state-axis order."""
    return tuple(state.innovation_family for state in compiled_dynamical_model.states)


def static_factor_ids(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[ConstructId, ...]:
    """Return construct IDs in compiled static-factor order."""
    return tuple(state.id for state in compiled_dynamical_model.static_factors)


def static_factor_names(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[str, ...]:
    """Return display labels in compiled static-factor order."""
    return tuple(state.name for state in compiled_dynamical_model.static_factors)


def n_states(compiled_dynamical_model: CompiledDynamicalModel) -> int:
    """Count compiled state coordinates, including fixed input coordinates."""
    return len(compiled_dynamical_model.states)


def n_observations(compiled_dynamical_model: CompiledDynamicalModel) -> int:
    """Count compiled observation channels rather than measured time points."""
    return len(compiled_dynamical_model.observations)


def parameter_blocks(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[
    DiffusionBlockSpec,
    SparseBlockSpec[tuple[int, int]],
    SparseBlockSpec[int],
    ManifestCholBlockSpec,
    SparseBlockSpec[int],
    T0CholBlockSpec,
    SparseBlockSpec[int],
]:
    """Return compiled blocks in diffusion, loading, observation, initial-state, and static order."""
    return (
        compiled_dynamical_model.diffusion_block,
        compiled_dynamical_model.loading_block,
        compiled_dynamical_model.observation_mean_block,
        compiled_dynamical_model.observation_noise_block,
        compiled_dynamical_model.initial_mean_block,
        compiled_dynamical_model.initial_covariance_block,
        compiled_dynamical_model.static_scale_block,
    )
