"""Derive execution axis identities, labels, and family metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.compilation_errors import IncompleteModelError
from grassmoth.models.structure import (
    get_model_clock,
    selected_indicators,
    selected_state_ids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from grassmoth.artifacts.identity import ConstructId, IndicatorId
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.model.indicator import IndicatorSpec
    from grassmoth.artifacts.model.likelihood import LikelihoodSpec
    from grassmoth.models.structure import StructuralSelection


def get_construct_dt_days(
    dynamical_model_spec: DynamicalModelSpec,
) -> float:
    """Get the model clock interval in fractional days."""
    return get_model_clock(dynamical_model_spec).days


def state_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """Return retained construct IDs in the selection's execution-state order."""
    return selected_state_ids(selection)


def state_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Return state display labels in the same order as the compiled state axis."""
    dynamical_model_spec = selection.dynamical_model_spec
    return tuple(
        dynamical_model_spec.get_construct(identity).name for identity in state_ids(selection)
    )


def n_states(selection: StructuralSelection) -> int:
    """Count the construct coordinates retained on the compiled state axis."""
    return len(state_ids(selection))


def time_invariant_mask(selection: StructuralSelection) -> np.ndarray:
    """Mark compiled state coordinates whose constructs are time-invariant."""
    dynamical_model_spec = selection.dynamical_model_spec
    return np.asarray(
        [
            dynamical_model_spec.get_construct(identity).temporal_status == "time_invariant"
            for identity in state_ids(selection)
        ],
        dtype=bool,
    )


def input_mask(selection: StructuralSelection) -> np.ndarray:
    """Coordinates read from the panel instead of generated under a state law."""
    dynamical_model_spec = selection.dynamical_model_spec
    return np.asarray(
        [
            dynamical_model_spec.get_construct(identity).role == "exogenous"
            for identity in state_ids(selection)
        ],
        dtype=bool,
    )


def diffusion_families(selection: StructuralSelection) -> tuple[DistributionFamily, ...]:
    """Resolve innovation families, requiring diffusion scales for dynamic endogenous states."""
    from grassmoth.artifacts.probability.families import DistributionFamily

    dynamical_model_spec = selection.dynamical_model_spec
    result = []
    for identity in state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        if construct.role == "exogenous" or construct.temporal_status == "time_invariant":
            result.append(DistributionFamily.GAUSSIAN)
        elif construct.coefficient("diffusion_scale") is None:
            raise IncompleteModelError(f"Construct {construct.name!r} requires a diffusion scale")
        else:
            result.append(construct.innovation_family)
    return tuple(result)


def observed_indicators(selection: StructuralSelection) -> tuple[IndicatorSpec, ...]:
    """Return model-owned indicators in the selection's compiled observation order."""
    dynamical_model_spec = selection.dynamical_model_spec
    return tuple(
        dynamical_model_spec.indicator(identity)
        for identity in tuple(
            indicator.observation.id for indicator in selected_indicators(selection)
        )
    )


def observation_ids(selection: StructuralSelection) -> tuple[IndicatorId, ...]:
    """Return stable observation IDs in compiled indicator order."""
    return tuple(indicator.observation.id for indicator in observed_indicators(selection))


def observation_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Return observation display labels in compiled indicator order."""
    return tuple(indicator.observation.name for indicator in observed_indicators(selection))


def n_observations(selection: StructuralSelection) -> int:
    """Count observation channels in the compiled selection, not recorded rows."""
    return len(observed_indicators(selection))


def observation_families(selection: StructuralSelection) -> tuple[DistributionFamily, ...]:
    """Return each selected observation's likelihood family in compiled indicator order."""
    return tuple(likelihood.law.family for likelihood in _likelihoods(selection))


def observation_level_counts(selection: StructuralSelection) -> tuple[int, ...]:
    """Count declared levels for ordinal and categorical observations, using zero otherwise."""
    return tuple(
        len(indicator.observation.levels)
        if indicator.likelihood is not None
        and indicator.likelihood.law.family
        in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        else 0
        for indicator in observed_indicators(selection)
    )


def _likelihoods(selection: StructuralSelection) -> Iterator[LikelihoodSpec]:
    for indicator in observed_indicators(selection):
        if indicator.likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.observation.name!r} requires a likelihood"
            )
        yield indicator.likelihood
