"""Check observation semantics and sampler support before numerical fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    JointSmootherSpec,
    NutsDASpec,
    ParameterMixtureProposalSpec,
    SamplerSpec,
)
from grassmoth.models.ssm import compiled_access as numeric

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.preparation.binding import BoundPanel


@dataclass(frozen=True)
class ObservationPreflightFailure:
    """Observed data is inconsistent with the spec/prior configuration."""

    message: str

    @classmethod
    def rejected(cls, message: str) -> Self:
        """Own the rejection value shared by observation preparation and fitting."""
        return cls(message)


def validate_sampler_support(
    compiled: CompiledDynamicalModel,
    sampler: SamplerSpec,
) -> ObservationPreflightFailure | None:
    """Own sampler-specific restrictions, including checks before remote allocation."""
    if (failure := sampler.support_failure) is not None:
        return ObservationPreflightFailure.rejected(failure.message)
    if DistributionFamily.STUDENT_T in numeric.diffusion_families(compiled):
        return ObservationPreflightFailure.rejected(
            "Exact inference currently requires Gaussian latent diffusion for every state."
        )
    options = sampler.algorithm
    path_update = options.smoother if isinstance(options, NutsDASpec) else options.path_update
    if isinstance(path_update, JointSmootherSpec):
        return None
    intervals = any(item.support.support_kind == "interval" for item in compiled.observations)
    if isinstance(path_update.method, AuxiliaryKalmanMethodSpec):
        if not isinstance(options, NutsDASpec) and {"particle_identity", "particle_weights"} & set(
            options.diagnostic_metrics
        ):
            return ObservationPreflightFailure.rejected(
                "Auxiliary Kalman has no particle bank; particle identity and weight diagnostics "
                "require a particle path update."
            )
        return None
    if isinstance(path_update.proposal, ParameterMixtureProposalSpec) and (
        compiled.state_space.requires_replay or intervals
    ):
        return ObservationPreflightFailure.rejected(
            "Running-prefix mixture proposals support point readings without replay."
        )
    return None


def validate_observation_support_for_fit(panel: BoundPanel) -> ObservationPreflightFailure | None:
    """Reject observation semantics the particle target cannot represent."""
    intervals = [
        indicator.name
        for indicator in panel.compiled_dynamical_model.observations
        if indicator.support.support_kind == "interval"
        and not panel.compiled_dynamical_model.states[indicator.state_index].is_input
        and isinstance(indicator.law, DeltaLawSpec)
    ]
    if intervals:
        names = ", ".join(intervals)
        return ObservationPreflightFailure.rejected(
            "Exact endogenous interval constraints are not supported by particle inference; "
            f"use a noisy interval observation law for: {names}."
        )
    compiled = panel.compiled_dynamical_model
    observed = np.asarray(panel.observations)
    for index, state in enumerate(compiled.states):
        columns = tuple(
            column
            for column, indicator in enumerate(compiled.observations)
            if indicator.state_index == index and isinstance(indicator.law, DeltaLawSpec)
        )
        if state.is_input or not columns:
            continue
        exact = observed[:, columns]
        readings = exact[np.isfinite(exact)]
        if state.time_invariant and readings.size and np.any(readings != readings[0]):
            return ObservationPreflightFailure.rejected(
                f"Conflicting exact observations for static construct {state.name!r}."
            )
        initial = readings if state.time_invariant else exact[0][np.isfinite(exact[0])]
        if initial.size and index not in compiled.state_space.initial:
            if compiled.initial_mean_block.free_support[index]:
                return ObservationPreflightFailure.rejected(
                    f"Exact readings of {state.name!r} constrain its uncertain initial mean; "
                    "inference on parameter equality constraints is not supported."
                )
            if np.any(initial != np.asarray(compiled.initial_mean_block.template)[index]):
                return ObservationPreflightFailure.rejected(
                    f"Exact readings of {state.name!r} disagree with its fixed initial value."
                )
    for column, indicator in enumerate(compiled.observations):
        state = compiled.states[indicator.state_index]
        if (
            isinstance(indicator.law, DeltaLawSpec)
            and not state.is_input
            and not state.time_invariant
            and indicator.state_index not in compiled.state_space.transition
            and np.any(np.isfinite(observed[1:, column]))
        ):
            return ObservationPreflightFailure.rejected(
                f"Exact readings of deterministic evolving construct {state.name!r} after the "
                "initial time require constrained inference, which is not supported. "
                "Its declared deterministic dynamics have been preserved."
            )
    return None
