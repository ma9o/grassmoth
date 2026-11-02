"""Inference for SSM models.

Separates inference from model definition. CompiledDynamicalModel owns the numerical model;
this module provides fit() to run marginalized Particle Gibbs: collapsed joint
parameter/trajectory updates whose path update is Particle-mGRAD, twisted Particle-aGRAD or
auxiliary Kalman.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.models.ssm.preparation.preflight import (
    ObservationPreflightFailure,
    validate_observations_for_fit,
    validate_sampler_support,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.models.ssm.inference.types import (
        MCMCPosterior,
        UnretainedRun,
    )
    from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel


def fit(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    sampler: SamplerSpec,
    execution_context: ModelExecutionContext,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | ObservationPreflightFailure | InitializationFailure:
    """Run exact joint inference once the observations and sampler pass preflight."""
    failure = validate_observations_for_fit(priors, panel)
    if failure is not None:
        return failure
    failure = validate_sampler_support(panel.compiled_dynamical_model, sampler)
    if failure is not None:
        return failure
    from grassmoth.models.ssm.inference.samplers.mpgibbs.fit import (
        fit_marginal_particle_gibbs,
    )

    return fit_marginal_particle_gibbs(
        priors,
        panel,
        sampler=sampler,
        execution_context=execution_context,
        continuation=continuation,
    )
