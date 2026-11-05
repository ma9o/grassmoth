"""Inference for SSM models.

Separates inference from model definition. CompiledDynamicalModel owns the numerical model;
this module dispatches exact joint NUTS or marginalized Particle Gibbs with
Particle-mGRAD, twisted Particle-aGRAD or auxiliary Kalman path updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.sampling import NutsDASpec
from grassmoth.models.ssm.preparation.preflight import (
    ObservationPreflightFailure,
    validate_observation_support_for_fit,
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
    failure = validate_observation_support_for_fit(panel)
    if failure is not None:
        return failure
    failure = validate_sampler_support(panel.compiled_dynamical_model, sampler)
    if failure is not None:
        return failure
    if isinstance(sampler.algorithm, NutsDASpec):
        from grassmoth.models.ssm.inference.samplers.nuts import fit_nuts_da

        return fit_nuts_da(
            priors,
            panel,
            sampler=sampler,
            execution_context=execution_context,
            continuation=continuation,
        )
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
