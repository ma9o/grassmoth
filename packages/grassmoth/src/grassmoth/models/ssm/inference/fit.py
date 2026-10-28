"""Inference backends for SSM models.

Separates inference from model definition. CompiledDynamicalModel owns the numerical model; this module provides fit() to run inference with the supported backends.

Methods:
- NUTS-DA: joint parameter/trajectory updates with dual averaging and diagonal
  mass adaptation, with independent chains advanced together.
- Marginalized Particle Gibbs: collapsed joint parameter/trajectory updates
  using the dSMC smoother and its auxiliary proposals, optionally followed by
  conditional MH or NUTS parameter moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.sampling import NutsDASpec
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
    """Dispatch exact joint inference through the selected algorithm's owned settings."""
    failure = validate_observations_for_fit(priors, panel)
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
