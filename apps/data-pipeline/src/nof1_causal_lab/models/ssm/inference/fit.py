"""Inference backends for SSM models.

Separates inference from model definition. CompiledDynamicalModel owns the numerical model; this module provides fit() to run inference with the supported backends.

Methods:
- NUTS-DA: joint parameter/trajectory updates with dual averaging and diagonal
  mass adaptation, with independent chains advanced together.
- Marginalized Particle Gibbs: collapsed joint parameter/trajectory updates
  using the shared dSMC smoother and its auxiliary proposals.
- Particle Gibbs: conditional MH or NUTS parameter updates followed by dSMC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.sampling import NutsDASpec, ParticleGibbsSpec
from nof1_causal_lab.models.ssm.inference.reparameterization import AutoReparam
from nof1_causal_lab.models.ssm.preparation.preflight import (
    ObservationPreflightFailure,
    validate_observations_for_fit,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.reparameterization import Strategy
    from nof1_causal_lab.models.ssm.inference.types import (
        MCMCPosterior,
    )
    from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel


def fit(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | Literal["auto"] | None = "auto",
    clock: Callable[[], float],
) -> MCMCPosterior | ObservationPreflightFailure | InitializationFailure:
    """Dispatch exact joint inference through the selected algorithm's owned settings."""
    failure = validate_observations_for_fit(priors, panel)
    if failure is not None:
        return failure
    resolved_reparam = AutoReparam(centered=0.0) if reparam == "auto" else reparam
    if isinstance(sampler.algorithm, ParticleGibbsSpec):
        from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.fit import fit_particle_gibbs

        return fit_particle_gibbs(
            priors,
            panel,
            compute_budget=compute_budget,
            sampler=sampler,
            reparam=resolved_reparam,
            clock=clock,
        )
    if isinstance(sampler.algorithm, NutsDASpec):
        from nof1_causal_lab.models.ssm.inference.samplers.nuts import fit_nuts_da

        return fit_nuts_da(
            priors,
            panel,
            compute_budget=compute_budget,
            sampler=sampler,
            reparam=resolved_reparam,
            clock=clock,
        )
    from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.fit import (
        fit_marginal_particle_gibbs,
    )

    return fit_marginal_particle_gibbs(
        priors,
        panel,
        compute_budget=compute_budget,
        sampler=sampler,
        reparam=resolved_reparam,
        clock=clock,
    )
