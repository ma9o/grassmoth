"""Inference backends for SSM models.

Separates inference from model definition. CompiledDynamicalModel owns the numerical model; this module provides fit() to run inference with the supported backends.

Method:
- Marginalized Particle Gibbs: collapsed joint parameter/trajectory updates
  using the dSMC smoother with exactly corrected aMALA or PAID-mixture leaf
  proposals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.models.ssm.autoreparam import AutoReparam
from nof1_causal_lab.models.ssm.inference.types import (
    ParticleMCMCPosterior as ParticleMCMCPosterior,
)
from nof1_causal_lab.models.ssm.inference.types import (
    WarmupProposal as WarmupProposal,
)
from nof1_causal_lab.models.ssm.preflight import (
    ObservationPreflightFailure,
)
from nof1_causal_lab.models.ssm.preflight import (
    validate_observations_for_fit as validate_observations_for_fit,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws
    from nof1_causal_lab.sampler_config import SamplerSpec

__all__ = [
    "ParticleMCMCPosterior",
    "WarmupProposal",
    "fit",
    "prior_predictive",
    "validate_observations_for_fit",
]


def fit(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | Literal["auto"] | None = "auto",
    clock: Callable[[], float],
) -> ParticleMCMCPosterior | ObservationPreflightFailure:
    """Run the production particle sampler on resolved numerical controls."""
    failure = validate_observations_for_fit(priors, panel)
    if failure is not None:
        return failure
    resolved_reparam = AutoReparam(centered=0.0) if reparam == "auto" else reparam
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs import (
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


def prior_predictive(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    num_samples: int = 100,
    seed: int = 0,
    *,
    compute_budget: ComputeBudget,
) -> PredictiveDraws:
    """Sample the prior predictive distribution on the bound model's time grid."""
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        sample_prior_predictive_from_runtime,
    )

    return sample_prior_predictive_from_runtime(
        panel.compiled_dynamical_model,
        priors,
        panel.times,
        compute_budget=compute_budget,
        num_samples=num_samples,
        seed=seed,
        input_events=panel.input_events,
    )
