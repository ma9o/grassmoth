"""Owned sampler choices and explicit numerical initialization inputs."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, FiniteFloat, field_serializer, field_validator

from nof1_causal_lab.artifacts.base import Value

type DSMCLeafProposal = Literal["amala_exact", "paid_mix"]


class GaussianReferenceTemperingSpec(Value):
    """Optional replica exchange; each ladder retains only its exact beta-one chain.

    The ladder spans beta one to zero. The reference is fixed at initialization,
    its endpoint is independently refreshed every sweep, and even/odd neighboring
    swaps alternate. Rejection-based ladder adaptation runs during the first half
    of warmup, leaving the rest to settle before retaining posterior draws.
    """

    num_temperatures: int = Field(default=48, ge=2)
    reference_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    adaptation_interval: int = Field(default=500, ge=2)


class LapsInitializationSpec(Value):
    """Joint ensemble initialization; its draws are never reported as posterior samples."""

    ensemble_size: int = Field(default=1024, ge=2)
    num_unadjusted_steps: int = Field(default=10000, ge=10)
    num_adjusted_steps: int = Field(default=100, ge=1)
    steps_per_sample: int = Field(default=15, ge=1)
    initial_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    covariance_regularization: float = Field(default=1e-4, gt=0, allow_inf_nan=False)


class MarginalParticleGibbsSpec(Value):
    """Marginalized Particle Gibbs inference settings."""

    n_parameter_particles: int = 2
    tempering: GaussianReferenceTemperingSpec | None = None
    latent_delta: float = 0.2
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "pseudo_langevin"
    amala_delta_init: float = 1e-2
    amala_delta_min: float = 1e-5
    amala_delta_max: float = 1e1
    amala_target_accept: float = 0.75
    amala_adaptation_window: int = 100
    amala_adaptation_tolerance: float = 0.05
    amala_adaptation_rho: float = 0.5
    amala_adaptation_rho_min: float = 1e-3
    amala_adaptation_gamma: float = -0.5
    amala_kappa: float = 0.75
    amala_grad_clip: float = Field(default=math.inf, gt=0)
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_exact"
    # Coordinate-block proposals: number of latent coordinates proposed per sweep
    # (None = all). Blocks of 2-4 sidestep the joint-coherence weight degeneracy of
    # full-state proposals at higher latent dimension.
    latent_block_coords: int | None = None
    # paid_mix leaf mixture: z-anchored (amala_exact core) + fixed LAPS-pilot
    # component + wide tail (weight = 1 - z - pilot). Pilot variances are
    # pilot_var_scale x the LAPS paths' per-coordinate spread; the wide tail is
    # wide_mult x the same spread.
    paid_mix_z_weight: float = 0.85
    paid_mix_pilot_weight: float = 0.10
    paid_mix_pilot_var_scale: float = 0.25
    paid_mix_wide_mult: float = 4.0
    diagnostic_metrics_all: bool = False
    diagnostic_metrics: tuple[str, ...] = ()
    param_step_size: float = 0.02
    param_step_size_min: float = 1e-6
    param_step_size_max: float = 1e3
    param_target_accept: float = 0.35
    adaptation_rate: float = 0.05
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple"
    laps: LapsInitializationSpec = Field(default_factory=LapsInitializationSpec)
    compute_latent_posterior_summary: bool = True

    @field_validator("amala_grad_clip", mode="before")
    @classmethod
    def parse_gradient_clip(cls, value: float | Literal["infinity"]) -> float:
        """Decode the JSON string ``infinity`` as an unbounded gradient-clipping threshold."""
        return math.inf if value == "infinity" else value

    @field_serializer("amala_grad_clip")
    def serialize_gradient_clip(self, value: float) -> FiniteFloat | Literal["infinity"]:
        """Encode an infinite gradient-clipping threshold as the JSON-safe string ``infinity``."""
        return "infinity" if math.isinf(value) else value


class SamplerSpec(Value):
    """Fully resolved controls for the exact particle sampler."""

    num_warmup: int = Field(default=4000, ge=0)
    num_samples_per_chain: int = Field(default=1000, ge=1)
    num_chains: int = Field(default=4, ge=1)
    seed: int = Field(default=0, ge=0)
    num_particles: int = Field(default=64, ge=2)
    retain_latent_paths: bool = True
    marginal_particle_gibbs: MarginalParticleGibbsSpec = Field(
        default_factory=MarginalParticleGibbsSpec
    )
