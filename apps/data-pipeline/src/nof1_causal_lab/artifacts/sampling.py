"""Owned sampler choices and explicit numerical initialization inputs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, field_serializer, field_validator

from nof1_causal_lab.artifacts.base import Value

type DSMCLeafProposal = Literal["amala_exact", "paid_mix"]


class InferenceHardware(StrEnum):
    """Run here, or request one of Modal's named GPU types."""

    LOCAL = "local"
    T4 = "T4"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    L4 = "L4"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    A10 = "A10"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    L40S = "L40S"
    A100 = "A100"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    A100_40GB = "A100-40GB"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    A100_80GB = "A100-80GB"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    RTX_PRO_6000 = "RTX-PRO-6000"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    H100 = "H100"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    H100_ONLY = "H100!"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    H200 = "H200"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.
    B200 = "B200"  # noqa: V107 -- Pydantic constructs this GPU choice from configuration.


class NutsDASpec(Value):
    """Joint state/parameter NUTS with dual averaging and diagonal mass adaptation."""

    method: Literal["nuts_da"] = "nuts_da"
    target_accept_prob: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    max_tree_depth: int = Field(default=10, ge=1)
    initial_step_size: float = Field(default=1.0, gt=0, allow_inf_nan=False)


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
    """Custom inference seam, currently backed by LAPS and marginalized Particle Gibbs."""

    method: Literal["custom"] = "custom"
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
    amala_grad_clip: float = Field(
        default=math.inf, gt=0, json_schema_extra={"default": "infinity"}
    )
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

    @field_validator(
        "amala_grad_clip", mode="before", json_schema_input_type=FiniteFloat | Literal["infinity"]
    )
    @classmethod
    def parse_gradient_clip(cls, value: float | Literal["infinity"]) -> float:
        """Decode the JSON string ``infinity`` as an unbounded gradient-clipping threshold."""
        return math.inf if value == "infinity" else value

    @field_serializer("amala_grad_clip")
    def serialize_gradient_clip(self, value: float) -> FiniteFloat | Literal["infinity"]:
        """Encode an infinite gradient-clipping threshold as the JSON-safe string ``infinity``."""
        return "infinity" if math.isinf(value) else value


type InferenceAlgorithmSpec = Annotated[
    NutsDASpec | MarginalParticleGibbsSpec, Field(discriminator="method")
]


class SamplerSpec(Value):
    """Fully resolved controls for exact joint state and parameter inference."""

    num_warmup: int = Field(default=4000, ge=0)
    num_samples_per_chain: int = Field(default=1000, ge=1)
    num_chains: int = Field(default=4, ge=1)
    seed: int = Field(default=0, ge=0)
    num_particles: int = Field(default=64, ge=2)
    retain_latent_paths: bool = True
    algorithm: InferenceAlgorithmSpec = Field(default_factory=MarginalParticleGibbsSpec)


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured joint sampler."""

    algorithm: InferenceAlgorithmSpec | None = None
    num_samples_per_chain: int | None = Field(default=None, ge=1)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    num_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)
