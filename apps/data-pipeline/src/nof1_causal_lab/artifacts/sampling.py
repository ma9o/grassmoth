"""Owned sampler choices and explicit numerical initialization inputs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, field_serializer, field_validator, model_validator

from nof1_causal_lab.artifacts.base import Value

type DSMCLeafProposal = Literal["amala", "amala_plus"]


class PaidMixSpec(Value):
    """Optional fixed pilot and wide proposals mixed with an auxiliary particle proposal."""

    local_weight: float = Field(default=0.85, gt=0, lt=1, allow_inf_nan=False)
    pilot_weight: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)
    pilot_var_scale: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    wide_mult: float = Field(default=4.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def positive_wide_weight(self) -> PaidMixSpec:
        """Own the remaining, strictly positive wide-component probability."""
        if self.local_weight + self.pilot_weight >= 1:
            raise ValueError("paid_mix local_weight + pilot_weight must be less than one")
        return self


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
    """NUTS with dual averaging and diagonal mass adaptation, joint or conditional."""

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


class DsmcSpec(Value):
    """Conditional dSMC auxiliary law, particle proposal, and warmup adaptation."""

    amala_delta_init: float = Field(default=1e-2, gt=0, allow_inf_nan=False)
    amala_delta_min: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    amala_delta_max: float = Field(default=1e1, gt=0, allow_inf_nan=False)
    amala_target_accept: float = Field(default=0.75, gt=0, lt=1, allow_inf_nan=False)
    amala_adaptation_window: int = Field(default=100, ge=1)
    amala_adaptation_tolerance: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    amala_adaptation_rho: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    amala_adaptation_rho_min: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    amala_adaptation_gamma: float = Field(default=-0.5, le=0, allow_inf_nan=False)
    amala_kappa: float = Field(default=0.75, ge=0, allow_inf_nan=False)
    amala_grad_clip: float = Field(
        default=math.inf, gt=0, json_schema_extra={"default": "infinity"}
    )
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_plus"
    paid_mix: PaidMixSpec | None = Field(default_factory=PaidMixSpec)
    # Coordinate-block proposals: number of latent coordinates proposed per sweep
    # (None = all). Blocks of 2-4 sidestep the joint-coherence weight degeneracy of
    # full-state proposals at higher latent dimension.
    latent_block_coords: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def ordered_delta_bounds(self) -> DsmcSpec:
        """Own the positive, ordered adaptation interval and its initial value."""
        if not self.amala_delta_min <= self.amala_delta_init <= self.amala_delta_max:
            raise ValueError("amala_delta_init must be inside [amala_delta_min, amala_delta_max]")
        return self

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


class MetropolisHastingsSpec(Value):
    """Conditional Gaussian random walk with LAPS preconditioning and warmup dual averaging."""

    method: Literal["mh"] = "mh"
    initial_step_size: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    target_accept_prob: float = Field(default=0.3, gt=0, lt=1, allow_inf_nan=False)


type ConditionalParameterSpec = Annotated[
    MetropolisHastingsSpec | NutsDASpec, Field(discriminator="method")
]


class ParticleGibbsSpec(Value):
    """Conditional MH or NUTS parameters and dSMC trajectories on the exact joint posterior."""

    method: Literal["pgibbs"] = "pgibbs"
    parameter_kernel: ConditionalParameterSpec = Field(default_factory=MetropolisHastingsSpec)
    dsmc: DsmcSpec = Field(default_factory=DsmcSpec)
    laps: LapsInitializationSpec = Field(default_factory=LapsInitializationSpec)
    diagnostic_metrics_all: bool = False


class MarginalParticleGibbsSpec(Value):
    """Marginalized parameter ensemble and conditional dSMC trajectory updates."""

    method: Literal["mpgibbs"] = "mpgibbs"
    n_parameter_particles: int = 2
    tempering: GaussianReferenceTemperingSpec | None = None
    dsmc: DsmcSpec = Field(default_factory=DsmcSpec)
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "pseudo_langevin"
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


type InferenceAlgorithmSpec = Annotated[
    NutsDASpec | MarginalParticleGibbsSpec | ParticleGibbsSpec, Field(discriminator="method")
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
