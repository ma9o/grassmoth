"""Owned sampler choices and explicit numerical initialization inputs."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

type AuxiliaryDrift = Literal["none", "amala", "amala_plus"]
type LatentBlockSize = Literal[1, 2, "all"]
type JointParameterization = Literal["centred", "non_centred", "mixed"]


class AdaptedLatentStepSpec(Value):
    """Warmup adapts the per-time scale towards the assigned movement target."""

    kind: Literal["adapted"] = "adapted"
    target: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)


class FixedLatentStepSpec(Value):
    """An assigned scale which remains fixed during warmup and sampling."""

    kind: Literal["fixed"] = "fixed"
    delta: float = Field(default=0.01, gt=0, allow_inf_nan=False)


type LatentStepSpec = Annotated[
    AdaptedLatentStepSpec | FixedLatentStepSpec, Field(discriminator="kind")
]


class NutsTrajectorySpec(Value):
    """A dynamically terminated trajectory, bounded by tree depth."""

    kind: Literal["nuts"] = "nuts"
    max_tree_depth: int = Field(default=10, ge=1)


class FixedTrajectorySpec(Value):
    """A fixed number of leapfrog steps with the production jitter policy."""

    kind: Literal["fixed"] = "fixed"
    steps: int = Field(default=8, ge=1)


type JointTrajectorySpec = Annotated[
    NutsTrajectorySpec | FixedTrajectorySpec, Field(discriminator="kind")
]


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
    """Joint HMC with dual averaging and windowed diagonal mass adaptation."""

    method: Literal["nuts_da"] = "nuts_da"
    parameterization: JointParameterization = Field(
        default="centred",
        description="Experimental coordinates for the unchanged joint posterior; frozen per fit.",
    )
    target_accept_prob: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    trajectory: JointTrajectorySpec = Field(default_factory=NutsTrajectorySpec)
    initial_step_size: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class ConditionalNutsSpec(Value):
    """Conditional NUTS parameter updates within Particle Gibbs."""

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
    """Conditional dSMC auxiliary law, particle proposal, and assigned scale policy."""

    drift: AuxiliaryDrift = "none"
    latent_step: LatentStepSpec = Field(default_factory=AdaptedLatentStepSpec)
    latent_block_coords: LatentBlockSize = 1
    pilot_mixture: bool = True


class MetropolisHastingsSpec(Value):
    """Conditional Gaussian random walk with LAPS preconditioning and warmup dual averaging."""

    method: Literal["mh"] = "mh"
    initial_step_size: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    target_accept_prob: float = Field(default=0.3, gt=0, lt=1, allow_inf_nan=False)


type ConditionalParameterSpec = Annotated[
    MetropolisHastingsSpec | ConditionalNutsSpec, Field(discriminator="method")
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
    n_parameter_particles: int = Field(default=2, ge=2)
    tempering: GaussianReferenceTemperingSpec | None = None
    dsmc: DsmcSpec = Field(default_factory=DsmcSpec)
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "random_walk"
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
    num_samples_per_chain: int = Field(default=1000, ge=0)
    num_chains: int = Field(default=4, ge=1)
    seed: int = Field(default=0, ge=0)
    num_particles: int = Field(default=64, ge=2)
    hardware: InferenceHardware = InferenceHardware.L40S
    time_limit_seconds: int = Field(
        default=14400,
        ge=1,
        description="Modal startup and execution timeout per phase. Local fits have no enforced timeout.",
    )
    retain_latent_paths: bool = True
    algorithm: InferenceAlgorithmSpec = Field(default_factory=MarginalParticleGibbsSpec)


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured joint sampler."""

    hardware: InferenceHardware | None = None
    time_limit_seconds: int | None = Field(default=None, ge=1)
    algorithm: InferenceAlgorithmSpec | None = None
    num_samples_per_chain: int | None = Field(default=None, ge=0)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    num_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)
