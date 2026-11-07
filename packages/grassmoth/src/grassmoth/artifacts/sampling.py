"""Owned sampler choices and explicit numerical initialization inputs.

Each field description says what the choice does and why it is offered. Defaults
are starting values; adaptation constants not listed here are fixed.

References:
  https://doi.org/10.1111/j.1467-9868.2009.00736.x — Andrieu, Doucet & Holenstein (2010),
    "Particle Markov chain Monte Carlo methods": Particle Gibbs (section 2.4.3).
  https://arxiv.org/abs/2303.00301 — Corenflos & Särkkä (2025), "Auxiliary MCMC samplers for
    parallelisable inference in high-dimensional latent dynamical systems": Gaussian path
    proposals (2.2-2.4) and per-time-step targets (4).
  https://arxiv.org/abs/2401.14868 — Corenflos & Finke (2024), "Particle-MALA and
    Particle-mGRAD": Particle-aGRAD (Algorithm 6, twisted in 4.4) and Particle-mGRAD
    (Algorithm 7).
  https://arxiv.org/abs/2505.04611 — Corenflos (2025), "Particle Gibbs without the Gibbs bit":
    the auxiliary proposal (3.1), Barker selection (3.3), adaptation (3.4) and M = 2 (5).
  https://arxiv.org/abs/1111.4246 — Hoffman & Gelman (2014), "The No-U-Turn Sampler": dual
    averaging (3.2).
  https://doi.org/10.1214/aoap/1034625254 — Roberts, Gelman & Gilks (1997): optimal scaling.
  https://arxiv.org/abs/2104.02020 — Agrawal, Vats, Łatuszyński & Roberts (2023): optimal
    scaling of Barker's acceptance.
  https://doi.org/10.1214/20-BA1221 — Vehtari et al. (2021): rank-normalized R-hat.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import AfterValidator, Field

from grassmoth.artifacts.base import Value

type ParameterProposal = Literal["random_walk", "pseudo_langevin"]
type AdaptationScheme = Literal["simple", "dual_averaging"]
type ParticleDiagnosticMetric = Literal[
    "particle_identity",
    "parameter_movement",
    "particle_weights",
    "frequency_refresh",
    "regional_movement",
]


def _metric_set(
    metrics: tuple[ParticleDiagnosticMetric, ...],
) -> tuple[ParticleDiagnosticMetric, ...]:
    """Reject repeated metrics and order the rest, so equal selections share one call identity."""
    if len(set(metrics)) != len(metrics):
        raise ValueError("Diagnostic metrics must be unique")
    return tuple(sorted(metrics))


type ParticleDiagnosticMetrics = Annotated[
    tuple[ParticleDiagnosticMetric, ...], AfterValidator(_metric_set)
]


def _power_of_two(value: int) -> int:
    """Require sizes that fill the GPU kernels' power-of-two tiles exactly.

    Other sizes would pad those tiles, and padded tiles cost more than full ones:
    L40S SV C10/P128/u8, separate sessions, M17 takes 63.80 ms per sweep against 43.85 ms
    at M32.
    """
    if value.bit_count() != 1:
        raise ValueError(f"Must be a power of two (2, 4, 8, ...), got {value}")
    return value


type ParticleCount = Annotated[int, Field(ge=2), AfterValidator(_power_of_two)]

MGRAD_STATE_LIMIT: Annotated[
    int,
    Field(
        description=(
            "Particle-mGRAD rejects larger models, the end of its fused GPU tail's measured range. "
            "Paired L40S L96 D40, C10/P128/K8/u8: extending the four-coordinate tail and finisher "
            "beyond D16 increased ordinary sweep time 7.35 to 8.10 ms."
        )
    ),
] = 16

MGRAD_PARAMETER_PARTICLE_LIMIT: Annotated[
    int,
    Field(
        description=(
            "Largest Particle-mGRAD parameter-particle count that has been gated and timed. "
            "L40S: D1/3/9 P128/K64 pass isolated no-spill gates. Paired C10/P128/u8: the fused "
            "tail at K64 reduced SV ordinary sweeps 91.93 to 64.52 ms and PARAM D3 88.01 to "
            "83.87 ms. Nothing larger has been gated or timed."
        )
    ),
] = 64

MGRAD_BANK_LIMIT: Annotated[
    int,
    Field(
        description=(
            "Largest particles-by-parameter-particles product in Particle-mGRAD's fused GPU tail. "
            "Paired L40S C10/P128/u8: the tail reduced ordinary sweeps at 8192 entries (K64) as "
            "stated for the parameter-particle limit, and at 4096 entries (K32) 113.24 to 43.85 ms "
            "on SV and 75.64 to 57.39 ms on PARAM. Many shapes within it haven't been timed."
        )
    ),
] = 8192

MGRAD_LARGE_PARTICLE_COUNT: Annotated[
    int,
    Field(
        description=(
            "L40S isolated forward gates at D9: P512/K16 passes; P1024/K8 spills, with the "
            "responsible kernel not isolated. Paired SV C10/u8 P512/K8 ordinary sweeps took "
            "49.96 ms on the tail against 68.73 ms off it."
        )
    ),
] = 512

MGRAD_LARGE_PARTICLE_BANK_LIMIT: Annotated[
    int,
    Field(
        description=(
            "Heuristic: above the large-particle count, retain the tail's earlier 4096-entry "
            "limit. No shape above 512 particles has been timed."
        )
    ),
] = 4096


class AdaptedLatentStepSpec(Value):
    """Warmup adapts the per-time scale towards the assigned movement target."""

    kind: Literal["adapted"] = "adapted"
    target: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description=(
            "Fraction of proposals at each time step that should change its state. In a new "
            "fit every time step's scale starts at 0.01, adapts towards this target during "
            "warmup and is then frozen. Corenflos & Särkkä (2025, section 4) target 0.5 per "
            "time step, below Finke & Thiery's 1 - (N + 1)^(-1/3), because a higher target can "
            "be unreachable at some time steps and leaves the sampler stuck there. Corenflos & "
            "Finke (2024, section 5.2) use 0.75 and note that the optimum depends on N and on "
            "the state dimension."
        ),
    )


class FixedLatentStepSpec(Value):
    """An assigned scale which remains fixed during warmup and sampling."""

    kind: Literal["fixed"] = "fixed"
    delta: float = Field(
        default=0.01,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Twice the auxiliary variance at every time step: each sweep refreshes the auxiliary "
            "point around the current state with variance delta/2, and the same variance ties "
            "the candidates to it (Corenflos & Finke 2024, Algorithms 6 and 7). It is in the "
            "state's own units and shared by all its coordinates. Fix it to compare proposals "
            "at a matched scale, or when adaptation settles on a scale at which the path stops "
            "moving. The default is the starting step of their Appendix F, Algorithm 16."
        ),
    )


type LatentStepSpec = Annotated[
    AdaptedLatentStepSpec | FixedLatentStepSpec, Field(discriminator="kind")
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


class GaussianInitializationSpec(Value):
    """Gaussian starts around a data-conditioned MPGibbs path center.

    A damped iterated extended Kalman smoother conditions the path center on
    point readings at the initial parameter center, preserving the native
    Gaussian jitter and parameter starts. Interval readings are omitted from
    this seed. Exact replay restores recurrences, static values and constraints;
    subsequent particle transitions and diagnostics use the exact nonlinear target.
    """

    scale: float = Field(
        default=0.1,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Standard deviation of the starts around the native starting values, in the "
            "centred chart. The initial parameter preconditioner is scale times the identity, "
            "and MPGibbs re-estimates it in warmup windows. The default is a heuristic; no paper "
            "or recorded benchmark sets it."
        ),
    )


_GUIDED_PARTICLES = (
    "Particles per sweep, including the current path. Each extends a parent chosen at the "
    "previous time, so work grows linearly in particles and times, and the filter's depth "
    "grows linearly with the number of times (Corenflos & Finke 2024, section 4.7). The default "
    "is a heuristic; no paper or recorded benchmark sets it. Corenflos & Finke (2024, section "
    "5.2) use 32 particles; Corenflos & Särkkä (2025, section 4) use 25 and call that somewhat "
    "arbitrary. Must be a power of two; Particle-mGRAD also limits it jointly with the parameter "
    "particles."
)
_PATH_STEP = (
    "Per-time-step scale: adapted towards a movement target, or fixed. Small scales keep "
    "proposals near the current path. For particle updates, large scales approach conditional "
    "SMC with the transition as proposal (Corenflos & Finke 2024, section 4.6). For Kalman, "
    "delta/2 is the auxiliary variance in the whole-path or block Gaussian proposal "
    "(Corenflos & Särkkä 2025, sections 2.2–2.4)."
)
_GUIDED_KAPPA = (
    "1 shifts the auxiliary point along the gradient of the observation densities, mixed "
    "over parameter labels; 0 omits it (Corenflos & Finke 2024, Algorithm 6). The shift is "
    "a Langevin step on the observation term: for a Gaussian reading it overshoots once the "
    "scale exceeds four times that reading's variance. The default 1 is the gradient-informed "
    "form (section 4.5). Figure 5 of that paper puts it well ahead of 0 for twisted "
    "Particle-aGRAD, but its grayed curves don't distinguish the two for Particle-mGRAD, the "
    "default method."
)
_GUIDED_SCAN_UNROLL = (
    "Time steps per iteration of the forward filter and backward-selection loops. Unrolling "
    "changes how XLA compiles these sequential loops, not the algorithm or its target. Each "
    "step runs small kernels, so on a GPU per-iteration overhead is a large share of a sweep; "
    "unrolling amortizes it at the cost of a larger program and longer compilation. A CPU has "
    "little such overhead to remove. 1 disables unrolling. The default 8 was the fastest of 1, "
    "4 and 8 for twisted Particle-aGRAD (kappa 1) on an A10, with identical transitions. Retune "
    "it per device and model size."
)


class GuidedMethodSpec(Value):
    """Particle-mGRAD or twisted Particle-aGRAD with their particle-bank settings.

    Corenflos & Finke (2024), sections 4.1–4.4. mGRAD integrates out the path
    auxiliary; twisted aGRAD retains it and adds Gaussian look-ahead. These
    methods use the same exact parameter-mixture target, with different weights.
    """

    kind: Literal["particle_mgrad", "twisted_particle_agrad"] = Field(
        default="particle_mgrad",
        description=(
            "particle_mgrad integrates the path auxiliary out of the particle weights with the "
            "untwisted proposal (section 4.2); it has no look-ahead, so separated path "
            "components and informative future readings can slow its mixing. It requires a "
            f"parameter_mixture proposal, at most {MGRAD_STATE_LIMIT} state dimensions and a "
            "limited particle bank, the shapes its fused GPU tail covers. "
            "twisted_particle_agrad keeps the auxiliary variable "
            "in the weights (section 4.1) and adds a Gaussian look-ahead built at the auxiliary "
            "path: a parallel backward information scan guides the sequential filter, and "
            "proposals combine exact nonlinear transition means with the approximate twist, "
            "whose normalized densities and twist ratios are paid in full (section 4.4); poor "
            "local fits and multiple modes cost efficiency."
        ),
    )
    num_particles: ParticleCount = Field(default=64, description=_GUIDED_PARTICLES)
    kappa: Literal[0, 1] = Field(default=1, description=_GUIDED_KAPPA)
    scan_unroll: int = Field(default=8, ge=1, description=_GUIDED_SCAN_UNROLL)


class AuxiliaryKalmanMethodSpec(Value):
    """Parallel Gaussian proposals corrected to exact nonlinear densities.

    Corenflos & Särkkä (2025), sections 2.2–2.4. Poor Gaussian fits, long memory
    and separated modes can reduce acceptance without changing the target.
    """

    kind: Literal["auxiliary_kalman"] = "auxiliary_kalman"


type PathMethodSpec = Annotated[
    GuidedMethodSpec | AuxiliaryKalmanMethodSpec, Field(discriminator="kind")
]


class WholePathUpdateSpec(Value):
    """Update the complete trajectory; particle methods still build it sequentially."""

    kind: Literal["whole_path"] = "whole_path"


class BlockedPathUpdateSpec(Value):
    """Update contiguous trajectory blocks."""

    kind: Literal["blocked"] = "blocked"
    block_length: int = Field(
        default=300,
        ge=1,
        description="Time points per block. The implemented Kalman update alternates colours and random half-block shifts. Each block pays its exact nonlinear boundary and interior densities. Colours separate blocks by the full window support, with cached factors covering every affected reading. Deterministic recurrences that reach a density or later reading require sequential updates through the whole suffix; output-only recurrences preserve parallel block updates. The default 300 led path movement among 30, 100 and 300 in short probes on 3,000 days of stochastic volatility. At full length it gave 3.6 times the whole path's worst bulk ESS per second; the whole path's single accept/reject left R-hat at 1.16. On Lorenz-96's 300 steps, 25-step blocks gave a quarter of the whole path's, so retune per model.",
    )


type PathUpdateScopeSpec = Annotated[
    WholePathUpdateSpec | BlockedPathUpdateSpec, Field(discriminator="kind")
]


class ParameterMeanProposalSpec(Value):
    """Build proposal dynamics at the mean of supported parameter candidates.

    This repository's shared-guide construction is symmetric in the ensemble.
    Separated parameter components can be poorly represented by their mean.
    Particle-mGRAD doesn't offer it: only its mixture proposal runs the fused GPU tail.
    """

    kind: Literal["parameter_mean"] = "parameter_mean"


class ParameterMixtureProposalSpec(Value):
    """Mix parameter-specific proposals with exact density or auxiliary-label corrections.

    Corenflos (2025), section 3.4 motivates running-prefix transition mixtures.
    Particle-mGRAD retains each proposal component label and integrates out the
    Gaussian path auxiliary, extending Corenflos & Finke (2024), Appendix D.2.
    Twisted Particle-aGRAD pays the full mixture density with a shared auxiliary
    and look-ahead. Both require point observations without deterministic
    recurrences. Prior-weighted and Kalman mixtures are not implemented. Early
    label collapse and poor shared guidance can reduce mixing.

    Design decision (2026-11-06): use candidate-specific proposal coverage to
    address parameter/path mismatch directly, without learned translation or
    diffusion-scale charts. Exact proposal corrections preserve the target;
    they do not guarantee efficient mixing. A scalar low-diffusion experiment
    found that twisted-aGRAD's running-prefix mixture in physical coordinates
    matched the charted mean-guide control in total effective samples per second. The separate
    SV blocked-Kalman run used conditional-parameter proposals, not this mixture,
    and retained parameter R-hat of 1.07--1.11; it does not establish general chart
    replacement. Source runs: scratchpad/chart-replacement-20261106 and
    scratchpad/sv-blocked-chart-off-20261106.
    """

    kind: Literal["parameter_mixture"] = "parameter_mixture"
    weights: Literal["running_prefix", "prior"] = Field(
        default="running_prefix",
        description=(
            "running_prefix uses each particle's exact label probabilities through the "
            "previous time (Corenflos 2025, section 3.4). prior uses parameter priors and "
            "the ensemble-proposal correction, without path evidence; currently unsupported."
        ),
    )


class ConditionalParameterProposalSpec(Value):
    """Build proposals under a label sampled conditional on the current path.

    The blocked Kalman update holds this label fixed during its conditional block
    moves. This is not the ensemble mean and can retain parameter/path dependence.
    """

    kind: Literal["conditional_parameter"] = "conditional_parameter"


type PathProposalSpec = Annotated[
    ParameterMeanProposalSpec | ParameterMixtureProposalSpec | ConditionalParameterProposalSpec,
    Field(discriminator="kind"),
]


def _default_path_proposal(data: dict[str, object]) -> PathProposalSpec:
    """Resolve the authored method's proposal default before constructing its value."""
    method = cast("PathMethodSpec", data["method"])
    return (
        ParameterMixtureProposalSpec()
        if method.kind == "particle_mgrad"
        else ParameterMeanProposalSpec()
    )


class UnsupportedPathUpdate(Value):
    """An authored path configuration for which no execution kernel is implemented."""

    message: str


def _mgrad_bank_failure(particles: int, labels: int) -> UnsupportedPathUpdate | None:
    """Reject a particle bank too large for Particle-mGRAD's fused GPU tail."""
    limit = (
        MGRAD_BANK_LIMIT
        if particles <= MGRAD_LARGE_PARTICLE_COUNT
        else MGRAD_LARGE_PARTICLE_BANK_LIMIT
    )
    if labels <= MGRAD_PARAMETER_PARTICLE_LIMIT and particles * labels <= limit:
        return None
    return UnsupportedPathUpdate(
        message=(
            f"Particle-mGRAD allows at most {MGRAD_PARAMETER_PARTICLE_LIMIT} parameter particles "
            f"and at most {MGRAD_BANK_LIMIT} particles times parameter particles, or "
            f"{MGRAD_LARGE_PARTICLE_BANK_LIMIT} above {MGRAD_LARGE_PARTICLE_COUNT} particles; "
            f"got {particles} particles and {labels} parameter particles."
        )
    )


class PathUpdateSpec(Value):
    """Common declarative path settings, with support resolved before execution.

    The schema describes independent choices; support_failure owns the current
    implementation matrix. Model-dependent restrictions belong to fit preflight.
    """

    method: PathMethodSpec = Field(default_factory=GuidedMethodSpec)
    update: PathUpdateScopeSpec = Field(default_factory=WholePathUpdateSpec)
    proposal: PathProposalSpec = Field(
        default_factory=_default_path_proposal,
        description=(
            "Particle-mGRAD requires a running-prefix parameter mixture, the only proposal "
            "with its fused GPU tail, which also covers individual parameter candidates. "
            "Heuristic: an ESS advantage has not been established. Other methods default to "
            "parameter_mean. Mixtures currently require point observations without "
            "deterministic recurrences; select twisted_particle_agrad with parameter_mean, or "
            "auxiliary_kalman, for intervals or recurrences."
        ),
    )
    latent_step: LatentStepSpec = Field(
        default_factory=AdaptedLatentStepSpec, description=_PATH_STEP
    )

    @property
    def support_failure(self) -> UnsupportedPathUpdate | None:
        """Resolve method, scope and proposal support without silently changing a choice."""
        method, update, proposal = self.method, self.update, self.proposal
        if isinstance(update, BlockedPathUpdateSpec):
            if not isinstance(method, AuxiliaryKalmanMethodSpec):
                return UnsupportedPathUpdate(
                    message="Blocked updates are implemented only for auxiliary Kalman."
                )
            if not isinstance(proposal, ConditionalParameterProposalSpec):
                return UnsupportedPathUpdate(
                    message="Blocked auxiliary Kalman requires a conditional_parameter proposal."
                )
            return None
        if isinstance(proposal, ConditionalParameterProposalSpec):
            return UnsupportedPathUpdate(
                message="Conditional-parameter proposals are implemented only for blocked auxiliary Kalman."
            )
        if isinstance(proposal, ParameterMeanProposalSpec):
            return (
                UnsupportedPathUpdate(
                    message="Particle-mGRAD requires a parameter_mixture proposal; select "
                    "twisted_particle_agrad for parameter_mean."
                )
                if method.kind == "particle_mgrad"
                else None
            )
        if isinstance(method, AuxiliaryKalmanMethodSpec):
            return UnsupportedPathUpdate(
                message="Parameter-mixture proposals are not implemented for auxiliary Kalman."
            )
        if proposal.weights != "running_prefix":
            return UnsupportedPathUpdate(
                message="Particle proposals implement only running_prefix mixture weights."
            )
        if method.kind == "twisted_particle_agrad" and method.kappa != 1:
            return UnsupportedPathUpdate(
                message="Running-prefix mixture proposals require twisted Particle-aGRAD with kappa=1."
            )
        return None

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify the method, scope and proposal law for native continuation."""
        method, update, proposal = self.method, self.update, self.proposal
        return (
            method.kind,
            method.kappa if isinstance(method, GuidedMethodSpec) else 0,
            update.kind,
            update.block_length if isinstance(update, BlockedPathUpdateSpec) else 0,
            proposal.kind,
            proposal.weights if isinstance(proposal, ParameterMixtureProposalSpec) else "",
        )


type JointParameterization = Literal["centred", "non_centred", "mixed"]


class NutsTrajectorySpec(Value):
    """A dynamically terminated trajectory, bounded by tree depth (Hoffman & Gelman 2014)."""

    kind: Literal["nuts"] = "nuts"
    max_tree_depth: int = Field(
        default=10,
        ge=1,
        description=(
            "Maximum number of trajectory doublings per transition, so at most "
            "2^max_tree_depth - 1 leapfrog steps. The report gives the fraction of transitions "
            "that reach it; frequent hits mean the step is small against the posterior's "
            "longest scale, and each extra level can double the cost. 10 is the NumPyro and "
            "Stan default."
        ),
    )


class FixedTrajectorySpec(Value):
    """A fixed number of leapfrog steps, with the step size jittered by up to 20%."""

    kind: Literal["fixed"] = "fixed"
    steps: int = Field(
        default=8,
        ge=1,
        description=(
            "Leapfrog steps per transition. The default 8 and jitter width 20% are heuristics; "
            "no paper or recorded benchmark sets them. Each transition scales the adapted step size by a "
            "uniform factor in [0.8, 1.2], so trajectories cannot stay periodic (Neal 2011, "
            "section 5.3.2). Every transition costs the same number of gradients, so chains "
            "advanced together never wait for the longest NUTS tree, which limits NUTS on "
            "accelerators (Hoffman, Radul & Sountsov 2021)."
        ),
    )


type JointTrajectorySpec = Annotated[
    NutsTrajectorySpec | FixedTrajectorySpec, Field(discriminator="kind")
]


class JointSmootherSpec(Value):
    """Include free latent coordinates in the Hamiltonian update."""

    method: Literal["joint"] = "joint"  # noqa: FIELD002 -- Distinguishes joint smoothing from the object-valued method in PathUpdateSpec.


type NutsSmootherSpec = JointSmootherSpec | PathUpdateSpec


class NutsDASpec(Value):
    """Hamiltonian updates with dual averaging and diagonal mass adaptation.

    The joint smoother moves parameters and free latent coordinates together.
    Other smoothers alternate the same Hamiltonian parameter kernel with an exact
    conditional path update. Both reconstruct deterministic recurrences exactly
    and target the same joint posterior.
    """

    method: Literal["nuts_da"] = "nuts_da"
    smoother: NutsSmootherSpec = Field(
        default_factory=JointSmootherSpec,
        description=(
            "joint moves parameters and free latent coordinates together. Selecting a path "
            "smoother alternates NUTS on parameters conditional on the current path coordinates "
            "with that exact path update at the new parameters. The configured parameterization "
            "also applies to this conditional parameter block; non-centred holds innovations "
            "fixed, centred holds states fixed. Both preserve the same joint posterior."
        ),
    )
    parameterization: JointParameterization = Field(
        default="centred",
        description=(
            "Coordinates the sampler moves in; the posterior is the same in all three, only "
            "its geometry changes. centred samples each path directly, standardized by the "
            "prior's per-step covariances at the starting parameters. non_centred samples each "
            "free coordinate's Euler-Maruyama innovation and rebuilds the path from the current "
            "parameters. mixed keeps coordinates with a direct point reading centred and makes "
            "the rest non-centred. Centred mixes well where data pin the states, non-centred "
            "where data are weak against the dynamics (Papaspiliopoulos, Roberts & Sköld 2007). "
            "Frozen per fit; changing it on a continued fit requires warmup."
        ),
    )
    target_accept_prob: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description=(
            "Mean acceptance statistic that dual averaging steers the step size towards during "
            "warmup (Hoffman & Gelman 2014, section 3.2). Raising it towards 0.9-0.99 shrinks "
            "the step, which removes divergences in strongly curved regions at the cost of "
            "longer trajectories. 0.8 is the NumPyro and Stan default."
        ),
    )
    trajectory: JointTrajectorySpec = Field(
        default_factory=NutsTrajectorySpec,
        description="NUTS chooses each trajectory's length; fixed runs a set number of steps.",
    )

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify native coordinates and the trajectory law required by continuation."""
        trajectory = self.trajectory
        return (
            self.method,
            self.parameterization,
            *(
                (self.smoother.method,)
                if isinstance(self.smoother, JointSmootherSpec)
                else self.smoother.kernel_identity
            ),
            trajectory.kind,
            trajectory.max_tree_depth
            if isinstance(trajectory, NutsTrajectorySpec)
            else trajectory.steps,
        )


class MarginalParticleGibbsSpec(Value):
    """Exact marginalized parameter and path updates with particle-specific tuning."""

    method: Literal["mpgibbs"] = "mpgibbs"
    n_parameter_particles: ParticleCount = Field(
        default=8,
        description=(
            "Parameter particles M per sweep: the current value and M - 1 proposals around "
            "one auxiliary point. Work grows as O(TNM). Must be a power of two. Particle-mGRAD "
            f"allows M up to {MGRAD_PARAMETER_PARTICLE_LIMIT} and NM up to {MGRAD_BANK_LIMIT}, "
            f"or {MGRAD_LARGE_PARTICLE_BANK_LIMIT} above N = {MGRAD_LARGE_PARTICLE_COUNT}, the "
            "shapes its fused GPU tail covers. Heuristic: default M = 8; no ESS advantage has "
            "been established."
        ),
    )
    path_update: PathUpdateSpec = Field(
        default_factory=PathUpdateSpec,
        description=(
            "Exact path update: particle_mgrad (default) or twisted_particle_agrad, both "
            "guided, or auxiliary_kalman."
        ),
    )
    parameter_proposal: ParameterProposal = Field(
        default="random_walk",
        description=(
            "random_walk splits a preconditioned Gaussian step into two symmetric halves "
            "through the auxiliary point. pseudo_langevin drifts the first half along the "
            "log-posterior gradient at a path fixed when the kernel is built, and corrects the "
            "asymmetry exactly in the selection weights; the current path would break that "
            "correction (Corenflos 2025, section 3.1). It is kept as the control for gradients "
            "within the same kernel."
        ),
    )
    diagnostic_metrics: ParticleDiagnosticMetrics = Field(
        default=(),
        description=(
            "Optional traces to record: particle_identity (the selected particle per time "
            "step, reference-path hit rate and distinct selected particles), "
            "parameter_movement (RMS parameter jump) and particle_weights (filter weight ESS, "
            "entropy and spread at each time step). frequency_refresh summarizes retained "
            "path movement in DCT bands; regional_movement summarizes movement and frozen "
            "step sizes over existing question windows and reading coverage, with the ten worst times. They cost memory and output size, and no "
            "convergence check reads them; they explain a failing sampler."
        ),
    )
    param_target_accept: float = Field(
        default=0.2,
        description=(
            "Target fraction of sweeps in which the parameters move to a proposed particle. "
            "Selection is a Barker-type step (Corenflos 2025, section 3.3), whose optimal rate "
            "for random-walk proposals is about 0.27 in one dimension and 0.158 in high "
            "dimension (Agrawal et al. 2023). The default 0.2 sits between those rates and near "
            "the 18% that marginal Particle Gibbs reaches in Corenflos (2025, figure 1). At 0.35, "
            "warmup on a six-parameter linear PARAM ended at 43% acceptance with moves of 0.3 "
            "posterior SD, which costs an ideal Barker walk about 3.5 times the sweeps per "
            "effective draw of the optimal step."
        ),
    )
    adaptation_scheme: AdaptationScheme = Field(
        default="dual_averaging",
        description=(
            "How warmup adapts the parameter step: dual_averaging (default) keeps the averaged "
            "step at the end of warmup (Hoffman & Gelman 2014, section 3.2); simple uses a "
            "constant gain and keeps the last step."
        ),
    )
    initialization: GaussianInitializationSpec = Field(
        default_factory=GaussianInitializationSpec,
        description="Independent Gaussian starts, which also set the initial proposal scale.",
    )

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify kernels whose private state has the same scientific interpretation."""
        path_update = self.path_update
        return (
            "mpgibbs",
            *path_update.kernel_identity,
            self.parameter_proposal,
        )


type InferenceAlgorithmSpec = Annotated[
    MarginalParticleGibbsSpec | NutsDASpec, Field(discriminator="method")
]


class SamplerSpec(Value):
    """Fully resolved controls for exact joint state and parameter inference.

    The default algorithm is marginalized particle Gibbs. m-PGibbs (Corenflos 2025) proposes
    parameter particles around a shared auxiliary point, runs conditional SMC against the
    posterior mixture over them and selects a parameter together with the new path.
    Parameter moves are thus marginalized over paths, as in PMMH, instead of conditioned on
    one. The proposal covariance starts from the initializer and is re-estimated at the end
    of each warmup window (section 3.4).
    """

    num_warmup: int = Field(
        default=4000,
        ge=0,
        description=(
            "Warmup sweeps per chain. Every adaptive step, scale and covariance is "
            "learned here and frozen before draws are retained. Zero keeps the tuning as it "
            "is: a continued fit's frozen tuning, or the starting values of a new fit. With "
            "zero samples as well, both sweep programs compile and neither runs. Changing the "
            "sampler kernel on a continued fit requires warmup. The default is a heuristic; no "
            "paper or recorded benchmark sets it. Corenflos & Särkkä (2025, section 4) adapt for "
            "2,500 to 5,000 sweeps and Corenflos & Finke (2024, section 5.2) for 10,000."
        ),
    )
    num_samples_per_chain: int = Field(
        default=1000,
        ge=0,
        description=(
            "Retained draws per chain after warmup. Every fit runs this exact count; "
            "a continued fit appends this many draws to each chain. "
            "Zero returns checks without a conditioned model. The default is a heuristic; no "
            "paper or recorded benchmark sets it."
        ),
    )
    num_chains: int = Field(
        default=4,
        ge=1,
        description=(
            "Independent chains, which the convergence checks compare: rank-normalized R-hat, "
            "and bulk and tail ESS of at least 400 in total, however many chains run. "
            "Vehtari et al. (2021) recommend at least four."
        ),
    )
    thinning: bool = Field(
        default=True,
        description=(
            "Cap transferred posterior draws at 10,000,000 joint draw values using the "
            "smallest uniform integer stride; requests below the cap are unchanged. "
            "Modal retains the full result for analysis and continuation. R-hat, ESS, "
            "the convergence certificate, marginal summaries and rank histograms describe "
            "all retained draws pooled across every segment, computed before thinning. "
            "Transferred draws, traces and aligned telemetry are a subset; native last "
            "states are unchanged. Continuations thin the combined full cache from the "
            "same first draw of each chain. The actual completed count determines the "
            "stride, including an early checkpoint stop. If one draw per chain exceeds the "
            "heuristic cap, retain one per chain and report that stride and count. "
            "False transfers every draw."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Seeds initialization and sampling; another seed is an independent replicate.",
    )
    hardware: InferenceHardware = Field(
        default=InferenceHardware.L40S,
        description=(
            "local runs in the server process; any other value requests that Modal GPU type. "
            "Modal may run A100 on an 80 GB card at the same price, and H100 on an H200 unless "
            "it is written H100!. The fit records the device it ran on."
        ),
    )
    time_limit_seconds: int = Field(
        default=14400,
        ge=1,
        description=(
            "Modal startup and execution timeout per phase. Reaching it fails the fit, with no "
            "retry. Local fits have no enforced timeout."
        ),
    )
    algorithm: InferenceAlgorithmSpec = Field(
        default_factory=MarginalParticleGibbsSpec,
        description="Exact joint sampler: MPGibbs parameter/path updates or NUTS-DA.",
    )

    @property
    def support_failure(self) -> UnsupportedPathUpdate | None:
        """Reject unsupported path choices before compilation or remote dispatch."""
        algorithm = self.algorithm
        path = algorithm.smoother if isinstance(algorithm, NutsDASpec) else algorithm.path_update
        if isinstance(path, JointSmootherSpec):
            return None
        method = path.method
        return path.support_failure or (
            _mgrad_bank_failure(
                method.num_particles,
                1 if isinstance(algorithm, NutsDASpec) else algorithm.n_parameter_particles,
            )
            if isinstance(method, GuidedMethodSpec) and method.kind == "particle_mgrad"
            else None
        )

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify the algorithm whose private state the checkpoint retains."""
        return self.algorithm.kernel_identity
