"""Owned sampler choices and explicit numerical initialization inputs.

Each field description says what the choice does and why it is offered. Defaults
are starting values; adaptation constants not listed here are fixed.
"""

# References:
#   https://doi.org/10.1111/j.1467-9868.2009.00736.x — Andrieu, Doucet & Holenstein (2010),
#     "Particle Markov chain Monte Carlo methods": Particle Gibbs (section 2.4.3).
#   https://arxiv.org/abs/2202.02264 — Corenflos, Chopin & Särkkä (2022), "De-Sequentialized
#     Monte Carlo": the conditional tree (Algorithm 4); proposals and N x N merges (section 4).
#   https://arxiv.org/abs/2303.00301 — Corenflos & Särkkä (2025), "Auxiliary MCMC samplers for
#     parallelisable inference in high-dimensional latent dynamical systems": blocking (3.1),
#     per-time-step targets (4), GPU efficiency (4.1) and failure modes (4.4).
#   https://arxiv.org/abs/2401.14868 — Corenflos & Finke (2024), "Particle-MALA and
#     Particle-mGRAD": Particle-aMALA (Algorithm 3) and Particle-aMALA+ (Algorithm 5).
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), "Particle Gibbs without the Gibbs bit":
#     the auxiliary proposal (3.1), Barker selection (3.3), adaptation (3.4) and M = 2 (5).
#   https://arxiv.org/abs/1905.02939 — Syed, Bouchard-Côté, Deligiannidis & Doucet (2022),
#     "Non-reversible parallel tempering": round trips (4.2), schedule (5.1-5.2), size (5.3).
#   https://arxiv.org/abs/1111.4246 — Hoffman & Gelman (2014), "The No-U-Turn Sampler".
#   https://proceedings.mlr.press/v130/hoffman21a.html — Hoffman, Radul & Sountsov (2021):
#     NUTS's control flow limits many parallel chains on accelerators.
#   https://arxiv.org/abs/1206.1901 — Neal (2011), "MCMC using Hamiltonian dynamics" (5.3.2).
#   https://doi.org/10.1214/088342307000000014 — Papaspiliopoulos, Roberts & Sköld (2007):
#     centred and non-centred parameterizations.
#   https://doi.org/10.1214/aoap/1034625254 — Roberts, Gelman & Gilks (1997): optimal scaling.
#   https://arxiv.org/abs/2104.02020 — Agrawal, Vats, Łatuszyński & Roberts (2023): optimal
#     scaling of Barker's acceptance.
#   https://doi.org/10.1080/00401706.1995.10484303 — Hesterberg (1995): defensive mixtures.
#   https://doi.org/10.1214/20-BA1221 — Vehtari et al. (2021): rank-normalized R-hat.

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from grassmoth.artifacts.base import Value

type AuxiliaryDrift = Literal["none", "amala", "amala_plus"]
type LatentBlockSize = Literal[1, 2, "all"]
type JointParameterization = Literal["centred", "non_centred", "mixed"]
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
            "Variance of the candidates around each proposed state, at every time step: half "
            "comes from refreshing the auxiliary point and half from drawing around it "
            "(Corenflos & Finke 2024, Algorithm 3). It is in the state's own units and shared "
            "by all its coordinates. Fix it to compare proposals at a matched scale, or when "
            "adaptation settles on a scale at which the path stops moving."
        ),
    )


type LatentStepSpec = Annotated[
    AdaptedLatentStepSpec | FixedLatentStepSpec, Field(discriminator="kind")
]


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
            "Leapfrog steps per transition. Each transition scales the adapted step size by a "
            "uniform factor in [0.8, 1.2], so trajectories cannot stay periodic (Neal 2011, "
            "section 5.3.2). Every transition costs the same number of gradients, so chains "
            "advanced together never wait for the longest NUTS tree, which limits NUTS on "
            "accelerators (Hoffman, Radul & Sountsov 2021)."
        ),
    )


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
    """Joint HMC with dual averaging and windowed diagonal mass adaptation.

    It moves every parameter and free latent coordinate at once on the exact joint
    density, with no particle approximation. It also reconstructs deterministic
    recurrences that the particle samplers reject, and is a diagnostic when particle
    chains fail to converge.
    """

    method: Literal["nuts_da"] = "nuts_da"
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


class UntemperedSpec(Value):
    """The exact posterior without temperature replicas."""

    kind: Literal["untempered"] = "untempered"


class GaussianReferenceTemperingSpec(Value):
    """Replica exchange; each ladder retains only its exact beta-one chain.

    The ladder spans beta one to zero. The reference is fixed at initialization,
    its endpoint is independently refreshed every sweep, and even/odd neighboring
    swaps alternate. Rejection-based ladder adaptation runs during the first half
    of warmup, leaving the rest to settle before retaining posterior draws.

    It is for posteriors whose chains stay in separate modes: states that move freely
    under the reference reach the exact chain through swaps (Syed et al. 2022). Its
    limits are in docs/assumptions.md, under Model class.
    """

    kind: Literal["gaussian_reference"] = "gaussian_reference"
    num_temperatures: int = Field(
        default=48,
        ge=2,
        description=(
            "Replicas per chain, initially evenly spaced from beta one (the exact posterior) "
            "to beta zero (the Gaussian reference). Each replica runs a full sweep, so cost "
            "grows linearly. With alternating even/odd swaps the round-trip rate tends to "
            "1 / (2 + 2Λ) instead of collapsing as replicas are added, and about 2Λ replicas "
            "are optimal, where Λ is the posterior's communication barrier (Syed et al. 2022, "
            "sections 4.2 and 5.3)."
        ),
    )
    reference_scale: float = Field(
        default=1.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Multiplies the reference's standard deviations for parameters and path "
            "innovations. The reference is built at initialization from the starting draws "
            "and the transition Jacobians along their mean path. Above 1 it covers mass the "
            "starting draws missed, but overlaps the posterior less, which raises the barrier "
            "and so the replicas needed."
        ),
    )
    adaptation_interval: int = Field(
        default=500,
        ge=2,
        description=(
            "Sweeps between ladder updates in the first half of warmup. Each update respaces "
            "the temperatures to equalize the swap rejection rates estimated since the last "
            "update (Syed et al. 2022, sections 5.1-5.2). An interval above half the warmup "
            "leaves the evenly spaced ladder unadapted."
        ),
    )


type TemperingSpec = Annotated[
    UntemperedSpec | GaussianReferenceTemperingSpec, Field(discriminator="kind")
]


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
            "and MPGibbs re-estimates it in warmup windows."
        ),
    )


class DsmcSpec(Value):
    """Conditional dSMC auxiliary law, particle proposal, and assigned scale policy.

    Each sweep refreshes an auxiliary point near the current path, draws candidates
    around it at every time step, and selects a new path through the conditional dSMC
    tree (Corenflos, Chopin & Särkkä 2022, Algorithm 4), whose depth grows with log T.
    The auxiliary law is Particle-RWM or Particle-aMALA (Corenflos & Finke 2024).
    """

    method: Literal["particle_rwm", "particle_amala", "particle_amala_plus"] = Field(
        default="particle_rwm",
        description=(
            "Auxiliary law for conditional dSMC. particle_rwm (kappa 0) centres the auxiliary "
            "point on the current state. particle_amala (Algorithm 3) drifts it along the "
            "gradient of the local prior and observation terms; particle_amala_plus (Algorithm "
            "5) adds the next transition. With gradients the scale need only shrink like "
            "D^(-1/3) as the state dimension D grows, against D^(-1) without (Corenflos & "
            "Finke 2024, figure 1). There aMALA+ had up to three times Particle-RWM's ESS and "
            "twice aMALA's, for twice aMALA's gradients (section 5.4). On a GPU, Corenflos & "
            "Särkkä (2025) found gradient and plain proposals equally efficient per second "
            "(section 4.1), with gradients more robust to sticky dynamics and very informative "
            "observations (section 4.4)."
        ),
    )
    num_particles: int = Field(
        default=64,
        ge=2,
        description=(
            "Particles per sweep, including the current path. More particles give each time "
            "step more candidates, but each dSMC merge scores N x N candidate pairs, so memory "
            "and work per merge grow as N^2 (Corenflos, Chopin & Särkkä 2022, section 4)."
        ),
    )
    latent_step: LatentStepSpec = Field(
        default_factory=AdaptedLatentStepSpec,
        description="Per-time-step scale: adapted towards a movement target, or fixed.",
    )
    latent_block_coords: LatentBlockSize = Field(
        default=1,
        description=(
            "State coordinates proposed per sweep, drawn uniformly from the free ones; the "
            "others keep their current paths at every time step. 1 proposes one coordinate's "
            "whole path, all proposes the full state. Importance weights degenerate as the "
            "proposed dimension grows, which blocking counters (Corenflos & Särkkä 2025, "
            "section 3.1): smaller blocks keep larger scales at the same movement rate, but "
            "each coordinate is proposed only every D/k sweeps, and coupled coordinates move "
            "slowly when proposed apart."
        ),
    )
    pilot_mixture: bool = Field(
        default=True,
        description=(
            "Mix two fixed Gaussians into every candidate draw: 85% of candidates come from "
            "the local auxiliary proposal, 10% from a Gaussian at the initializer's mean path "
            "with a quarter of its variance, and 5% from one with four times its variance. "
            "Weights use the full mixture density, so the target is unchanged (a defensive "
            "mixture, Hesterberg 1995). It adds candidates away from the current path, as "
            "Corenflos, Chopin & Särkkä (2022, section 4) suggest for separable proposals. "
            "The auxiliary factor still scores each candidate against the refreshed point, so "
            "at a small scale distant pilot draws are rarely kept. The moments come from the "
            "num_chains starting paths."
        ),
    )

    @property
    def drift(self) -> AuxiliaryDrift:
        """The named auxiliary method owns its drift; dSMC owns path assembly."""
        match self.method:
            case "particle_rwm":
                return "none"
            case "particle_amala":
                return "amala"
            case "particle_amala_plus":
                return "amala_plus"


_GUIDED_PARTICLES = (
    "Particles per sweep, including the current path. Each extends a parent chosen at the "
    "previous time, so work grows linearly in particles and times, and the filter's depth "
    "grows linearly with the number of times (Corenflos & Finke 2024, section 4.7)."
)
_GUIDED_STEP = (
    "Per-time-step scale: adapted towards a movement target, or fixed. Small scales keep "
    "proposals near the current path; large ones let the dynamics dominate, approaching "
    "conditional SMC with the transition as proposal (Corenflos & Finke 2024, section 4.6)."
)
_GUIDED_KAPPA = (
    "1 shifts the auxiliary point along the gradient of the observation densities, mixed "
    "over parameter labels; 0 omits it (Corenflos & Finke 2024, Algorithm 6). The shift is "
    "a Langevin step on the observation term: for a Gaussian reading it overshoots once the "
    "scale exceeds four times that reading's variance."
)
# The default 8 was the fastest of 1, 4 and 8 for twisted Particle-aGRAD (kappa 1) on an A10,
# with identical transitions. Retune it per device and model size.
_GUIDED_SCAN_UNROLL = (
    "Time steps per iteration of the forward filter and backward-selection loops. Unrolling "
    "changes how XLA compiles these sequential loops, not the algorithm or its target. Each "
    "step runs small kernels, so on a GPU per-iteration overhead is a large share of a sweep; "
    "unrolling amortizes it at the cost of a larger program and longer compilation. A CPU has "
    "little such overhead to remove. 1 disables unrolling."
)


class GuidedPathSpec(Value):
    """Sequential conditional SMC with exact Gaussian transition-informed proposals.

    Corenflos & Finke (2024). Parameter labels remain marginalized, and exact nonlinear
    transitions and densities pay every weight, so each method's approximation costs
    mixing, never the target.
    """

    method: Literal["particle_agrad", "twisted_particle_agrad", "particle_mgrad"] = Field(
        default="particle_agrad",
        description=(
            "particle_agrad keeps the auxiliary variable in the weights (section 4.1); strong "
            "future information or multiple modes can reduce its mixing. "
            "twisted_particle_agrad adds a Gaussian look-ahead built at the auxiliary path: a "
            "parallel backward information scan guides the sequential filter, and proposals "
            "combine exact nonlinear transition means with the approximate twist, whose "
            "normalized densities and twist ratios are paid in full (section 4.4); poor local "
            "fits and multiple modes cost efficiency. particle_mgrad integrates the path "
            "auxiliary out of the particle weights with the untwisted proposal (section 4.2); it "
            "supports point readings without deterministic recurrences, and separated path "
            "components and informative future readings can slow its mixing."
        ),
    )
    num_particles: int = Field(default=64, ge=2, description=_GUIDED_PARTICLES)
    latent_step: LatentStepSpec = Field(
        default_factory=AdaptedLatentStepSpec, description=_GUIDED_STEP
    )
    kappa: Literal[0, 1] = Field(default=1, description=_GUIDED_KAPPA)
    scan_unroll: int = Field(default=8, ge=1, description=_GUIDED_SCAN_UNROLL)
    prefix_mixture: bool = Field(
        default=False,
        description=(
            "Opt in to per-parameter-label local Gaussian proposals, mixed with the "
            "unfloored prefix label probabilities. Every particle pays the full mixture "
            "density; the shared auxiliary law and look-ahead remain unchanged. Requires "
            "twisted_particle_agrad with kappa=1 and point readings without replay. "
            "Early label collapse, poor shared guidance and separated modes can reduce mixing."
        ),
    )


type PathUpdateSpec = Annotated[DsmcSpec | GuidedPathSpec, Field(discriminator="method")]


class MarginalParticleGibbsSpec(Value):
    """Marginalized parameter ensemble and exact conditional trajectory updates.

    m-PGibbs (Corenflos 2025), the default sampler, proposes parameter particles around
    a shared auxiliary point, runs conditional SMC against the posterior mixture over
    them and selects a parameter together with the new path. Parameter moves are thus
    marginalized over paths, as in PMMH, instead of conditioned on one. The proposal
    covariance starts from the initializer and is re-estimated at the end of each
    warmup window (section 3.4).
    """

    method: Literal["mpgibbs"] = "mpgibbs"
    n_parameter_particles: int = Field(
        default=2,
        ge=2,
        description=(
            "Parameter particles M per sweep: the current value and M - 1 proposals around "
            "one auxiliary point. Work grows as O(TNM); Corenflos (2025, section 5) recommends "
            "M = 2, given the weak performance of larger ensembles (Pozza & Zanella 2024)."
        ),
    )
    tempering: TemperingSpec = Field(
        default_factory=UntemperedSpec,
        description=(
            "untempered runs the exact posterior alone; "
            "gaussian_reference runs Gaussian-reference parallel tempering."
        ),
    )
    path_update: PathUpdateSpec = Field(default_factory=DsmcSpec, description="Exact path update.")
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
            "parameter_movement (RMS parameter jump) and particle_weights (dSMC weight ESS, "
            "entropy and spread per tree node). frequency_refresh summarizes retained "
            "path movement in DCT bands; regional_movement summarizes movement and frozen "
            "step sizes over existing question windows and reading coverage, with the ten worst times. They cost memory and output size, and no "
            "convergence check reads them; they explain a failing sampler."
        ),
    )
    # 0.2 sits between Barker's optimal rates for random-walk proposals and near the 18%
    # that m-PGibbs reaches in Corenflos (2025, figure 1). At 0.35, warmup on a
    # six-parameter linear PARAM ended at 43% acceptance with moves of 0.3 posterior SD,
    # which costs an ideal Barker walk about 3.5 times the sweeps per effective draw of
    # the optimal step.
    param_target_accept: float = Field(
        default=0.2,
        description=(
            "Target fraction of sweeps in which the parameters move to a proposed particle. "
            "Selection is a Barker-type step (Corenflos 2025, section 3.3), whose optimal rate "
            "for random-walk proposals is about 0.27 in one dimension and 0.158 in high "
            "dimension (Agrawal et al. 2023)."
        ),
    )
    adaptation_scheme: AdaptationScheme = Field(
        default="simple",
        description=(
            "How warmup adapts the parameter step: simple uses a constant gain and keeps the "
            "last step; dual_averaging keeps the averaged step at the end of warmup (Hoffman & "
            "Gelman 2014, section 3.2)."
        ),
    )
    initialization: GaussianInitializationSpec = Field(
        default_factory=GaussianInitializationSpec,
        description="Independent Gaussian starts, which also set the initial proposal scale.",
    )


type InferenceAlgorithmSpec = Annotated[
    NutsDASpec | MarginalParticleGibbsSpec, Field(discriminator="method")
]


class ConvergenceStoppingSpec(Value):
    """Declared retained-draw checkpoints for the native production convergence gate."""

    checkpoint_draws: int = Field(
        ge=1,
        description="Check after each multiple of this many newly retained draws per chain.",
    )
    min_draws: int = Field(
        ge=1,
        description=(
            "Minimum newly retained draws per chain before a passing checkpoint may stop this "
            "execution. The requested sample count remains the ceiling."
        ),
    )


class SamplerSpec(Value):
    """Fully resolved controls for exact joint state and parameter inference."""

    num_warmup: int = Field(
        default=4000,
        ge=0,
        description=(
            "Warmup sweeps per chain. Every adaptive step, scale, covariance and ladder is "
            "learned here and frozen before draws are retained. Zero keeps the tuning as it "
            "is: a continued fit's frozen tuning, or the starting values of a new fit. With "
            "zero samples as well, both sweep programs compile and neither runs. Switching "
            "the algorithm or joint parameterization on a continued fit requires warmup."
        ),
    )
    num_samples_per_chain: int = Field(
        default=1000,
        ge=0,
        description=(
            "Requested retained draws per chain after warmup, or a ceiling when "
            "stop_when_converged is set. A continued fit appends its actual retained draws. "
            "Zero returns checks without a conditioned model."
        ),
    )
    num_chains: int = Field(
        default=4,
        ge=1,
        description=(
            "Independent chains, which the convergence checks compare: rank-normalized R-hat, "
            "and bulk and tail ESS of at least 100 per chain. Vehtari et al. (2021) recommend "
            "at least four. Tempering runs one ladder per chain."
        ),
    )
    stop_when_converged: ConvergenceStoppingSpec | None = Field(
        default=None,
        description=(
            "Optional production convergence checks at declared retained-draw checkpoints. "
            "Warmup is unchanged; the first passing checkpoint at or above min_draws stops "
            "sampling. None retains the full requested count. Each continued execution checks "
            "only its newly retained draws."
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
        description="Exact joint sampler: mpgibbs (default) or nuts_da.",
    )

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify kernels whose private state has the same scientific interpretation."""
        algorithm = self.algorithm
        if isinstance(algorithm, NutsDASpec):
            return (algorithm.method, algorithm.trajectory.kind, algorithm.parameterization)
        tempering = algorithm.tempering
        return (
            algorithm.method,
            algorithm.path_update.method,
            0 if isinstance(algorithm.path_update, DsmcSpec) else algorithm.path_update.kappa,
            False
            if isinstance(algorithm.path_update, DsmcSpec)
            else algorithm.path_update.prefix_mixture,
            algorithm.parameter_proposal,
            1 if isinstance(tempering, UntemperedSpec) else tempering.num_temperatures,
        )
