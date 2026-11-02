"""Owned sampler choices and explicit numerical initialization inputs.

Each field description says what the choice does and why it is offered. Defaults
are starting values; adaptation constants not listed here are fixed.
"""

# References:
#   https://doi.org/10.1111/j.1467-9868.2009.00736.x — Andrieu, Doucet & Holenstein (2010),
#     "Particle Markov chain Monte Carlo methods": Particle Gibbs (section 2.4.3).
#   https://arxiv.org/abs/2303.00301 — Corenflos & Särkkä (2025), "Auxiliary MCMC samplers for
#     parallelisable inference in high-dimensional latent dynamical systems": Gaussian path
#     proposals (2.2-2.4) and per-time-step targets (4).
#   https://arxiv.org/abs/2401.14868 — Corenflos & Finke (2024), "Particle-MALA and
#     Particle-mGRAD": Particle-aGRAD (Algorithm 6, twisted in 4.4) and Particle-mGRAD
#     (Algorithm 7).
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), "Particle Gibbs without the Gibbs bit":
#     the auxiliary proposal (3.1), Barker selection (3.3), adaptation (3.4) and M = 2 (5).
#   https://arxiv.org/abs/1111.4246 — Hoffman & Gelman (2014), "The No-U-Turn Sampler": dual
#     averaging (3.2).
#   https://doi.org/10.1214/aoap/1034625254 — Roberts, Gelman & Gilks (1997): optimal scaling.
#   https://arxiv.org/abs/2104.02020 — Agrawal, Vats, Łatuszyński & Roberts (2023): optimal
#     scaling of Barker's acceptance.
#   https://doi.org/10.1214/20-BA1221 — Vehtari et al. (2021): rank-normalized R-hat.

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

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
            "moving."
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
            "and MPGibbs re-estimates it in warmup windows."
        ),
    )


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

    method: Literal["particle_mgrad", "twisted_particle_agrad"] = Field(
        default="particle_mgrad",
        description=(
            "particle_mgrad integrates the path auxiliary out of the particle weights with the "
            "untwisted proposal (section 4.2); it supports point readings and deterministic "
            "recurrences, and separated path components and informative future readings can "
            "slow its mixing. twisted_particle_agrad keeps the auxiliary variable "
            "in the weights (section 4.1) and adds a Gaussian look-ahead built at the auxiliary "
            "path: a parallel backward information scan guides the sequential filter, and "
            "proposals combine exact nonlinear transition means with the approximate twist, "
            "whose normalized densities and twist ratios are paid in full (section 4.4). It "
            "also supports interval readings; poor local fits and multiple modes cost "
            "efficiency."
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


class AuxiliaryKalmanSpec(Value):
    """Exact whole-path or blocked Metropolis update with parallel Gaussian proposals.

    Corenflos & Särkkä (2025), https://arxiv.org/abs/2303.00301, sections 2.2–2.4.
    Whole-path forward and reverse approximations are rebuilt at their respective
    paths. Blocks instead linearize at the fixed auxiliary under a sampled label.
    Normalized proposal densities correct to the true nonlinear target.
    """

    method: Literal["auxiliary_kalman"] = "auxiliary_kalman"
    # 300 led path movement among 30, 100 and 300 in short probes on 3,000 days of stochastic
    # volatility. At full length it gave 3.6 times the whole path's worst bulk ESS per second;
    # the whole path's single accept/reject left R-hat at 1.16. On Lorenz-96's 300 steps,
    # 25-step blocks gave a quarter of the whole path's, so retune per model.
    block_length: int | None = Field(
        default=300,
        ge=1,
        description="Time points per block; null updates the whole path. Blocks use a conditional parameter label, alternating colours and random half-block shifts. Each block pays its exact nonlinear boundary and interior densities.",
    )
    latent_step: LatentStepSpec = Field(
        default_factory=AdaptedLatentStepSpec,
        description=(
            "Auxiliary variance delta/2, fixed or adapted during warmup using per-time-point "
            "acceptance. Parallel Kalman operations draw Gaussian paths or conditional blocks. "
            "Poor local Gaussian fits, strong nonlinearity and separated parameter or "
            "path modes can lower acceptance. Exact nonlinear Metropolis corrections "
            "preserve the target. The whole path also supports interval readings and "
            "deterministic recurrences, which every label replays exactly; blocks support "
            "point readings without recurrences. Parameter proposals and final "
            "parameter-label selection remain MPGibbs."
        ),
    )


type PathUpdateSpec = Annotated[GuidedPathSpec | AuxiliaryKalmanSpec, Field(discriminator="method")]


class SamplerSpec(Value):
    """Fully resolved controls for exact joint state and parameter inference.

    The sampler is marginalized particle Gibbs. m-PGibbs (Corenflos 2025) proposes
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
            "sampler kernel on a continued fit requires warmup."
        ),
    )
    num_samples_per_chain: int = Field(
        default=1000,
        ge=0,
        description=(
            "Retained draws per chain after warmup. Every fit runs this exact count; "
            "a continued fit appends this many draws to each chain. "
            "Zero returns checks without a conditioned model."
        ),
    )
    num_chains: int = Field(
        default=4,
        ge=1,
        description=(
            "Independent chains, which the convergence checks compare: rank-normalized R-hat, "
            "and bulk and tail ESS of at least 100 per chain. Vehtari et al. (2021) recommend "
            "at least four."
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
    n_parameter_particles: int = Field(
        default=2,
        ge=2,
        description=(
            "Parameter particles M per sweep: the current value and M - 1 proposals around "
            "one auxiliary point. Work grows as O(TNM); Corenflos (2025, section 5) recommends "
            "M = 2, given the weak performance of larger ensembles (Pozza & Zanella 2024)."
        ),
    )
    path_update: PathUpdateSpec = Field(
        default_factory=GuidedPathSpec,
        description=(
            "Exact path update: particle_mgrad (default) or twisted_particle_agrad, both "
            "guided, or auxiliary_kalman."
        ),
    )
    compiler_reparameterization: bool = Field(
        default=False,
        description=(
            "Compute time-indexed path translations from Gauss-Newton mean cross curvature, "
            "including deterministic recurrences and interval readings. Build at mean initial "
            "parameters and their IEKS path; rebuild at the chains' mean parameters and their "
            "IEKS path at warmup window ends. Freeze before retained draws and preserve the "
            "chart in native continuation state. Every density and reported path retains "
            "the exact physical model. Local slopes, strong nonlinearity and multiple modes "
            "can limit mixing."
        ),
    )
    scale_chart: bool = Field(
        default=False,
        description=(
            "Rescale each stochastic state's deviations from a center path by the change in "
            "its own diffusion scale, in the sampler's actual parameter coordinates: structural "
            "non-centring of the diffusion. The center is the mean warmup path in the translated "
            "chart, learned in the existing warmup windows and frozen before retained draws. "
            "Every label pays the exact log-Jacobian, and all densities and reported paths "
            "retain the exact physical model. Requires point readings "
            "without deterministic recurrences. Data-pinned deviations, a poor center or a "
            "state-dependent diffusion cost efficiency, never the target."
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

    @property
    def kernel_identity(self) -> tuple[str | int, ...]:
        """Identify kernels whose private state has the same scientific interpretation."""
        path_update = self.path_update
        guided = isinstance(path_update, GuidedPathSpec)
        return (
            path_update.method,
            path_update.kappa if guided else 0,
            path_update.prefix_mixture if guided else False,
            self.parameter_proposal,
            self.compiler_reparameterization,
            self.scale_chart,
        )
