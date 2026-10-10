"""Predictive findings and scientific posterior summaries."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, ConfigDict, Field, FiniteFloat, computed_field, model_validator

from nof1_causal_lab.artifacts.arrays import ScalarValues
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.sampler_config import SamplerSpec

from .checks import (
    Assessment,
    ConvergenceSubject,
    Evaluated,
    IndicatorCheckSubject,
    LatentConvergenceSubject,
    NotEvaluated,
    NumericCriterionEvidence,
)
from .display_frames import central_frame
from .effects import HistogramBin
from .empirical import EmpiricalPoint
from .identity import ConstructRef, IndicatorId, ParameterRef


class ParameterDiagnostics(Value):
    """Measurements on one scientifically identified retained scalar chain."""

    parameter: str
    subject: ParameterRef
    r_hat: float | None
    ess_bulk: float | None
    ess_tail: float | None
    mcse_mean: float | None


class TraceSeries(Value):
    """Every retained draw, grouped in original chain order."""

    subject: ParameterRef
    chains: tuple[ScalarValues, ...]


class RankHistogram(Value):
    """Pooled-rank bin counts, grouped in original chain order."""

    subject: ParameterRef
    expected_per_bin: float
    chains: tuple[tuple[int, ...], ...]


class DensityCurve(Value):
    """Aligned density ordinates; the owning field distinguishes PDF samples from histogram heights."""

    x: tuple[float, ...] = ()
    density: tuple[Annotated[float, Field(ge=0)], ...] = ()

    @model_validator(mode="after")
    def aligned(self) -> "DensityCurve":
        """Require one density ordinate for every evaluation coordinate."""
        if len(self.x) != len(self.density):
            raise ValueError("Density coordinates and ordinates must align")
        return self


class EnergyDiagnostics(Value):
    """Energy distributions and the producer's chain-specific BFMI values."""

    energy_hist: DensityCurve
    energy_transition_hist: DensityCurve
    bfmi: tuple[float, ...]


class LatentDiagnostics(Value):
    """Worst chain estimators across a construct's independent, unobserved time coordinates."""

    subject: ConstructRef
    label: str
    coordinates: int
    r_hat: float | None
    ess_bulk: float | None
    ess_tail: float | None
    mcse_mean: float | None


class ChainDiagnostics(Value):
    """Compact retained-chain measurements; plot series compose the report detail."""

    num_chains: int
    num_samples_per_chain: int
    per_parameter: tuple[ParameterDiagnostics, ...]
    per_latent: tuple[LatentDiagnostics, ...] | None = None
    num_divergences: int | None = None
    divergence_rate: float | None = None
    tree_depth_mean: float | None = None
    tree_depth_max: int | None = None
    accept_prob_mean: float | None = None
    latent_accept_prob_mean: float | None = None
    parameter_accept_prob_mean: float | None = None
    energy: EnergyDiagnostics | None = None


type ConvergenceAssessmentSubject = (
    ConvergenceSubject
    | LatentConvergenceSubject
    | Literal["recorded_parameter_chains", "recorded_latent_chains"]
)


class ConvergenceReport(Value):
    """Recorded-chain criteria cover parameters and independent latent coordinates."""

    findings: tuple[Assessment[ConvergenceAssessmentSubject, NumericCriterionEvidence], ...]

    @computed_field
    @property
    def checked(self) -> int:
        """Number of distinct parameter and construct references with convergence findings."""
        return len(
            frozenset(
                item.subject.parameter
                if isinstance(item.subject, ConvergenceSubject)
                else item.subject.state
                for item in self.findings
                if isinstance(item.subject, (ConvergenceSubject, LatentConvergenceSubject))
            )
        )

    @computed_field
    @property
    def status(self) -> Literal["passed", "failed", "not_evaluated"]:
        """Convergence verdict with failed checks taking precedence over unevaluated checks."""
        if any(isinstance(item, Evaluated) and item.outcome == "failed" for item in self.findings):
            return "failed"
        if not self.findings or any(isinstance(item, NotEvaluated) for item in self.findings):
            return "not_evaluated"
        return "passed"

    @computed_field
    @property
    def messages(self) -> tuple[str, ...]:
        """Explanations for convergence checks that failed or could not be evaluated."""
        return tuple(
            item.detail
            if isinstance(item, NotEvaluated)
            else f"{item.evidence.criterion} fails for {item.evidence.note}: {item.evidence.value:g}"
            for item in self.findings
            if isinstance(item, NotEvaluated) or item.outcome != "passed"
        )


class ParetoKPoint(Value):
    """One PSIS influence measurement with its original row and scientific class."""

    timestep: int
    k: float | Literal["infinity", "-infinity", "undefined"]
    status: Literal["passed", "failed", "not_evaluated"]


class LOOPITPoint(Value):
    """A retained PIT value and its empirical cumulative probability."""

    pit: float
    ecdf: float


class LOODiagnostics(Value):
    """Exact-emission leave-one-measurement-row-out interpolation diagnostics."""

    elpd_loo: float
    p_loo: float
    se: float
    n_data_points: int
    n_bad_k: int | None = None
    n_warn_k: int | None = None
    pareto_warning_limit: float = 0.5
    pareto_failure_limit: float = 0.7


class ParticleMCMCEvidence(Value):
    """The production particle-MCMC target and its exact latent transition."""


class PosteriorMarginal(Value):
    """One parameter's posterior interval, scale and density plot."""

    model_config = ConfigDict(allow_inf_nan=False)
    parameter: str
    subject: ParameterRef
    density_curve: DensityCurve
    empirical: tuple[EmpiricalPoint, ...]
    mean: float
    lower: float
    upper: float
    interval_kind: Literal["hdi", "equal_tail"]
    interval_mass: float = Field(gt=0, lt=1)
    sd: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "PosteriorMarginal":
        """Reject posterior intervals whose lower bound exceeds their upper bound."""
        if self.lower > self.upper:
            raise ValueError("Posterior interval lower bound must not exceed its upper bound")
        return self


class PPCOverlay(Value):
    """A predictive overlay sets one indicator's observed values against simulated ones.

    It carries the predictive median and a few individual replicated series, the
    spaghetti plot of a visual predictive check.
    """

    times: tuple[FiniteFloat, ...]
    time_origin: AwareDatetime
    standardized: bool
    indicator_id: IndicatorId
    observed: tuple[float | None, ...]
    median: tuple[float | None, ...]
    spaghetti_draws: tuple[tuple[float | None, ...], ...] = Field(default_factory=tuple)

    @computed_field
    @property
    def frame(self) -> tuple[float, float] | None:
        """Plot range spanning the widest per-time central 95% of draws and every observed value."""
        return central_frame(self.spaghetti_draws, self.observed)

    @model_validator(mode="after")
    def aligned_schedule(self) -> "PPCOverlay":
        """Require predictive series to share a strictly increasing evaluation schedule."""
        if (
            len(self.times) != len(self.observed)
            or len(self.times) != len(self.median)
            or any(len(draw) != len(self.times) for draw in self.spaghetti_draws)
        ):
            raise ValueError("Predictive overlay columns must align with their evaluated schedule")
        if any(right <= left for left, right in zip(self.times, self.times[1:], strict=False)):
            raise ValueError("Predictive overlay times must increase")
        return self


class PPCTestStat(Value):
    """An observed summary compared with the same statistic across predictive replicates.

    Provides the data for Gabry's ppc_stat plots: a histogram of T(y_rep)
    with a vertical line at T(y_observed).
    """

    indicator_id: IndicatorId
    stat_name: Literal["mean", "sd", "min", "max"]
    observed_value: float
    rep_values: tuple[float, ...]
    p_value: float | None
    histogram: tuple[HistogramBin, ...]

    @computed_field
    @property
    def frame(self) -> tuple[float, float] | None:
        """Value range charts show: the replicates' central 95%, covering the observed value."""
        return central_frame(((value,) for value in self.rep_values), (self.observed_value,))


class PosteriorPredictiveChecks(Value):
    """Posterior predictive checks report exact-model checks and their supporting plot data."""

    findings: tuple[Assessment[IndicatorCheckSubject, NumericCriterionEvidence], ...] = Field(
        default_factory=tuple
    )
    n_subsample: int = 0
    overlays: tuple[PPCOverlay, ...] = Field(default_factory=tuple)
    test_stats: tuple[PPCTestStat, ...] = Field(default_factory=tuple)


class ElboScoringDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    n_elbo_batch_evaluations: int
    n_elbo_screen_candidates: int
    n_elbo_refine_candidates: int
    best_elbo_candidate_index: int


class PathfinderStartDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    n_elbo_batch_evaluations: int
    n_elbo_screen_candidates: int
    n_elbo_refine_candidates: int
    best_elbo_candidate_index: int

    start_idx: int
    n_trajectory_points: int
    n_valid_iterates: int
    n_elbo_candidates: int
    n_lbfgs_iterations: int
    final_log_posterior: float
    best_elbo_this_start: float | None
    scipy_success: bool
    scipy_status: int


class PathfinderDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    n_pathfinder_starts: int
    n_pathfinder_starts_finite: int
    pathfinder_parallel_workers: int
    pathfinder_setup_seconds: float
    pathfinder_jax_compile_seconds: float
    pathfinder_jax_compile_batch_sizes: tuple[int, ...]
    pathfinder_runtime_seconds: float
    pathfinder_total_seconds: float
    best_pathfinder_elbo: float
    pathfinder_elbo_min: float
    pathfinder_elbo_max: float
    pathfinder_elbo_spread: float
    pathfinder_elbos: tuple[float, ...]
    pathfinder_maxiter: int
    pathfinder_lbfgs_memory: int
    pathfinder_elbo_samples: int
    pathfinder_elbo_screen_samples: int
    pathfinder_elbo_refine_candidates: int
    pathfinder_elbo_candidate_batch_size: int
    pathfinder_per_start: tuple[PathfinderStartDiagnostics, ...]


class LapsInitializationDiagnostics(Value):
    """Realized joint-ensemble initialization, without any claim of posterior convergence."""

    ensemble_size: int
    joint_dimension: int
    unadjusted_steps: int
    adjusted_steps: int
    unadjusted_step_size: float
    adjusted_step_size: float
    adjusted_acceptance: float
    early_stopped: bool


class ParameterWarmupDiagnostics(Value):
    """Realized initialization and preconditioning, retaining each initializer's evidence once."""

    pathfinder: PathfinderDiagnostics | None = None
    laps: LapsInitializationDiagnostics | None = None
    pathfinder_run_count: int
    pathfinder_consumers: tuple[str, ...]
    init_source: str
    preconditioner_source: str
    preconditioner_device: str | None = None
    dim: int
    duration_seconds: float
    pathfinder_sampling_mode: str | None = None
    pathfinder_init_scale: float | None = None
    prior_released_site_names: tuple[str, ...] = ()
    prior_released_site_indices: tuple[int, ...] = ()
    prior_release_scale: float = 0.0


class ReplicaSwapCounts(Value):
    """Accepted and attempted exchanges at one neighboring temperature pair."""

    accepted: int
    attempted: int


class TemperingDiagnostics(Value):
    """Frozen ladder and retained swap counts; columns are neighboring temperature edges."""

    inverse_temperatures: tuple[float, ...]
    ladder_frozen_at_warmup_step: int
    swaps: tuple[tuple[ReplicaSwapCounts, ...], ...]


class ParticleSamplerDiagnostics(Value):
    """Typed exact-sampler settings and transition telemetry from the native producer."""

    settings: SamplerSpec
    parameter_kernel: str
    mcmc_phase_seconds: float
    dsmc_leaf_proposal: str
    diagnostic_metrics: tuple[str, ...]
    param_target_accept: float
    parameter_preconditioned: bool
    diagnostic_summary_phase: str
    parameter_accept_rate: float
    latent_update_fraction: float
    latent_frozen_fraction: float
    latent_block_coords: int | None
    initial_param_step_size: tuple[float, ...]
    final_param_step_size: tuple[float, ...]
    latent_sign_flip_moves: bool | None = None
    chain_post_warmup_complete_log_posterior_mean: tuple[float, ...]
    latent_move_rms_mean: float | None = None
    parameter_jump_rms_mean: float | None = None
    reference_path_hit_rate_mean: float | None = None
    selected_particle_unique_count_mean: float | None = None
    amala_grad_norm_mean: float | None = None
    amala_grad_norm_max: float | None = None
    parameter_warmup: ParameterWarmupDiagnostics
    tempering: TemperingDiagnostics | None = None
