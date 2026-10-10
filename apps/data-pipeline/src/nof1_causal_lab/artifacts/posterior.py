"""Persisted posterior results and sampling metadata."""

from collections.abc import Mapping

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

from .arrays import NumericalArray
from .checks import SpecificationAssessment
from .identity import ConstructId, DistributionId, ParameterId
from .model_checks import QuestionCheckReport
from .posterior_diagnostics import (
    ChainDiagnostics,
    ConvergenceReport,
    DensityCurve,
    LOODiagnostics,
    LOOPITPoint,
    ParetoKPoint,
    ParticleMCMCEvidence,
    ParticleSamplerDiagnostics,
    PosteriorMarginal,
    RankHistogram,
    TraceSeries,
)
from .validation_report import DataProfileReport


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured particle sampler."""

    num_samples_per_chain: int | None = Field(default=None, ge=1)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    num_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)


class InferenceMetadata(Value):
    """The production run's law, chain layout and sampler measurements."""

    distribution: DistributionId
    num_samples_total: int = Field(ge=1)
    num_chains: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    engine: ParticleMCMCEvidence
    sampler_diagnostics: ParticleSamplerDiagnostics | None


class PosteriorDrawsInfo(Value):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    n_draws: int = Field(ge=1)
    state_ids: tuple[ConstructId, ...] = ()


class InferenceEvidence(Value):
    """Native telemetry buffers; posterior atoms and coordinates belong to the model."""

    chain_extra_fields: Mapping[str, NumericalArray] = Field(default_factory=dict)
    observation_log_probs: NumericalArray | None = None
    observed_rows: NumericalArray | None = None
    exact_observation_rows: NumericalArray | None = None
    phase_extra_fields: Mapping[str, Mapping[str, NumericalArray]] = Field(default_factory=dict)
    warmup_complete_log_posterior_history: NumericalArray | None = None
    all_complete_log_posterior_history: NumericalArray | None = None
    initial_latent_delta: NumericalArray | None = None
    final_latent_delta: NumericalArray | None = None


class InferenceReportCore(Value):
    """Compact scientific report shared by snapshots and the full report."""

    inference_metadata: InferenceMetadata
    inference_diagnostics: ChainDiagnostics | None
    convergence: ConvergenceReport
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: tuple[PosteriorMarginal, ...]
    prior_densities: Mapping[ParameterId, DensityCurve] = Field(
        description="Input laws evaluated on the quantity scale of the posterior summaries."
    )


class InferenceReportDetail(Value):
    """Retained plot series served in full by the action result."""

    trace_data: tuple[TraceSeries, ...] = ()
    rank_histograms: tuple[RankHistogram, ...] = ()
    pareto_k: tuple[ParetoKPoint, ...] = ()
    loo_pit: tuple[LOOPITPoint, ...] = ()


class InferenceReport(Value):
    """One fit's provenance, run metadata, native evidence and computed findings."""

    evidence: InferenceEvidence
    core: InferenceReportCore
    detail: InferenceReportDetail


class FitCheckReport(Value):
    """Compatibility and question findings owned by one completed fit."""

    data: DataProfileReport
    preflight: tuple[SpecificationAssessment, ...]
    question: QuestionCheckReport
