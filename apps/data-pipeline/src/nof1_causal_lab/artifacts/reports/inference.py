"""Persisted posterior results and sampling metadata."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
from nof1_causal_lab.artifacts.identity import ConstructId, DistributionId, ParameterId
from nof1_causal_lab.artifacts.joint_parameterization import JointParameterizationRecord
from nof1_causal_lab.artifacts.reports.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import (
    ChainDiagnostics,
    ConvergenceReport,
    DensityCurve,
    LOODiagnostics,
    LOOPITPoint,
    MCMCEvidence,
    NutsDADiagnostics,
    ParetoKPoint,
    ParticleGibbsDiagnostics,
    ParticleSamplerDiagnostics,
    PosteriorMarginal,
    RankHistogram,
    TraceSeries,
)
from nof1_causal_lab.artifacts.sampling import SamplerSpec


class InferencePhase(Value):
    """Measured interval relative to this execution's start; intervals may overlap."""

    sampler: str
    stage: str
    start_seconds: float
    end_seconds: float
    outcome: Literal["complete", "failed"]


class FitRuntime(Value):
    """Actual numerical source, environment, compiler settings and devices."""

    source_digest: str
    versions: Mapping[str, str]
    compiler: Mapping[str, str]
    devices: tuple[str, ...]
    time_limit_seconds: int | None = Field(
        default=None, ge=1, description="Enforced Modal phase limit; null for local execution."
    )


class FitSegment(Value):
    """One execution's retained draw interval within every combined chain."""

    start: int = Field(ge=0)
    stop: int = Field(ge=0)
    settings: SamplerSpec
    runtime: FitRuntime
    phases: tuple[InferencePhase, ...]
    parameterization: JointParameterizationRecord | None = None


class InferenceMetadata(Value):
    """The production run's law, chain layout and sampler measurements."""

    distribution: DistributionId | None
    num_samples_total: int = Field(ge=0)
    segments: tuple[FitSegment, ...] = ()
    num_chains: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    engine: MCMCEvidence
    sampler_diagnostics: (
        ParticleSamplerDiagnostics | NutsDADiagnostics | ParticleGibbsDiagnostics | None
    )


class PosteriorDrawsInfo(Value):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    n_draws: int = Field(ge=1)
    state_ids: tuple[ConstructId, ...] = ()


class InferenceEvidence(Value):
    """Native telemetry buffers; posterior atoms and coordinates belong to the model."""

    checkpoint: SamplerCheckpoint | None = None
    latent_free_mask: NumericalArray | None = None
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
