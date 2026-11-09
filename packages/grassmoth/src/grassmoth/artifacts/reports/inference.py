"""Persisted posterior results and sampling metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.identity import DistributionId, ParameterId
from grassmoth.artifacts.reports.checks import SpecificationAssessment
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ChainDiagnostics,
    ConvergenceReport,
    DensityCurve,
    LOODiagnostics,
    LOOPITPoint,
    ParetoKPoint,
    PosteriorMarginal,
    RankHistogram,
    SamplerDiagnostics,
    TraceSeries,
)
from grassmoth.artifacts.reports.scorecard import FrequencyRefresh, RegionalMovement
from grassmoth.artifacts.sampling import SamplerSpec

type PhaseOutcome = Literal["complete", "failed"]


class InferencePhase(Value):
    """Measured interval relative to this execution's start; intervals may overlap."""

    sampler: str
    stage: str
    start_seconds: float
    end_seconds: float
    outcome: PhaseOutcome


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
    diagnostics: ChainDiagnostics | None = Field(
        description="Production diagnostics for this execution's retained draws alone."
    )
    convergence: ConvergenceReport = Field(
        description="Production convergence assessment for this segment, before combining chains."
    )


class InferenceMetadata(Value):
    """The production run's law, chain layout and sampler measurements."""

    distribution: DistributionId | None
    num_samples_total: int = Field(ge=0)
    thinning_factor: int = Field(
        default=1,
        ge=1,
        description="Uniform stride in this result's per-draw arrays. The certificate and summaries describe all full retained chains pooled across segments; transferred draws may be a thinned subset. The full Modal cache has stride one.",
    )
    draws_sampled_per_chain: int = Field(
        ge=0,
        description="Completed retained sweeps per chain across all segments, excluding warmup.",
    )
    draws_kept_per_chain: int = Field(
        ge=0,
        description="Draws per chain present in this result's per-draw arrays after transfer thinning.",
    )
    segments: tuple[FitSegment, ...] = ()
    num_chains: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    sampler_diagnostics: SamplerDiagnostics | None

    def with_transfer_counts(
        self, *, thinning_factor: int, draws_kept_per_chain: int, num_samples_total: int
    ) -> Self:
        """Return metadata describing the thinned transfer buffers."""
        return self._revised(
            thinning_factor=thinning_factor,
            draws_kept_per_chain=draws_kept_per_chain,
            num_samples_total=num_samples_total,
        )

    def with_run(self, *, duration_seconds: float, segments: tuple[FitSegment, ...]) -> Self:
        """Return metadata for the accumulated run duration and segments."""
        return self._revised(duration_seconds=duration_seconds, segments=segments)


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

    def with_previous_phases(self, previous: InferenceEvidence) -> InferenceEvidence:
        """Retain earlier phase buffers under one additional continuation prefix."""
        return self._revised(
            phase_extra_fields={
                **{
                    f"previous/{name}": values
                    for name, values in previous.phase_extra_fields.items()
                },
                **self.phase_extra_fields,
            }
        )

    def with_transfer_buffers(
        self,
        *,
        chain_extra_fields: Mapping[str, NumericalArray],
        observation_log_probs: NumericalArray | None,
        phase_extra_fields: Mapping[str, Mapping[str, NumericalArray]],
        warmup_complete_log_posterior_history: NumericalArray | None,
        all_complete_log_posterior_history: NumericalArray | None,
    ) -> Self:
        """Return evidence with the buffers selected for transfer."""
        return self._revised(
            chain_extra_fields=chain_extra_fields,
            observation_log_probs=observation_log_probs,
            phase_extra_fields=phase_extra_fields,
            warmup_complete_log_posterior_history=warmup_complete_log_posterior_history,
            all_complete_log_posterior_history=all_complete_log_posterior_history,
        )

    def with_retained_buffers(
        self,
        *,
        chain_extra_fields: Mapping[str, NumericalArray],
        observation_log_probs: NumericalArray | None,
    ) -> Self:
        """Return evidence with the retained chain diagnostics and observation scores."""
        return self._revised(
            chain_extra_fields=chain_extra_fields, observation_log_probs=observation_log_probs
        )

    def with_phases(
        self, *, phase_extra_fields: Mapping[str, Mapping[str, NumericalArray]]
    ) -> Self:
        """Return evidence with the accumulated phase buffers."""
        return self._revised(phase_extra_fields=phase_extra_fields)


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
    frequency_refresh: tuple[FrequencyRefresh, ...] = ()
    regional_movement: tuple[RegionalMovement, ...] = ()

    def with_metadata(self, *, inference_metadata: InferenceMetadata) -> Self:
        """Return the report core with updated run metadata."""
        return self._revised(inference_metadata=inference_metadata)


class InferenceReportDetail(Value):
    """Retained plot series served in full by the action result."""

    trace_data: tuple[TraceSeries, ...] = ()
    rank_histograms: tuple[RankHistogram, ...] = ()
    pareto_k: tuple[ParetoKPoint, ...] = ()
    loo_pit: tuple[LOOPITPoint, ...] = ()

    def with_traces(self, *, trace_data: tuple[TraceSeries, ...]) -> Self:
        """Return the report detail with its retained traces."""
        return self._revised(trace_data=trace_data)


class InferenceReport(Value):
    """One fit's provenance, run metadata, native evidence and computed findings."""

    evidence: InferenceEvidence
    core: InferenceReportCore
    detail: InferenceReportDetail

    def with_transfer(
        self,
        *,
        evidence: InferenceEvidence,
        core: InferenceReportCore,
        detail: InferenceReportDetail,
    ) -> Self:
        """Return the report with aligned transfer buffers, metadata, and traces."""
        return self._revised(evidence=evidence, core=core, detail=detail)

    def with_run(self, *, evidence: InferenceEvidence, core: InferenceReportCore) -> Self:
        """Return the report with accumulated run evidence and metadata."""
        return self._revised(evidence=evidence, core=core)


class FitCheckReport(Value):
    """Compatibility and question findings owned by one completed fit."""

    data: DataProfileReport
    preflight: tuple[SpecificationAssessment, ...]
    question: QuestionCheckReport
