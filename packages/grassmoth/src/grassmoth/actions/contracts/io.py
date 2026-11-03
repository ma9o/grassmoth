"""Resolved inputs and complete scientific outputs of the seven library actions.

Fit and simulate take input records, and prepare_data takes its source table and the
values its workers extracted; the other actions take artifact values directly. Typed
failures stay with the modules that construct them. The service resolves saved
references into these inputs and supplies HTTP request and polling envelopes; these
values are independent of their transport or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import (
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec, ModelEditResult
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.history import ObservationData
from grassmoth.artifacts.observations.preparation import (
    DataPreparationResult,
    PreparedDataMetadata,
)
from grassmoth.artifacts.reports.data_comparison import DataComparisonReport
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.artifacts.reports.inference import FitCheckReport, InferenceReport
from grassmoth.artifacts.reports.model_checks import ModelCheckReport
from grassmoth.artifacts.reports.profiling import FitProfileReport
from grassmoth.artifacts.simulation import SimulationReport

if TYPE_CHECKING:
    from datetime import date

    import pyarrow as pa

    from grassmoth.artifacts.identity import GitRef
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.artifacts.reports.inference import InferenceReportCore
    from grassmoth.artifacts.reports.predictive_provenance import (
        AuthoredLawProvenance,
        FittedLawProvenance,
        MixedLawProvenance,
        UnknownLawProvenance,
    )
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.artifacts.simulation import SimulationSpec
    from grassmoth.models.structure import StructuralSelection

# edit_question


class EditQuestionOutput(Value):
    """The question saved by this action."""

    question: QuestionSpec = Field(
        description="The saved question, including its outcome and intervention queries."
    )


# edit_model


class EditModelOutput(Value):
    """The produced model and its recorded scientific findings."""

    dynamical_model_spec: DynamicalModelSpec = Field(description="Saved scientific definition.")
    checks: ModelCheckReport = Field(
        description="Combined findings retained by the action that produced the model."
    )
    identification: IdentificationReport = Field(description="Causal identification findings.")
    pruning: ModelEditResult = Field(
        description="Entities removed because they do not serve the question."
    )


# prepare_data


@dataclass(frozen=True)
class SourceTable:
    """Captured source rows and the span of support windows they cover.

    The service guarantees a non-null date or datetime ``timestamp`` column when it
    captures the files. ``start`` is an inclusive and ``end`` an exclusive UTC date.
    """

    TIME_COLUMN: ClassVar[str] = "timestamp"

    rows: pa.Table
    start: date | None
    end: date | None


class ExtractedValue(Value):
    """A single extracted observation for an indicator within a support window."""

    window_start: str = Field(
        description="The support-window start time (e.g. '2024-01-15T00:00:00')"
    )
    indicator_id: IndicatorId = Field(description="Persistent identity of the indicator")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )


class PrepareDataOutput(Value):
    """Prepared observations, their resolved metadata, and data-quality findings."""

    data: ObservationData = Field(
        description=(
            "Full observation histories keyed by indicator ID, including measurement support "
            "intervals and missing values; times and empirical CDF coordinates are binary arrays."
        )
    )
    metadata: PreparedDataMetadata = Field(
        description=(
            "Preparation recipe, resolved observation schema, and the "
            "calendar origin used to interpret model time."
        )
    )
    profile: DataProfileReport = Field(
        description="Empirical statistics and data-quality findings for the prepared panel."
    )
    extraction: DataPreparationResult = Field(
        description="Retained extraction workers, failed chunks and reuse counts."
    )

    def observations(self) -> ObservationDataset:
        """Admit the stored histories under their preparation's recorded definitions."""
        from grassmoth.artifacts.observations.dataset import ObservationDataset

        return ObservationDataset.from_histories(
            self.data, self.metadata.variables, time_origin=self.metadata.time_origin
        )


# fit


@dataclass(frozen=True)
class FitInputs:
    """The scoped model, its question, the data and sampler, and any fit this one continues.

    ``compute_loo_diagnostics`` retains pointwise log-likelihoods for LOO.
    """

    selection: StructuralSelection
    question: QuestionSpec
    data_for_model: ObservationDataset
    sampler: SamplerSpec
    compute_loo_diagnostics: bool
    previous: FitOutput | None = None


class FitOutput(Value):
    """A conditioned model, its completed checks, and self-contained inference evidence."""

    dynamical_model_spec: DynamicalModelSpec | None = Field(
        description="The combined conditioned model, or null when this fit and its parents have no retained draws. Zero-draw continuations reuse their parent's model without producing a revision."
    )
    checks: FitCheckReport = Field(
        description="Completed compatibility and question findings for the selected history."
    )
    inference: InferenceReport = Field(
        description="Run provenance, native sampler evidence, diagnostics and parameter summaries."
    )
    profile: FitProfileReport | None = Field(
        default=None, description="Timings for this execution only when settings.profile is true."
    )

    @model_serializer(mode="wrap")
    def serialize_fit(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):  # noqa: ANN201 -- Retain the declared FitOutput schema through this serialization view.
        """Apply the requested transfer view without changing the cached value."""
        from grassmoth.artifacts.fit_transfer import transfer_fit

        return handler(
            transfer_fit(self) if info.context and info.context.get("thin_fit_draws") else self
        )


# simulate


@dataclass(frozen=True)
class FittedLaw:
    """Laws retained from a committed fit, with that fit's report."""

    provenance: FittedLawProvenance | MixedLawProvenance
    report: InferenceReportCore


type PredictiveLaw = AuthoredLawProvenance | UnknownLawProvenance | FittedLaw


@dataclass(frozen=True)
class SimulationInputs:
    """The scoped model, the design with its seed and replicate count, and its laws' ancestry.

    ``fitted`` pairs this revision with its own fit report when a fit produced it.
    """

    selection: StructuralSelection
    design: SimulationSpec
    seed: int
    num_replicates: int
    law: PredictiveLaw
    fitted: tuple[GitRef, InferenceReportCore] | None = None


class SimulateOutput(Value):
    """Generated histories and their complete scientific report, including owned numerical evidence."""

    data: tuple[ObservationData, ...] = Field(
        min_length=1,
        description=(
            "One observation history per retained replicate, in replicate order. "
            "Histories share one binary time buffer and each retains its binary empirical CDF."
        ),
    )
    report: SimulationReport = Field(description="Generation evidence and evaluated findings.")

    def observations(self, replicate: int) -> ObservationDataset:
        """Admit one replicate; the caller establishes 0 <= replicate < evidence.draws."""
        from grassmoth.artifacts.observations.dataset import ObservationDataset

        evidence = self.report.evidence
        return ObservationDataset.from_histories(
            self.data[replicate],
            evidence.observation_layout.variables,
            time_origin=evidence.time_origin,
        )


# data_diff


class DataDiffOutput(Value):
    """The computed comparison report; original histories remain in their producing results."""

    report: DataComparisonReport = Field(
        description=(
            "Exact history selections, point changes, per-history statistics, compatibility "
            "findings and predictive checks. Predictive overlays retain at most 20 binary "
            "spaghetti series; their median and frame use all replicates."
        )
    )


# model_diff


class ModelDiffOutput(Value):
    """Directional changes between saved specs, expressed in the model's editing language.

    DynamicalModelSpec.changes_from owns the document comparison. Applying its patch with
    merge_fields to the before spec reconstructs the after spec; equal specs yield
    an empty document. Only saved specification fields participate, including their
    exact law-buffer references. No execution findings or statistical comparisons
    are computed here; those remain owned by their producing actions.
    """

    model_config = ConfigDict(json_schema_mode_override="validation")

    changes: DynamicalModelSpec = Field(
        description=(
            "Merge this document into the before spec to obtain the after spec. Omitted fields "
            "are unchanged, supplied fields are added or updated, and null map entries delete "
            "their identities. A question revision denotes the empty spec."
        )
    )
