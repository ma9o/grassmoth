"""The seven public actions, each with its input and successful output body.

HTTP inputs use ``RevisionSelector`` (an exact revision or ``"latest"``);
saved inputs use resolved ``GitOid`` values. ``prepare_data.source`` is a
``SourceFolder`` over HTTP and a captured ``FileSourceRef`` in saved inputs.

Request envelopes live in ``actions.contracts.requests``. Each output becomes the
``body`` of the polling envelope in ``actions.contracts.polling``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec, ModelEditResult
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.history import ObservationData
from nof1_causal_lab.artifacts.observations.preparation import (
    DataPreparationResult,
    ExtractionSpec,
    PreparedDataMetadata,
)
from nof1_causal_lab.artifacts.observations.reference import DataRef, DataSelection
from nof1_causal_lab.artifacts.reports.data_comparison import DataComparisonReport
from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
from nof1_causal_lab.artifacts.reports.inference import FitCheckReport, InferenceReport
from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec

if TYPE_CHECKING:
    import numpy as np

    from nof1_causal_lab.artifacts.probability.joint_layout import JointLawLayout
    from nof1_causal_lab.artifacts.reports.inference import FitSegment

# edit_question


class EditQuestionInput(Value):
    """The study question, its required outcome, and any named intervention queries."""

    question: QuestionSpec = Field(
        description=(
            "The replacement study question, including the required outcome and any "
            "intervention queries to assess against candidate models."
        )
    )


class EditQuestionOutput(Value):
    """The question saved by this action."""

    question: QuestionSpec = Field(
        description="The saved question, including its outcome and intervention queries."
    )


# edit_model


class EditModelInput[RevisionT](Value):
    """Create or revise a model from a question or model parent."""

    model_config = ConfigDict(json_schema_mode_override="validation")

    parent_ref: RevisionT = Field(
        description=(
            "Question or model revision to start from. A model parent supplies its pinned "
            "question. 'latest' selects the current model, otherwise the current question. "
            "An exact Git hash can select either parent without being the current head."
        )
    )
    dynamical_model_spec: DynamicalModelSpec = Field(
        description=(
            "Scientific definitions merged by identity into a model parent, or into an empty "
            "model for a question parent. Omitted fields are retained; null entity entries "
            "delete their identities. The action retains the merged spec and its findings."
        )
    )


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


class PrepareDataInput[RevisionT, SourceT](Value):
    """Extract observations using the definitions of the selected model."""

    dynamical_model_spec_ref: RevisionT = Field(
        description="Model revision owning the clock and observation definitions."
    )
    source: SourceT = Field(
        description=(
            "Folder under `data/{workspace_id}/` at the HTTP boundary; the captured file "
            "reference in the saved request."
        )
    )
    extraction: Mapping[IndicatorId, ExtractionSpec] = Field(
        min_length=1,
        description=(
            "Computed, recorded-reading or semantic extraction instructions keyed by the model's "
            "observation IDs."
        ),
    )
    context: str = Field(
        default="", description="Additional background for interpreting the uploaded source tables."
    )


class PrepareDataOutput(Value):
    """Prepared observations, their resolved metadata, and data-quality findings."""

    data: ObservationData = Field(
        description=(
            "Full observation histories keyed by indicator ID, including measurement support "
            "intervals and missing values."
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


# fit


class NewFitTarget[RevisionT](Value):
    """Condition the selected model law on exactly one saved history."""

    kind: Literal["new"] = "new"
    dynamical_model_spec_ref: RevisionT
    data_ref: DataRef[RevisionT, int]


class ContinueFitTarget[RevisionT](Value):
    """Continue the original target at the last retained draw of a prior fit."""

    kind: Literal["continue"] = "continue"
    fit_ref: RevisionT


type FitTarget[RevisionT] = Annotated[
    NewFitTarget[RevisionT] | ContinueFitTarget[RevisionT], Field(discriminator="kind")
]


class FitInput[RevisionT](Value):
    """Start or continue a fit; sample counts always mean additional retained draws."""

    target: FitTarget[RevisionT]
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)
    repeat: int = Field(
        default=0,
        ge=0,
        description="Timing repeat identity; identical repeats reuse their saved result.",
    )


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

    def posterior_draws(
        self, segment: FitSegment | None = None
    ) -> tuple[JointLawLayout, np.ndarray] | None:
        """Read aligned scientific coordinates, optionally selecting one interval per chain."""
        from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms

        metadata = self.inference.core.inference_metadata
        model = self.dynamical_model_spec
        if model is None or metadata.distribution is None:
            return None
        layout = model.law_layouts[metadata.distribution]
        atoms = empirical_atoms(model.distributions[metadata.distribution])
        if segment is not None:
            atoms = atoms.reshape(metadata.num_chains, -1, atoms.shape[-1])[
                :, segment.start : segment.stop
            ].reshape(-1, atoms.shape[-1])
        return layout, atoms


# simulate


class SimulateInput[RevisionT](Value):
    """Generate histories from a selected model and an explicit simulation design."""

    dynamical_model_spec_ref: RevisionT = Field(
        description="Revision supplying the authored or fitted generative law."
    )
    simulation: SimulationSpec = Field(description="Requested start, horizon, and interventions.")
    seed: int = Field(default=0, ge=0, description="Generation seed, retained with the histories.")
    num_replicates: int = Field(default=100, ge=1, description="Number of independent histories.")


class SimulateOutput(Value):
    """Generated histories and their complete scientific report, including owned numerical evidence."""

    data: tuple[ObservationData, ...] = Field(
        min_length=1,
        description="One observation history per retained replicate, in replicate order.",
    )
    report: SimulationReport = Field(description="Generation evidence and evaluated findings.")


# data_diff


class DataDiffInput[RevisionT](Value):
    """Compare saved histories statistically; left/right do not imply before/after revisions.

    DataComparisonReport owns selection and comparison semantics. The action reads
    retained observations without generating new draws or changing scientific state.
    """

    left_ref: DataSelection[RevisionT] = Field(
        description=(
            "References on the left side. Each reference selects one replicate by index, or all"
            " its retained histories when the index is omitted."
        )
    )
    right_ref: DataSelection[RevisionT] = Field(
        description="References on the right side, using the same selection rule."
    )


class DataDiffOutput(Value):
    """The computed comparison report; original histories remain in their producing results."""

    report: DataComparisonReport = Field(
        description=(
            "Exact history selections, point changes, per-history statistics, compatibility "
            "findings and predictive checks."
        )
    )


# model_diff


class ModelDiffInput[RevisionT](Value):
    """Base and target specs for a directional patch, selected by question or model revisions.

    The selections need not be chronologically ordered or share an editing parent.
    Before/after determine patch direction; ModelDiffOutput defines its contract.
    """

    before_ref: RevisionT = Field(
        description="Question or model revision used as the base; a question denotes an empty spec."
    )
    after_ref: RevisionT = Field(
        description="Question or model revision reconstructed by applying the patch to the base."
    )


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
