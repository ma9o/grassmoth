"""Seven service action inputs, selecting exact revisions or HTTP latest selectors.

The service resolves these references and settings into the values consumed by
library actions. Saved requests always contain pinned revisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.preparation import (
    ExtractionSpec,
)
from nof1_causal_lab.artifacts.observations.reference import DataRef, DataSelection
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec


class EditQuestionInput(Value):
    """The study question, its required outcome, and any named intervention queries."""

    question: QuestionSpec = Field(
        description=(
            "The replacement study question, including the required outcome and any "
            "intervention queries to assess against candidate models."
        )
    )


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


class SimulateInput[RevisionT](Value):
    """Generate histories from a selected model and an explicit simulation design."""

    dynamical_model_spec_ref: RevisionT = Field(
        description="Revision supplying the authored or fitted generative law."
    )
    simulation: SimulationSpec = Field(description="Requested start, horizon, and interventions.")
    seed: int = Field(default=0, ge=0, description="Generation seed, retained with the histories.")
    num_replicates: int = Field(default=100, ge=1, description="Number of independent histories.")


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
