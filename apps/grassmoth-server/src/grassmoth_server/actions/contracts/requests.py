"""Request envelopes and call identity for the bodies in ``actions.contracts.bodies``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from grassmoth.public import GitOid, Value
from grassmoth_server.actions.contracts.identity import CallId, RevisionSelector
from grassmoth_server.actions.contracts.inputs import (
    DataDiffInput,
    EditModelInput,
    EditQuestionInput,
    FitInput,
    ModelDiffInput,
    PrepareDataInput,
    SimulateInput,
)
from grassmoth_server.actions.contracts.preparation import FileSourceRef, SourceFolder
from grassmoth_server.actions.contracts.reasoning import ACTION_REASONING_DESCRIPTION


class EditQuestionRequest(Value):
    """Save the supplied question; the successful body is ``EditQuestionOutput``."""

    action: Literal["edit_question"] = Field(
        default="edit_question", description="Scientific action that owns this request or result."
    )
    input: EditQuestionInput = Field(description="Typed arguments of the scientific action.")
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class EditModelRequest[RevisionT](Value):
    """Create or revise a model from its selected parent; return ``EditModelOutput``."""

    action: Literal["edit_model"] = Field(
        default="edit_model", description="Scientific action that owns this request or result."
    )
    input: EditModelInput[RevisionT] = Field(
        description="Typed arguments of the scientific action."
    )
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class PrepareDataRequest[RevisionT, SourceT](Value):
    """Prepare the selected source data; return ``PrepareDataOutput``."""

    action: Literal["prepare_data"] = Field(
        default="prepare_data", description="Scientific action that owns this request or result."
    )
    input: PrepareDataInput[RevisionT, SourceT] = Field(
        description="Typed arguments of the scientific action."
    )
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class FitRequest[RevisionT](Value):
    """Fit the selected model to the selected history; return ``FitOutput``."""

    action: Literal["fit"] = Field(
        default="fit", description="Scientific action that owns this request or result."
    )
    input: FitInput[RevisionT] = Field(description="Typed arguments of the scientific action.")
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)

    @property
    def call_id(self) -> CallId:
        """Name a fit whose input revisions are already pinned, before submitting it."""
        return call_identity(self)


class SimulateRequest[RevisionT](Value):
    """Generate observation histories and their complete report; return ``SimulateOutput``."""

    action: Literal["simulate"] = Field(
        default="simulate", description="Scientific action that owns this request or result."
    )
    input: SimulateInput[RevisionT] = Field(description="Typed arguments of the scientific action.")
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class DataDiffRequest[RevisionT](Value):
    """Compare the saved observation histories; return ``DataDiffOutput``."""

    action: Literal["data_diff"] = Field(
        default="data_diff", description="Scientific action that owns this request or result."
    )
    input: DataDiffInput[RevisionT] = Field(description="Typed arguments of the scientific action.")
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class ModelDiffRequest[RevisionT](Value):
    """Compare two saved specs as an edit document; return ``ModelDiffOutput``."""

    action: Literal["model_diff"] = Field(
        default="model_diff", description="Scientific action that owns this request or result."
    )
    input: ModelDiffInput[RevisionT] = Field(
        description="Typed arguments of the scientific action."
    )
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


type ScientificActionRequest = Annotated[
    EditQuestionRequest
    | EditModelRequest[GitOid]
    | PrepareDataRequest[GitOid, FileSourceRef]
    | FitRequest[GitOid]
    | SimulateRequest[GitOid],
    Field(discriminator="action"),
]

type ActionInput = Annotated[
    EditQuestionRequest
    | EditModelRequest[RevisionSelector]
    | PrepareDataRequest[RevisionSelector, SourceFolder]
    | FitRequest[RevisionSelector]
    | SimulateRequest[RevisionSelector]
    | DataDiffRequest[RevisionSelector]
    | ModelDiffRequest[RevisionSelector],
    Field(discriminator="action"),
]


def call_identity(request: Value) -> CallId:
    """Canonical scientific arguments name a call independently of authored reasoning."""
    return CallId.from_content(
        request.model_dump(mode="json", round_trip=True, exclude={"reasoning"})
    )
