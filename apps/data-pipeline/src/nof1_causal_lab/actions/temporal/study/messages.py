"""Study workflow requests and activity results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts.effects import ActionReportName
from nof1_causal_lab.actions.contracts.requests import (
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
)
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observations.preparation import (
    FileSourceRef,
)
from nof1_causal_lab.study.records import ActionMessage, Applied, AttemptRecord, StagedActionAttempt
from nof1_causal_lab.study.state import StudyState


class StudyInit(BaseModel):
    """Workspace identity and starting sequence number for a study workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Attempt numbering resumes after the journal; each call names its scientific inputs.
    initial_seq: int = 0


class ReadInputsInput(BaseModel):
    """Pinned action request whose input state or prior saved result must be read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]


class ActionRequest(BaseModel):
    """Submitted call arguments paired with the attempt identity used for progress tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    attempt_id: UUID = Field(default_factory=uuid4)


class ActionInput(BaseModel):
    """Resolved scientific request and input state handed to an execution activity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: Annotated[
        FitRequest[GitOid]
        | SimulateRequest[GitOid]
        | DataDiffRequest[GitOid]
        | ModelDiffRequest[GitOid],
        Field(discriminator="action"),
    ]
    state: StudyState


class EditQuestionActivityInput(BaseModel):
    """Question edit and destination workspace passed across the activity boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditQuestionRequest


class EditModelActivityInput(BaseModel):
    """Pinned model edit and destination workspace passed across the activity boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditModelRequest[GitOid]


class EvaluateChecksInput[ResultT](Value):
    """Successful scientific result and its input state supplied to post-action checks."""

    workspace_id: str
    state: StudyState
    applied: Applied[ResultT]
    request: (
        EditModelRequest[GitOid] | FitRequest[GitOid] | PrepareDataRequest[GitOid, FileSourceRef]
    )


class ChecksResult(Value):
    """Checks stay in the artifact store; orchestration carries references and labels."""

    reports: Mapping[ActionReportName, GitOid]
    messages: tuple[ActionMessage, ...]


class CompleteResultInput(Value):
    """The completed execution whose full scientific result must be saved before publication."""

    workspace_id: str
    attempt: StagedActionAttempt


class AttemptPublication(Value):
    """The writer receives the same record; Git publication identity is computed outside it."""

    workspace_id: str
    parent_id: GitOid | None
    record: AttemptRecord
