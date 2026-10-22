"""Small workflow-owned call states; scientific response bodies stay at the HTTP edge."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from grassmoth.public import GitOid, Value
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from grassmoth_server.study.records import ActionAttempt, ActionMessage


class PendingCall(Value):
    """An accepted call waiting for the study workflow to assign a sequence number."""

    kind: Literal["pending"] = "pending"
    attempt_id: UUID
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]


class RunningCall(Value):
    """A sequenced call in progress, with the messages emitted so far."""

    kind: Literal["running"] = "running"
    attempt_id: UUID
    seq: int
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    messages: tuple[ActionMessage, ...] = ()


class CompletedCall(Value):
    """A finished attempt and its publication commit, if publication succeeded."""

    kind: Literal["completed"] = "completed"
    attempt_id: UUID
    seq: int
    commit_id: GitOid | None
    attempt: ActionAttempt
    messages: tuple[ActionMessage, ...] = ()


type CallProgress = Annotated[
    PendingCall | RunningCall | CompletedCall, Field(discriminator="kind")
]
