"""The shared polling envelope and action-specific scientific result bodies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, computed_field

from nof1_causal_lab.public import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    GitOid,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
    Value,
)
from nof1_causal_service.actions.contracts.identity import ActionId, CallId
from nof1_causal_service.actions.contracts.logs import ExecutionMessage
from nof1_causal_service.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from nof1_causal_service.study.persistence.codec import pack_result, unpack_result


class RunningPoll(Value):
    """An accepted call that can be polled by an external action client."""

    call_id: CallId  # noqa: FIELD003 -- External action clients use this ID for GET polling.
    action: ActionId  # noqa: FIELD003 -- External action clients use this name for GET polling.
    status: Literal["running"] = "running"
    messages: tuple[ExecutionMessage, ...] = ()

    @computed_field
    @property
    def commit_id(self) -> None:  # noqa: FIELD003 -- The shared action wire envelope requires null before publication.
        """A running call has no published revision."""
        return None

    @computed_field
    @property
    def body(self) -> None:  # noqa: FIELD003 -- The shared action wire envelope requires null while running.
        """The shared envelope has no scientific result while running."""
        return None


class FailedPoll(Value):
    """A terminal failure with its full details in the accumulated messages."""

    call_id: CallId  # noqa: FIELD003 -- External action clients correlate cached failures by call ID.
    action: ActionId  # noqa: FIELD003 -- The shared action wire envelope identifies the failed action.
    status: Literal["failed"] = "failed"
    commit_id: GitOid | None  # noqa: FIELD003 -- External action clients retain the recorded failure's Git reference.
    messages: tuple[ExecutionMessage, ...]

    @computed_field
    @property
    def body(self) -> None:  # noqa: FIELD003 -- The shared action wire envelope requires null on failure.
        """The shared envelope has no scientific result after failure."""
        return None


class SuccessfulPoll[ActionT: str, BodyT](Value):
    """A published scientific result and the complete execution log of its call."""

    call_id: CallId  # noqa: FIELD003 -- External action clients correlate cached results by call ID.
    action: ActionT
    status: Literal["success"] = "success"  # noqa: FIELD002 -- Discriminates ActionPoll through the nested generic ActionSuccess union.
    commit_id: GitOid  # noqa: FIELD003 -- External action clients use this revision as a subsequent action input.
    body: BodyT
    messages: tuple[ExecutionMessage, ...]


type ActionSuccess = Annotated[
    SuccessfulPoll[Literal["edit_question"], EditQuestionOutput]
    | SuccessfulPoll[Literal["edit_model"], EditModelOutput]
    | SuccessfulPoll[Literal["prepare_data"], PrepareDataOutput]
    | SuccessfulPoll[Literal["fit"], FitOutput]
    | SuccessfulPoll[Literal["simulate"], SimulateOutput]
    | SuccessfulPoll[Literal["data_diff"], DataDiffOutput]
    | SuccessfulPoll[Literal["model_diff"], ModelDiffOutput],
    Field(discriminator="action"),
]

type ActionPoll = Annotated[RunningPoll | FailedPoll | ActionSuccess, Field(discriminator="status")]


class RunningAction(Value):
    """Timeline discovery of the active call and its accumulated messages."""

    call_id: CallId  # noqa: FIELD003 -- External clients discover the active call here before GET polling.
    action: ActionId
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    messages: tuple[ExecutionMessage, ...]


class StudyError(Value):
    """An expected rejection at the public study boundary."""

    code: Literal["invalid_request", "read_only", "invalid_inputs", "unknown_call"]
    detail: str


_POLL: TypeAdapter[ActionPoll] = TypeAdapter(ActionPoll)


class ActionResult(Value):
    """An intact action envelope; decoding is optional for callers forwarding saved bytes."""

    payload: bytes

    @classmethod
    def from_poll(cls, poll: ActionPoll) -> ActionResult:
        """Encode a live status using the same numerical codec as saved results."""
        return cls(payload=pack_result(poll))

    def decode(self) -> ActionPoll:
        """Read the typed result without executing or rebuilding scientific outputs."""
        return _POLL.validate_python(unpack_result(self.payload))
