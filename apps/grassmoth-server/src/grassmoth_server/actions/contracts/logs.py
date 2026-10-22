"""One typed accumulator for every kind of action execution message."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from grassmoth.public import Value
from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.llm.traces import LLMTrace
from grassmoth_server.study.records import ActionMessage, FailedOutcome


class ProgressMessage(Value):
    """A structured progress event with its original cursor and measurements."""

    kind: Literal["progress"] = "progress"
    timestamp: AwareDatetime
    progress: ProgressEvent


class TraceLogMessage(Value):
    """The accumulated conversation and tool exchanges for one subroutine."""

    kind: Literal["trace"] = "trace"
    timestamp: AwareDatetime
    trace_id: str
    trace: LLMTrace


class FailureMessage(Value):
    """The complete terminal failure carried by the public execution log."""

    kind: Literal["failure"] = "failure"
    timestamp: AwareDatetime
    failure: Annotated[FailedOutcome, Field(discriminator="status")]  # noqa: FIELD003 -- External action clients read failure details exclusively from messages.


type ExecutionMessage = Annotated[
    ActionMessage | ProgressMessage | TraceLogMessage | FailureMessage,
    Field(discriminator="kind"),
]


class ActionLog(Value):
    """The durable accumulator retained with an action's Git record."""

    messages: tuple[ExecutionMessage, ...]
