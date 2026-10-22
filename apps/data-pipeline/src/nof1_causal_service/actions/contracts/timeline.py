"""Transport-independent timeline records for the public study interface."""

from nof1_causal_lab.public import GitOid, Value
from nof1_causal_service.actions.contracts.identity import ActionId, CallId
from nof1_causal_service.actions.contracts.polling import RunningAction
from nof1_causal_service.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from nof1_causal_service.study.records import ActionMessage, Attempt, RecordDependency


class TimelineRecord(Value):
    """A replayable call log, excluding every scientific result and check payload."""

    seq: int
    ts: str
    messages: tuple[ActionMessage, ...]
    trace_ids: tuple[str, ...]
    attempt: Attempt[
        ActionId, ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid], None
    ]


class TimelineRevision(Value):
    """A lightweight attempt record linked to its call identity and Git publication."""

    call_id: CallId | None
    commit_id: GitOid
    parent_ids: tuple[GitOid, ...]
    record: TimelineRecord


class TimelineResponse(Value):
    """Journal attempts, declared input dependencies, and all currently running actions."""

    attempts: tuple[TimelineRevision, ...]
    dependencies: tuple[RecordDependency, ...]
    running: tuple[RunningAction, ...]
