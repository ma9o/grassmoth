"""Transport-independent timeline records for the public study interface."""

from grassmoth.public import GitOid, Value
from grassmoth_server.actions.contracts.identity import ActionId, CallId
from grassmoth_server.actions.contracts.polling import RunningAction
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from grassmoth_server.study.records import Attempt, RecordDependency


class TimelineRecord(Value):
    """Call arguments and publication metadata; execution messages belong to ActionPoll."""

    seq: int
    ts: str
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
