"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth_server.actions.contracts.call_state import CompletedCall
from grassmoth_server.actions.contracts.logs import ActionLog, CallLogSnapshot
from grassmoth_server.actions.contracts.requests import call_identity
from grassmoth_server.actions.progress import read_events
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage.paths import call_dir
from grassmoth_server.storage.records import append_record, read_records
from grassmoth_server.storage.traces import collect_run_traces
from grassmoth_server.study.persistence.log_codec import assemble_call_log
from grassmoth_server.study.records import ActionMessage

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth_server.actions.contracts.call_state import RunningCall
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.records import FailedOutcome


def emit_message(workspace_id: str, call_id: str, message: ActionMessage) -> None:
    """Retain an execution message for live polling and final publication."""
    directory = storage.join(call_dir(workspace_id, call_id), "messages")
    append_record(directory, message.model_dump_json())


def collect_call_snapshot(workspace_id: str, call_id: str) -> CallLogSnapshot:
    """Read each call-owned source once for live polling or journal publication."""
    directory = storage.join(call_dir(workspace_id, call_id), "messages")
    return CallLogSnapshot(
        traces=collect_run_traces(workspace_id, call_id),
        events=tuple(read_events(workspace_id, call_id)),
        execution_messages=tuple(
            ActionMessage.model_validate_json(payload) for _, payload in read_records(directory)
        ),
    )


def collect_call_log(
    workspace_id: str,
    call: RunningCall | CompletedCall,
    *,
    at: datetime,
    failure: FailedOutcome | None = None,
) -> ActionLog:
    """Gather progress events and LLM traces into the log retained for one attempt.

    Args:
        workspace_id: Workspace containing the call's temporary events and traces.
        call: Sequenced call naming the trace directory, the attempt's progress events
            and the action messages already emitted by the workflow.
        at: Timestamp to use when assembling the final log entries.
        failure: Rejection or exception to include in the log, if the attempt failed.

    Returns:
        Combined action log ready to be published with the attempt.

    Raises:
        AssertionError: If a completed execution has no captured request.
    """
    request = call.attempt.request if isinstance(call, CompletedCall) else call.request
    assert request is not None
    identity = call_identity(request)
    snapshot = collect_call_snapshot(workspace_id, identity)
    return assemble_call_log(
        call.messages,
        snapshot,
        at=at,
        failure=failure,
    )


def read_call_log(repository: StudyRepository, commit_id: str) -> ActionLog:
    """Read the action log retained at an exact study commit."""
    return ActionLog.model_validate_json(repository.read_file(commit_id, "log.json"))
