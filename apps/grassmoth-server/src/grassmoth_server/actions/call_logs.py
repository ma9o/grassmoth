"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from grassmoth_server.actions.contracts.logs import ActionLog
from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.actions.progress import read_events
from grassmoth_server.storage.traces import collect_run_traces
from grassmoth_server.study.persistence.log_codec import assemble_call_log

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth_server.actions.contracts.call_state import CompletedCall, RunningCall
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.records import FailedOutcome

_PROGRESS = TypeAdapter(tuple[ProgressEvent, ...])


def collect_call_log(
    workspace_id: str,
    call: RunningCall | CompletedCall,
    *,
    at: datetime,
    failure: FailedOutcome | None = None,
) -> ActionLog:
    """Gather progress events and LLM traces into the log retained for one attempt.

    Args:
        workspace_id: Workspace containing the attempt's scratch events and traces.
        call: Sequenced call naming the trace directory, the attempt's progress events
            and the action messages already emitted by the workflow.
        at: Timestamp to use when assembling the final log entries.
        failure: Rejection or exception to include in the log, if the attempt failed.

    Returns:
        Combined action log ready to be published with the attempt.
    """
    return assemble_call_log(
        call.messages,
        {
            **collect_run_traces(workspace_id, call.seq),
            "progress.json": _PROGRESS.dump_json(tuple(read_events(workspace_id, call.attempt_id))),
        },
        at=at,
        failure=failure,
    )


def read_call_log(repository: StudyRepository, commit_id: str) -> ActionLog:
    """Read the action log retained at an exact study commit."""
    return ActionLog.model_validate_json(repository.read_file(commit_id, "logs/messages.json"))
