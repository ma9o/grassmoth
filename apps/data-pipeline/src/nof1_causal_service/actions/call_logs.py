"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_service.actions.contracts.logs import ActionLog
from nof1_causal_service.actions.contracts.progress import ProgressEvent
from nof1_causal_service.actions.progress import read_events
from nof1_causal_service.storage.traces import collect_run_traces
from nof1_causal_service.study.persistence.log_codec import assemble_call_log

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from nof1_causal_service.study.persistence.history import StudyRepository
    from nof1_causal_service.study.records import ActionMessage, FailedOutcome

_PROGRESS = TypeAdapter(tuple[ProgressEvent, ...])


def collect_call_log(
    workspace_id: str,
    *,
    seq: int,
    attempt_id: UUID,
    messages: tuple[ActionMessage, ...],
    at: datetime,
    failure: FailedOutcome | None = None,
) -> ActionLog:
    """Gather progress events and LLM traces into the log retained for one attempt.

    Args:
        workspace_id: Workspace containing the attempt's scratch events and traces.
        seq: Study sequence number identifying the run's trace directory.
        attempt_id: Attempt whose progress events are included.
        messages: Action messages already emitted by the workflow.
        at: Timestamp to use when assembling the final log entries.
        failure: Rejection or exception to include in the log, if the attempt failed.

    Returns:
        Combined action log ready to be published with the attempt.
    """
    return assemble_call_log(
        messages,
        {
            **collect_run_traces(workspace_id, seq),
            "progress.json": _PROGRESS.dump_json(tuple(read_events(workspace_id, attempt_id))),
        },
        at=at,
        failure=failure,
    )


def read_call_log(repository: StudyRepository, commit_id: str) -> ActionLog:
    """Read the action log retained at an exact study commit."""
    return ActionLog.model_validate_json(repository.read_file(commit_id, "logs/messages.json"))
