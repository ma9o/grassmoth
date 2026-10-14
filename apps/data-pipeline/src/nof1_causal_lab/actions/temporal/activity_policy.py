"""Replay-safe support shared by the study's workflows."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, ChildWorkflowError

from nof1_causal_lab.actions.temporal.extraction.messages import ProgressEventInput
from nof1_causal_lab.study.records import Raised

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.progress import ProgressEvent

EVENT_TIMEOUT = timedelta(seconds=30)
EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)


async def emit_progress(workspace_id: str, event: ProgressEvent) -> None:
    """Publish one live progress event using the common activity policy."""
    await workflow.execute_activity(
        "emit_progress_event_activity",
        ProgressEventInput(workspace_id=workspace_id, event=event),
        start_to_close_timeout=EVENT_TIMEOUT,
        retry_policy=EVENT_RETRY,
    )


def temporal_failure(exc: BaseException) -> Raised:
    """Only execution-failure handlers consume this foreign transport payload."""
    cause = exc
    while not isinstance(cause, ApplicationError):
        next_cause = (
            cause.cause
            if isinstance(cause, (ActivityError, ChildWorkflowError))
            else cause.__cause__
        )
        if next_cause is None:
            break
        cause = next_cause
    if isinstance(cause, ApplicationError):
        return Raised(
            error_type=cause.type or "ApplicationError",
            error_message=cause.message,
            details=tuple(json.dumps(value, sort_keys=True) for value in cause.details),
        )
    return Raised(error_type=type(cause).__name__, error_message=str(cause))
