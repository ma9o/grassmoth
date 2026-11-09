"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from grassmoth_server.actions.contracts.logs import (
    ActionLog,
    CallLogSnapshot,
    ExecutionMessage,
    FailureMessage,
    ProgressMessage,
    TraceLogMessage,
)
from grassmoth_server.actions.messages import completion_message, extraction_failure_messages
from grassmoth_server.study.records import ActionMessage, Applied

if TYPE_CHECKING:
    from grassmoth_server.study.records import AttemptRecord, FailedOutcome


def assemble_call_log(
    messages: tuple[ActionMessage, ...],
    snapshot: CallLogSnapshot,
    *,
    at: datetime,
    failure: FailedOutcome | None,
) -> ActionLog:
    """All publication paths use this one owner for the complete execution log."""
    entries: list[ExecutionMessage] = [*messages, *snapshot.execution_messages]
    entries.extend(
        ProgressMessage(
            timestamp=datetime.fromtimestamp(
                int(event.cursor.split("-", 1)[0]) / 1_000_000_000, UTC
            ),
            progress=event,
        )
        for event in snapshot.events
    )
    entries.extend(
        TraceLogMessage(timestamp=at, trace_id=trace_id, trace=trace)
        for trace_id, trace in snapshot.traces.items()
    )
    if failure is not None:
        entries.append(FailureMessage(timestamp=at, failure=failure))
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    return ActionLog(messages=tuple(ordered))


def finalize_call_log(
    record: AttemptRecord, messages: tuple[ActionMessage, ...], snapshot: CallLogSnapshot
) -> ActionLog:
    """Add publication findings and the final outcome after publication policy has run."""
    at = datetime.fromisoformat(record.ts)
    failures = extraction_failure_messages(snapshot.events, at) if record.attempt_id else ()
    return assemble_call_log(
        (*messages, *failures, completion_message(record.attempt, at)),
        snapshot,
        at=at,
        failure=None if isinstance(record.attempt.outcome, Applied) else record.attempt.outcome,
    )
