"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_service.actions.contracts.logs import (
    ActionLog,
    ExecutionMessage,
    FailureMessage,
    ProgressMessage,
    TraceLogMessage,
)
from nof1_causal_service.actions.contracts.progress import ProgressEvent
from nof1_causal_service.llm.traces import LLMTrace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_service.study.records import ActionMessage, FailedOutcome

_PROGRESS = TypeAdapter(tuple[ProgressEvent, ...])


def assemble_call_log(
    messages: tuple[ActionMessage, ...],
    logs: Mapping[str, bytes],
    *,
    at: datetime,
    failure: FailedOutcome | None,
) -> ActionLog:
    """All publication paths use this one owner for the complete execution log."""
    entries: list[ExecutionMessage] = list(messages)
    for path, raw in logs.items():
        if path == "progress.json":
            entries.extend(
                ProgressMessage(
                    timestamp=datetime.fromtimestamp(
                        int(event.cursor.split("-", 1)[0]) / 1_000_000_000, UTC
                    ),
                    progress=event,
                )
                for event in _PROGRESS.validate_json(raw)
            )
        elif path.startswith("traces/"):
            entries.append(
                TraceLogMessage(
                    timestamp=at,
                    trace_id=path.removeprefix("traces/").removesuffix(".json"),
                    trace=LLMTrace.model_validate_json(raw),
                )
            )
    if failure is not None:
        entries.append(FailureMessage(timestamp=at, failure=failure))
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    return ActionLog(messages=tuple(ordered))
