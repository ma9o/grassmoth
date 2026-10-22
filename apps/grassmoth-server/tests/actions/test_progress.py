from __future__ import annotations

from uuid import uuid4

import pytest

from grassmoth_server.actions.contracts.progress import (
    ExtractionPlanEvent,
    ExtractionSnapshotEvent,
    ExtractionWorkerEvent,
    StepError,
    StepEvent,
)
from grassmoth_server.actions.progress import emit_event, read_events

pytestmark = pytest.mark.contract


def test_progress_events_roundtrip_per_attempt(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    workspace_id = "progress-event-contracts"
    attempt, other = uuid4(), uuid4()

    emit_event(
        workspace_id,
        StepEvent(
            attempt_id=attempt,
            status="failed",
            error=StepError(type="ValidationError", message="unreadable file"),
        ),
    )
    emit_event(workspace_id, StepEvent(attempt_id=other, status="running"))
    emit_event(
        workspace_id,
        ExtractionPlanEvent(attempt_id=attempt, total_workers=2, max_concurrent_workers=1),
    )
    emit_event(
        workspace_id,
        ExtractionWorkerEvent(
            attempt_id=attempt, worker_id=0, state="completed", n_windows=3, n_extractions=4
        ),
    )
    emit_event(
        workspace_id,
        ExtractionSnapshotEvent(
            attempt_id=attempt,
            total_workers=2,
            pending_workers=1,
            running_workers=0,
            completed_workers=1,
            failed_workers=0,
        ),
    )

    events = read_events(workspace_id, attempt)
    assert [type(event) for event in events] == [
        StepEvent,
        ExtractionPlanEvent,
        ExtractionWorkerEvent,
        ExtractionSnapshotEvent,
    ]
    assert all(event.attempt_id == attempt and event.cursor for event in events)
    step = events[0]
    assert isinstance(step, StepEvent)
    assert step.error == StepError(type="ValidationError", message="unreadable file")
    assert read_events(workspace_id, attempt, after=events[1].cursor) == events[2:]
    (other_step,) = read_events(workspace_id, other)
    assert isinstance(other_step, StepEvent)
    assert other_step.status == "running"
