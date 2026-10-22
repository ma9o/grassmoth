"""Live progress of a running attempt: extraction step events.

Events land as one JSON file each under ``data/{workspace_id}/scratch/events/`` (neither
local fs nor R2 supports atomic append); readers list the directory and sort by
filename, which is time-ordered by construction. Progress is disposable: the attempt
record and its traces are authoritative, and the sweep truncates the stream freely.
Every event names the attempt that emitted it, and each attempt reads only its own.
"""

from __future__ import annotations

import json
import time
import uuid

from pydantic import TypeAdapter

from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module

_PROGRESS_EVENT_ADAPTER = TypeAdapter(ProgressEvent)


def events_dir(workspace_id: str) -> str:
    """Locate the workspace's scratch directory for append-only progress event files."""
    return data_module.scratch_events_dir(workspace_id)


def emit_event(workspace_id: str, event: ProgressEvent) -> None:
    """Persist a progress event under a unique time-ordered filename without its read cursor."""
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps(event.model_dump(mode="json", exclude_none=True, exclude={"cursor"})),
    )


def read_events(
    workspace_id: str, attempt_id: uuid.UUID, *, after: str | None = None
) -> list[ProgressEvent]:
    """One attempt's retained events in emission order; ``after`` is the last seen cursor."""
    directory = events_dir(workspace_id)
    if not storage.exists(directory):
        return []
    events: list[ProgressEvent] = []
    for entry in sorted(e for e in storage.listdir(directory) if e.endswith(".json")):
        cursor = entry.rsplit("/", 1)[-1]
        if after is not None and cursor <= after:
            continue
        event = _PROGRESS_EVENT_ADAPTER.validate_python(
            {**storage.read_json(entry), "cursor": cursor}
        )
        if event.attempt_id == attempt_id:
            events.append(event)
    return events
