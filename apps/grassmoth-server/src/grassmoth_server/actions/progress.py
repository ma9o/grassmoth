"""Call-owned progress files retained until their final log is committed."""

from __future__ import annotations

import json
import time
import uuid

from pydantic import TypeAdapter

from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module

_PROGRESS_EVENT_ADAPTER = TypeAdapter(ProgressEvent)


def events_dir(workspace_id: str, call_id: str) -> str:
    """Locate the workspace's scratch directory for append-only progress event files."""
    return storage.join(data_module.call_dir(workspace_id, call_id), "events")


def emit_event(workspace_id: str, call_id: str, event: ProgressEvent) -> None:
    """Persist a progress event under a unique time-ordered filename without its read cursor."""
    directory = events_dir(workspace_id, call_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps(event.model_dump(mode="json", exclude_none=True, exclude={"cursor"})),
    )


def read_events(
    workspace_id: str, call_id: str, *, after: str | None = None
) -> list[ProgressEvent]:
    """One attempt's retained events in emission order; ``after`` is the last seen cursor."""
    directory = events_dir(workspace_id, call_id)
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
        events.append(event)
    return events
