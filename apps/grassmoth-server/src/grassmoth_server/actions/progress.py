"""Call-owned progress files retained until their final log is committed."""

from __future__ import annotations

from pydantic import TypeAdapter

from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.storage.records import append_record, read_records

_PROGRESS_EVENT_ADAPTER = TypeAdapter(ProgressEvent)


def events_dir(workspace_id: str, call_id: str) -> str:
    """Locate the workspace's scratch directory for append-only progress event files."""
    return storage.join(data_module.call_dir(workspace_id, call_id), "events")


def emit_event(workspace_id: str, call_id: str, event: ProgressEvent) -> None:
    """Persist a progress event under a unique time-ordered filename without its read cursor."""
    append_record(
        events_dir(workspace_id, call_id),
        event.model_dump_json(exclude_none=True, exclude={"cursor"}),
    )


def read_events(
    workspace_id: str, call_id: str, *, after: str | None = None
) -> list[ProgressEvent]:
    """One attempt's retained events in emission order; ``after`` is the last seen cursor."""
    return [
        _PROGRESS_EVENT_ADAPTER.validate_json(payload).revised(cursor=cursor)
        for cursor, payload in read_records(events_dir(workspace_id, call_id), after=after)
    ]
