"""Atomic, append-only JSON files with time-ordered read cursors."""

import time
from collections.abc import Iterator
from uuid import uuid4

from grassmoth_server.storage import backend as storage


def append_record(directory: str, payload: str) -> None:
    """Publish one complete JSON record under a unique time-ordered cursor."""
    cursor = f"{time.time_ns():020d}-{uuid4().hex[:8]}.json"
    storage.write_text(storage.join(directory, cursor), payload)


def read_records(directory: str, *, after: str | None = None) -> Iterator[tuple[str, str]]:
    """Read complete records in cursor order, optionally after an exclusive cursor."""
    for entry in sorted(storage.listdir(directory)):
        if not entry.endswith(".json"):
            continue
        cursor = entry.rsplit("/", 1)[-1]
        if after is None or cursor > after:
            yield cursor, storage.read_text(entry)
