"""Reachability-based collection for workspace scratch state and caches."""

from __future__ import annotations

import argparse
import time

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module

DEFAULT_EVENT_RETENTION_SECONDS = 24 * 60 * 60
DEFAULT_CACHE_RETENTION_SECONDS = 30 * 24 * 60 * 60
DEFAULT_CACHE_MAX_BYTES = 5 * 1024**3


class SweepResult(BaseModel):
    """Counts of scratch entries and cache bytes removed by a workspace sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    removed_runs: int = 0
    removed_events: int = 0
    removed_cache_files: int = 0
    removed_cache_bytes: int = 0


def collect_completed_runs(workspace_id: str) -> int:
    """Delete completed run scratch while holding the action lock or working offline."""
    entries = storage.listdir(data_module.scratch_runs_dir(workspace_id))
    for entry in entries:
        storage.rm_tree(entry)
    return len(entries)


def _sweep_events(workspace_id: str, *, cutoff_seconds: float) -> int:
    removed = 0
    for entry in storage.listdir(data_module.scratch_events_dir(workspace_id)):
        cursor = entry.rsplit("/", 1)[-1]
        emitted_ns = int(cursor.split("-", 1)[0])
        if emitted_ns / 1_000_000_000 >= cutoff_seconds:
            continue
        storage.rm_file(entry)
        removed += 1
    return removed


def _sweep_cache(
    workspace_id: str,
    *,
    cutoff_seconds: float,
    max_bytes: int,
) -> tuple[int, int]:
    entries = [
        (path, info.modified_seconds, info.size)
        for path in storage.walk_files(data_module.cache_dir(workspace_id))
        for info in (storage.file_info(path),)
    ]
    removed_files = 0
    removed_bytes = 0
    retained: list[tuple[str, float, int]] = []
    for path, modified, size in entries:
        if modified < cutoff_seconds:
            storage.rm_file(path)
            removed_files += 1
            removed_bytes += size
        else:
            retained.append((path, modified, size))

    total = sum(size for _, _, size in retained)
    for path, _, size in sorted(retained, key=lambda item: (item[1], item[0])):
        if total <= max_bytes:
            break
        storage.rm_file(path)
        total -= size
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


def sweep_workspace(
    workspace_id: str,
    *,
    now_seconds: float | None = None,
    event_retention_seconds: int = DEFAULT_EVENT_RETENTION_SECONDS,
    cache_retention_seconds: int = DEFAULT_CACHE_RETENTION_SECONDS,
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
    collect_runs: bool = False,
) -> SweepResult:
    """Expire telemetry/caches and optionally collect runs while the study is offline."""
    now = time.time() if now_seconds is None else now_seconds
    if collect_runs:
        removed_runs = collect_completed_runs(workspace_id)
    else:
        removed_runs = 0
    removed_events = _sweep_events(
        workspace_id,
        cutoff_seconds=now - event_retention_seconds,
    )
    removed_cache_files, removed_cache_bytes = _sweep_cache(
        workspace_id,
        cutoff_seconds=now - cache_retention_seconds,
        max_bytes=cache_max_bytes,
    )
    return SweepResult(
        removed_runs=removed_runs,
        removed_events=removed_events,
        removed_cache_files=removed_cache_files,
        removed_cache_bytes=removed_cache_bytes,
    )


def sweep_cli() -> None:
    """Parse workspace retention limits, perform the requested sweep, and print its removal counts."""
    parser = argparse.ArgumentParser(description="Collect workspace scratch state and caches.")
    parser.add_argument("workspace_id")
    parser.add_argument("--event-retention-hours", type=int, default=24)
    parser.add_argument("--cache-retention-days", type=int, default=30)
    parser.add_argument("--cache-max-gib", type=float, default=5.0)
    parser.add_argument(
        "--collect-runs",
        action="store_true",
        help="collect run scratch; use only while the study is offline",
    )
    args = parser.parse_args()
    result = sweep_workspace(
        args.workspace_id,
        event_retention_seconds=args.event_retention_hours * 60 * 60,
        cache_retention_seconds=args.cache_retention_days * 24 * 60 * 60,
        cache_max_bytes=int(args.cache_max_gib * 1024**3),
        collect_runs=args.collect_runs,
    )
    print(result.model_dump_json())


if __name__ == "__main__":
    sweep_cli()
