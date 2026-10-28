"""Reachability-based collection for workspace scratch state and caches."""

from __future__ import annotations

import argparse
import time

from pydantic import BaseModel, ConfigDict

from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module

_HOUR_SECONDS = 60 * 60
_DAY_SECONDS = 24 * _HOUR_SECONDS
_GIB = 1024**3


class RetentionPolicy(BaseModel):
    """How long scratch events and cached files live, and the cache's size cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_retention_seconds: int = 24 * _HOUR_SECONDS
    cache_retention_seconds: int = 30 * _DAY_SECONDS
    cache_max_bytes: int = 5 * _GIB


DEFAULT_RETENTION = RetentionPolicy()


class SweepResult(BaseModel):
    """Counts of scratch entries and cache bytes removed by a workspace sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    removed_runs: int = 0
    removed_events: int = 0
    removed_cache_files: int = 0
    removed_cache_bytes: int = 0


def collect_completed_runs(workspace_id: str) -> int:
    """Delete only scratch whose attempt is published; other calls may still be executing."""
    from grassmoth_server.study.persistence.history import StudyRepository

    completed = {
        f"seq-{revision.record.seq:06d}" for revision in StudyRepository(workspace_id).attempts()
    }
    entries = [
        entry
        for entry in storage.listdir(data_module.scratch_runs_dir(workspace_id))
        if entry.rsplit("/", 1)[-1] in completed
    ]
    for entry in entries:
        storage.rm_tree(entry)
    return len(entries)


def _collect_offline_runs(workspace_id: str) -> int:
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
    policy: RetentionPolicy = DEFAULT_RETENTION,
    now_seconds: float | None = None,
    collect_runs: bool = False,
) -> SweepResult:
    """Expire telemetry/caches and optionally collect runs while the study is offline."""
    now = time.time() if now_seconds is None else now_seconds
    if collect_runs:
        removed_runs = _collect_offline_runs(workspace_id)
    else:
        removed_runs = 0
    removed_events = _sweep_events(
        workspace_id,
        cutoff_seconds=now - policy.event_retention_seconds,
    )
    removed_cache_files, removed_cache_bytes = _sweep_cache(
        workspace_id,
        cutoff_seconds=now - policy.cache_retention_seconds,
        max_bytes=policy.cache_max_bytes,
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
    parser.add_argument(
        "--event-retention-hours",
        type=int,
        default=DEFAULT_RETENTION.event_retention_seconds // _HOUR_SECONDS,
    )
    parser.add_argument(
        "--cache-retention-days",
        type=int,
        default=DEFAULT_RETENTION.cache_retention_seconds // _DAY_SECONDS,
    )
    parser.add_argument(
        "--cache-max-gib", type=float, default=DEFAULT_RETENTION.cache_max_bytes / _GIB
    )
    parser.add_argument(
        "--collect-runs",
        action="store_true",
        help="collect run scratch; use only while the study is offline",
    )
    args = parser.parse_args()
    result = sweep_workspace(
        args.workspace_id,
        policy=RetentionPolicy(
            event_retention_seconds=args.event_retention_hours * _HOUR_SECONDS,
            cache_retention_seconds=args.cache_retention_days * _DAY_SECONDS,
            cache_max_bytes=int(args.cache_max_gib * _GIB),
        ),
        collect_runs=args.collect_runs,
    )
    print(result.model_dump_json())


if __name__ == "__main__":
    sweep_cli()
