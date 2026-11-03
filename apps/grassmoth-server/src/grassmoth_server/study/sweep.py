"""Collect call-owned temporary files only after durable publication or offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from grassmoth_server.actions.contracts.requests import call_identity
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths


def collect_completed_runs(workspace_id: str) -> int:
    """Remove saved calls' temporary state, including leftovers after a publication crash."""
    from grassmoth_server.study.persistence.history import StudyRepository

    removed = 0
    for revision in StudyRepository(workspace_id).attempts():
        request = revision.record.attempt.request
        assert request is not None
        directory = paths.call_dir(workspace_id, call_identity(request))
        if storage.exists(directory):
            storage.rm_tree(directory)
            removed += 1
    return removed


def cleanup_saved_calls() -> None:
    """Recover completed-call leftovers at startup without touching unfinished calls or uploads."""
    root = Path(paths.data_root())
    root.mkdir(parents=True, exist_ok=True)
    for directory in root.iterdir():
        if directory.is_dir() and Path(paths.history_dir(directory.name)).is_dir():
            collect_completed_runs(directory.name)


def sweep_cli() -> None:
    """Remove saved-call leftovers, or explicitly clear all temporary state while offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_id")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Clear all tmp/ files; requires a stopped study, including uploads and orphan captures.",
    )
    args = parser.parse_args()
    if args.offline:
        storage.rm_tree(paths.tmp_dir(args.workspace_id))
    else:
        print(f"Removed {collect_completed_runs(args.workspace_id)} saved call directories")


if __name__ == "__main__":
    sweep_cli()
