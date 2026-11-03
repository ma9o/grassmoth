"""Collect call-owned temporary files only after durable publication or offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths


def collect_completed_runs(workspace_id: str) -> int:
    """Collect temporary call directories using ref existence, without parsing saved calls."""
    from grassmoth_server.study.persistence.git_objects import open_repository

    temporary = Path(paths.tmp_dir(workspace_id))
    if not temporary.exists():
        return 0
    repository = open_repository(workspace_id)
    removed = 0
    for directory in temporary.iterdir():
        if (
            directory.is_dir()
            and directory.name.startswith("call:")
            and f"refs/calls/{directory.name.removeprefix('call:')}" in repository.references
        ):
            storage.rm_tree(str(directory))
            removed += 1
    return removed


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
