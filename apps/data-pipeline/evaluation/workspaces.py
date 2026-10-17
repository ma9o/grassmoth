"""Copy completed synthetic study ledgers without copying live execution state."""

from __future__ import annotations

import hashlib
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.client import ActionClient

_DURABLE = ("input", "store", "study")


def ledger_hashes(root: Path) -> dict[str, str]:
    """Fingerprint every durable file, including the Git refs and owned numerical blobs."""
    hashes = {}
    for tier in _DURABLE:
        for path in sorted((root / tier).rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    hashes[path.relative_to(root).as_posix()] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    return hashes


def copy_anchor_workspace(root: Path, source: ActionClient, destination: str) -> dict[str, str]:
    """Copy a stopped ledger to a new workspace and verify the complete durable snapshot."""
    source.require_idle()
    origin = root / source.workspace
    target = root / destination
    if origin.resolve().parent != root.resolve() or target.resolve().parent != root.resolve():
        raise ValueError("Benchmark workspace names must be direct children of their data root")
    if not (origin / "study" / "history.git").is_dir():
        raise ValueError("Anchor workspace has no durable study repository")
    before = ledger_hashes(origin)
    target.mkdir(parents=False, exist_ok=False)
    for tier in _DURABLE:
        if (origin / tier).exists():
            shutil.copytree(origin / tier, target / tier)
    if ledger_hashes(target) != before or ledger_hashes(origin) != before:
        raise ValueError("Anchor ledger changed while copying")
    return before
