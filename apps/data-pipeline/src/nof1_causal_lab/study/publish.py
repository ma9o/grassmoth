"""Copy missing durable workspace files to the hosted (R2) store.

See docs/guides/agentic_integration_testing.md#publishing-a-workspace for
supported use and the current exclusion and synchronization limits.

Usage (needs the ``cloud`` dependency group and the production R2 env:
``R2_ENDPOINT_URL``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``,
``R2_BUCKET``, ``R2_PREFIX``)::

    uv run nof1-publish SYNTHETIC_WORKSPACE [--exclude input]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem


def _dest_fs() -> AbstractFileSystem:
    import fsspec

    filesystem: AbstractFileSystem = fsspec.filesystem(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    return filesystem


def _is_publishable(rel: str) -> bool:
    """Only explicit durable tiers cross the publication boundary."""
    return rel.startswith(("input/", "store/", "study/"))


def _is_excluded(rel: str, excludes: list[str]) -> bool:
    for name in excludes:
        prefix = "input/" if name == "input" else f"store/{name}/"
        if rel.startswith(prefix):
            return True
    return False


def publish_workspace(workspace_id: str, excludes: list[str]) -> dict[str, int]:
    """Copy missing local workspace files to hosted storage, honoring the selected exclusions."""
    from nof1_causal_lab.storage import backend as storage
    from nof1_causal_lab.storage import paths as data_module

    if storage.is_remote():
        raise RuntimeError("publish copies FROM the local store; unset DEPLOYMENT_ENV=production")
    src_root = Path(data_module.workspace_dir(workspace_id))
    if not src_root.is_dir():
        raise FileNotFoundError(f"No local workspace at {src_root}")

    bucket = os.environ["R2_BUCKET"]
    prefix = os.environ.get("R2_PREFIX", "data")
    dest_root = f"{bucket}/{prefix}/{workspace_id}"
    fs = _dest_fs()
    existing = {found.lstrip("/") for found in fs.find(dest_root)}

    counts = {"uploaded": 0, "skipped": 0, "excluded": 0}
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root).as_posix()
        if not _is_publishable(rel) or _is_excluded(rel, excludes):
            counts["excluded"] += 1
            continue
        dest = f"{dest_root}/{rel}"
        if dest in existing:
            counts["skipped"] += 1
            continue
        fs.put_file(str(path), dest)
        counts["uploaded"] += 1
    return counts


def main() -> None:
    """Parse workspace publication options and report uploaded, existing, and excluded file counts."""
    parser = argparse.ArgumentParser(
        description="Publish a local workspace to the hosted (R2) store."
    )
    parser.add_argument("workspace_id")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Withhold input/ for 'input', otherwise only store/<name>/; repeatable. "
            "Does not filter content-addressed table blobs or Git history."
        ),
    )
    args = parser.parse_args()
    counts = publish_workspace(args.workspace_id, args.exclude)
    print(
        f"{args.workspace_id}: uploaded {counts['uploaded']}, "
        f"skipped {counts['skipped']} existing, excluded {counts['excluded']}"
    )


if __name__ == "__main__":
    main()
