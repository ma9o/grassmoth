"""HTTP policy boundary."""

from __future__ import annotations

import os
import re

from fastapi import (
    HTTPException,
)


def actions_enabled() -> bool:
    """Check whether the process permits action execution rather than serving saved calls only."""
    return os.environ.get("READ_ONLY_FACADE") != "1"


def _require_actions_enabled() -> None:
    if not actions_enabled():
        raise HTTPException(403, "This read-only facade answers saved calls only")


def _safe_workspace_id(value: str) -> str:
    workspace_id = value.strip()
    if (
        not workspace_id
        or len(workspace_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id) is None
    ):
        raise HTTPException(400, "Invalid workspaceId format")
    return workspace_id
