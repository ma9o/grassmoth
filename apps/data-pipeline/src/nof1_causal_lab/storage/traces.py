"""Collect scratch LLM traces for publication in study history."""

import re

from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module

_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def trace_log_path(subroutine_id: str) -> str:
    """A trace's path within the owning commit tree."""
    if _TRACE_ID.fullmatch(subroutine_id) is None:
        raise ValueError(f"Invalid attempt trace subroutine id: {subroutine_id!r}")
    return f"traces/{subroutine_id}.json"


def collect_run_traces(workspace_id: str, seq: int) -> dict[str, bytes]:
    """Collect the accumulated trace snapshots before scratch is swept."""
    llm_root = storage.join(data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"), "llm")
    logs: dict[str, bytes] = {}
    for subroutine_root in sorted(storage.listdir(llm_root)):
        subroutine_id = subroutine_root.rstrip("/").rsplit("/", 1)[-1]
        source = storage.join(subroutine_root, "trace.json")
        if storage.exists(source):
            logs[trace_log_path(subroutine_id)] = storage.read_text(source).encode()
    return logs
