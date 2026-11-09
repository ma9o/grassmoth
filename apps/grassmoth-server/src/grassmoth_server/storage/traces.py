"""Collect scratch LLM traces for publication in study history."""

import re

from grassmoth_server.llm.traces import LLMTrace
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module

_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def collect_run_traces(workspace_id: str, call_id: str) -> dict[str, LLMTrace]:
    """Collect the accumulated trace snapshots before scratch is swept."""
    llm_root = storage.join(data_module.call_dir(workspace_id, call_id), "llm")
    traces: dict[str, LLMTrace] = {}
    for subroutine_root in sorted(storage.listdir(llm_root)):
        subroutine_id = subroutine_root.rstrip("/").rsplit("/", 1)[-1]
        source = storage.join(subroutine_root, "trace.json")
        if storage.exists(source):
            if _TRACE_ID.fullmatch(subroutine_id) is None:
                raise ValueError(f"Invalid attempt trace subroutine id: {subroutine_id!r}")
            traces[subroutine_id] = LLMTrace.model_validate_json(storage.read_text(source))
    return traces
