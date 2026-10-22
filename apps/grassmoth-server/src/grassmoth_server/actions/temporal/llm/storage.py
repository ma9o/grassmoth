"""Storage helpers for Temporal LLM subroutines.

Conversations, tool results, and traces live in run sidecars so their large,
domain-specific payloads do not have to be carried solely by Temporal history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, overload
from uuid import uuid4

from pydantic import TypeAdapter

from grassmoth.public import JsonObject
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module

if TYPE_CHECKING:
    from grassmoth_server.llm.traces import LLMTrace


def write_subroutine_trace(path: str, trace: LLMTrace) -> None:
    """Publish a complete trace snapshot atomically for concurrent call polling."""
    payload = trace.model_dump_json()
    if storage.is_remote():
        storage.write_text(path, payload)
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{uuid4().hex}.partial")
        partial.write_text(payload)
        partial.replace(target)


def subroutine_root(workspace_id: str, run_id: str, subroutine_id: str) -> str:
    """Locate a subroutine's scratch storage within its workspace and run."""
    return storage.join(data_module.scratch_run_dir(workspace_id, run_id), "llm", subroutine_id)


def subroutine_conversation_path(
    workspace_id: str,
    run_id: str,
    subroutine_id: str,
    name: str,
) -> str:
    """Locate a named conversation snapshot within a subroutine's scratch storage."""
    return storage.join(subroutine_root(workspace_id, run_id, subroutine_id), "conversation", name)


def write_subroutine_json(path: str, value: object) -> None:
    """Serialize a subroutine payload as JSON at the supplied storage path."""
    storage.write_text(path, json.dumps(value))


@overload
def read_subroutine_json(path: str) -> JsonObject: ...


@overload
def read_subroutine_json[ResultT](path: str, target: type[ResultT]) -> ResultT: ...


def read_subroutine_json(path: str, target: object = JsonObject) -> object:
    """Read stored JSON and parse it against the requested payload type."""
    return TypeAdapter(target).validate_json(storage.read_text(path))
