"""HTTP uploads boundary."""

from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from nof1_causal_lab.api.policy import _require_actions_enabled, _safe_workspace_id
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module

uploads_router = APIRouter(prefix="/api")


@uploads_router.post("/upload", response_model=str)
async def upload_file(
    file: Annotated[UploadFile, File()], workspace_id: Annotated[str, Form(alias="workspaceId")]
) -> str:
    """Stage one named raw input file for prepare_data; that call captures its SHA-256."""
    _require_actions_enabled()
    workspace_id = _safe_workspace_id(workspace_id)
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not filename or filename in {".", ".."}:
        raise HTTPException(400, "Invalid file name")
    directory = data_module.input_dir(workspace_id)
    storage.makedirs(directory)
    with storage.open_file(storage.join(directory, filename), "wb") as handle:
        handle.write(await file.read())
    return f"{workspace_id}/input/{filename}"
