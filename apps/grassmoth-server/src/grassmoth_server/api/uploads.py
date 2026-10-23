"""HTTP adapter for public input-file staging."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from grassmoth_server.actions.contracts.polling import StudyError
from grassmoth_server.api.policy import raise_study_error
from grassmoth_server.study.workspaces import stage_input

uploads_router = APIRouter(prefix="/api")


@uploads_router.post("/upload", response_model=str)
async def upload_file(
    file: Annotated[UploadFile, File()], workspace_id: Annotated[str, Form(alias="workspaceId")]
) -> str:
    """Stage one named raw input file in the workspace's input folder. prepare_data reads that folder when input.source is "input" and captures each file's SHA-256."""
    result = stage_input(workspace_id, file.filename or "", await file.read())
    if isinstance(result, StudyError):
        raise_study_error(result)
    return result
