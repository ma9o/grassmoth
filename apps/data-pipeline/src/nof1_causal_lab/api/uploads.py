"""HTTP adapter for public input-file staging."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from nof1_causal_lab.api.policy import raise_study_error
from nof1_causal_lab.study.interface import StudyError
from nof1_causal_lab.study.workspaces import stage_input

uploads_router = APIRouter(prefix="/api")


@uploads_router.post("/upload", response_model=str)
async def upload_file(
    file: Annotated[UploadFile, File()], workspace_id: Annotated[str, Form(alias="workspaceId")]
) -> str:
    """Stage one named raw input file for prepare_data; that call captures its SHA-256."""
    result = stage_input(workspace_id, file.filename or "", await file.read())
    if isinstance(result, StudyError):
        raise_study_error(result)
    return result
