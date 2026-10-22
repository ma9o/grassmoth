"""HTTP adapter for public workspace discovery."""

from collections.abc import Mapping

from fastapi import APIRouter, Response

from grassmoth_server.study import workspaces
from grassmoth_server.study.interface import actions_enabled

workspaces_router = APIRouter(prefix="/api")


@workspaces_router.get("/workspaces", response_model=Mapping[str, str | None])
def list_workspaces(response: Response) -> Mapping[str, str | None]:
    """Available workspaces and their immutable study questions.

    X-Actions-Enabled reports whether the facade accepts new calls.
    """
    response.headers["X-Actions-Enabled"] = "true" if actions_enabled() else "false"
    return workspaces.list_workspaces()
