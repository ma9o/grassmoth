"""HTTP adapter for public workspace discovery."""

from fastapi import APIRouter, Response

from grassmoth_server.study import workspaces
from grassmoth_server.study.calls import actions_enabled

workspaces_router = APIRouter(prefix="/api")


@workspaces_router.get("/workspaces")
def list_workspaces(response: Response) -> tuple[str, ...]:
    """List workspace IDs in sorted order.

    X-Actions-Enabled reports whether the facade accepts new calls.
    """
    response.headers["X-Actions-Enabled"] = "true" if actions_enabled() else "false"
    return workspaces.list_workspaces()
