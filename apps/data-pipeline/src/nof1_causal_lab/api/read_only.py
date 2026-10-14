"""Read-only study facade: the hosted viewer's entire backend.

Serves the same saved action results, slim timeline, workspace list and cached
model comparisons as the full tool server. Deployments set READ_ONLY_FACADE=1
so unsaved writing calls and uploads return 403. No Temporal connection, LLM
execution, fit or simulation is started by a saved call or model comparison.

Deployed as a Modal ASGI app (see :mod:`nof1_causal_lab.api.deployment`),
or run locally::

    READ_ONLY_FACADE=1 uv run uvicorn --factory \
        nof1_causal_lab.api.read_only:create_read_facade_app --port 8100
"""

from __future__ import annotations

from fastapi import FastAPI

from nof1_causal_lab.api.clients import TemporalClientProvider
from nof1_causal_lab.api.studies import router as study_router
from nof1_causal_lab.api.uploads import uploads_router
from nof1_causal_lab.api.workspaces import workspaces_router


def create_read_facade_app() -> FastAPI:
    """Build the HTTP facade with study, workspace, and upload routes and a shared client provider."""
    app = FastAPI(title="Study Read Facade")
    app.state.study_clients = TemporalClientProvider()
    app.include_router(study_router)
    app.include_router(workspaces_router)
    app.include_router(uploads_router)
    return app
