"""Study HTTP API: content-named Temporal calls and immutable viewer projections.

Run with ``uv run uvicorn grassmoth_server.api.app:app --port 8100``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi.middleware.cors import CORSMiddleware

from grassmoth_server.api.schema import StudyAPI
from grassmoth_server.api.studies import router as study_router
from grassmoth_server.api.uploads import uploads_router
from grassmoth_server.api.workspaces import workspaces_router
from grassmoth_server.study.calls import TemporalClientProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_API_DESCRIPTION = """\
Start a study with `edit_question`. Endpoint examples follow; `/api/docs` has
complete input/output schemas, defaults and alternatives.

- **Submit and poll:** POST JSON `{action, input, reasoning}` to
  `/api/studies/{workspace_id}/{action}`; `reasoning` is optional. Poll GET
  `/api/studies/{workspace_id}/{action}/{call_id}` until `status` is `success` or
  `failed`. GET never starts or retries work; an unknown call returns 404.
- **Results and errors:** `body` contains the action's scientific output only on
  success; otherwise it is null. `messages` holds progress, traces and failure
  details. Completed results are saved and GET never recomputes them.
- **JSON responses:** send `Accept: application/json` on action POST/GET to read
  results with curl and jq. Numerical values are nested lists; non-finite values
  are null. Workspace, timeline, upload and HTTP error responses are also JSON.
- **Call reuse:** revision selectors accept exact Git hashes or `"latest"`, pinned
  once at POST. Identical resolved inputs reuse running calls, successes **and
  failures**. Changing `reasoning` does not create a new call; change the inputs
  to make a new attempt.
- **Artifact references:** response `commit_id` identifies the call's journal entry.
  For subsequent inputs, GET `/api/studies/{workspace_id}/timeline`, find the call,
  and select the required artifact by `artifact_id` in
  `record.attempt.outcome.effects.produced`. Use its `revision` for model,
  question, data and fit references.
- **Read-only access:** `/api/workspaces` reports `X-Actions-Enabled`. With
  `READ_ONLY_FACADE=1`, saved calls remain readable; an unsaved POST returns 403.
"""


@asynccontextmanager
async def _lifespan(_app: StudyAPI) -> AsyncIterator[None]:
    from grassmoth_server.config import configure_jax_persistent_cache

    configure_jax_persistent_cache()
    yield


app = StudyAPI(
    lifespan=_lifespan,
    title="grassmoth study API",
    description=_API_DESCRIPTION,
    docs_url="/api/docs",
)
app.state.temporal_client = TemporalClientProvider()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Actions-Enabled"],
)

app.include_router(study_router)
app.include_router(workspaces_router)
app.include_router(uploads_router)
