"""Study HTTP API: content-named Temporal calls and immutable viewer projections.

Run with ``uv run uvicorn nof1_causal_lab.api.app:app --port 8100``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nof1_causal_lab.api.studies import router as study_router
from nof1_causal_lab.api.uploads import uploads_router
from nof1_causal_lab.api.workspaces import workspaces_router
from nof1_causal_lab.study.interface import TemporalClientProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_API_DESCRIPTION = """\
The public scientific interface has seven actions: `edit_question`, `edit_model`,
`prepare_data`, `fit`, `simulate`, `data_diff`, and `model_diff`.
Start a call with `POST /api/studies/{workspace_id}/{action}` using
`{action, input, reasoning}`. Each action owns its typed `input`; optional top-level
`reasoning` explains the caller's intent and is excluded from call identity.

POST and polling GET return `{call_id, action, status, commit_id, body, messages}`.
Action responses use `application/msgpack`; requests remain JSON. Python callers
can decode a saved response with `msgpack.unpackb(response_bytes, raw=False)`;
the shared TypeScript client decodes responses automatically. `arrays` entries
carry binary NPY bytes, read with `numpy.load(io.BytesIO(entry["npy"]), allow_pickle=False)`
in Python or `readNumericalArray(entry)` in TypeScript. NPY retains dtype, shape,
NaN and infinities without scalar JSON or base64 conversion. Timeline, workspace,
upload and HTTP error responses remain JSON.
`status` is `running`, `failed`, or `success`. `commit_id` is null before publication;
a journaled failure also has a commit. `body` is null while running or failed and
contains only that action's typed scientific output on success.
`messages` is the sole execution-log accumulator: lifecycle entries, structured
progress, full LLM/tool traces, and failure details all remain there.

Poll `GET /api/studies/{workspace_id}/{action}/{call_id}`. GET only reads an existing
call; it never starts a workflow, executes an action, or retries a failure. Unknown
call IDs return 404. Identical resolved inputs reuse running calls and both successful
and failed completed calls, preserving the first request's reasoning.

Revision selectors accept exact Git hashes or `"latest"`. POST resolves selectors
once before computing `call_id`; GET keeps that selection when newer revisions appear.
Missing valid inputs return HTTP 422. `prepare_data` takes `input.source` as one
folder name under `data/{workspace_id}/`, such as `"input"`. Every file beneath
that folder, including subfolders, is captured and hashed before call identity.
Polling saved calls does not require the original source folder.
`edit_model` selects a question or model parent through `input.parent_ref`;
a question parent starts a new model; a model parent supplies its definition and pinned
question. The same DynamicalModelSpec accepts partial definitions: omission retains fields and null
entity entries delete their IDs. The editing boundary prunes outcome-unrelated components
with warnings and validates the assembled model. Every study starts with `edit_question`.

Each action owns its successful `body`; input state is read from its producing calls.
For `model_diff` and `data_diff`, `body` is the complete saved comparison.
Fit and simulation outputs include their own retained arrays and paths without paging.
Display summaries use null for missing values; numerical buffers retain native missingness.
Reports and checks are retained with their action in Git and never recomputed by GET,
including after cache deletion or code changes. Comparisons and failures remain
journal leaves without advancing scientific state.

`GET /api/studies/{workspace_id}/timeline` returns call IDs, retained requests,
outcome summaries, lifecycle messages, trace IDs, input dependencies and the running
call. The read-only web viewer follows recorded input dependencies and uses their call
IDs to compose saved results and logs. For example, a simulation reads its model
from the call named by its model dependency.
Agents create studies and submit actions directly to the facade.
`GET /api/workspaces` lists studies and questions; X-Actions-Enabled reports whether
new actions are enabled. `POST /api/upload` stages a named input file using
`multipart/form-data` with `workspaceId` and `file`.

With READ_ONLY_FACADE=1, existing calls remain readable. POST for an unsaved call
returns 403; GET for an unknown call returns 404.
"""


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from nof1_causal_lab.config import configure_jax_persistent_cache

    configure_jax_persistent_cache()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="nof1-causal-lab study API",
    description=_API_DESCRIPTION,
    docs_url="/api/docs",
)
app.state.study_clients = TemporalClientProvider()

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
