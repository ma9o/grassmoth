"""HTTP adapter for submitting and reading Temporal-backed study calls."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request, Response
from fastapi.responses import JSONResponse

from grassmoth_server.actions.contracts.identity import ActionId, CallId, RevisionSelector
from grassmoth_server.actions.contracts.polling import ActionPoll, ActionResult, StudyError
from grassmoth_server.actions.contracts.preparation import SourceFolder
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from grassmoth_server.actions.contracts.timeline import TimelineResponse
from grassmoth_server.api.clients import temporal_client
from grassmoth_server.api.policy import raise_study_error
from grassmoth_server.api.responses import MessagePackResponse, prefers_json
from grassmoth_server.study import calls
from grassmoth_server.study.calls import TemporalClientProvider
from grassmoth_server.study.persistence.codec import json_result

router = APIRouter(prefix="/api/studies", default_response_class=MessagePackResponse)


def _http_response(result: ActionResult | StudyError, request: Request) -> Response:
    if isinstance(result, StudyError):
        raise_study_error(result)
    if prefers_json(request.headers.get("accept", "")):
        return Response(
            content=json_result(result.payload),
            media_type="application/json",
            headers={"Vary": "Accept"},
        )
    return Response(
        content=result.payload,
        media_type=MessagePackResponse.media_type,
        headers={"Vary": "Accept"},
    )


@router.get(
    "/{workspace_id}/{action}/{call_id}", response_model=ActionPoll, operation_id="poll_action"
)
async def poll_action(
    workspace_id: str,
    request: Request,
    action: ActionId,
    call_id: CallId,
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures."""
    return _http_response(
        await calls.poll_action(workspace_id, action, call_id, temporal_client), request
    )


@router.post(
    "/{workspace_id}/edit_question", response_model=ActionPoll, operation_id="edit_question"
)
async def edit_question(
    workspace_id: str,
    request: Request,
    body: Annotated[EditQuestionRequest, Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.post("/{workspace_id}/edit_model", response_model=ActionPoll, operation_id="edit_model")
async def edit_model(
    workspace_id: str,
    request: Request,
    body: Annotated[EditModelRequest[RevisionSelector], Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Merge input.dynamical_model_spec into input.parent_ref: start empty for a question revision, or retain omitted fields from a model revision and its pinned question. Null entity entries delete their IDs. Prune constructs outside the outcome ancestry with warnings, then validate and evaluate data-independent model checks. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.post("/{workspace_id}/prepare_data", response_model=ActionPoll, operation_id="prepare_data")
async def prepare_data(
    workspace_id: str,
    request: Request,
    body: Annotated[
        PrepareDataRequest[RevisionSelector, SourceFolder], Body(discriminator="action")
    ],
    clients: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Prepare observations from uploaded tables using the selected model's definitions. input.source is a folder under data/{workspace_id}/, including subfolders. CSV and Parquet tables must supply a date or datetime timestamp column and are concatenated in captured file order. Source bytes and revision selectors are pinned before computing call identity. GET polls call_id."""
    return _http_response(await calls.submit_action(workspace_id, body, clients), request)


@router.post("/{workspace_id}/fit", response_model=ActionPoll, operation_id="fit")
async def fit(
    workspace_id: str,
    request: Request,
    body: Annotated[FitRequest[RevisionSelector], Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Fit target.new model and data, or continue target.fit_ref at its last retained state. Counts request additional draws; zero warmup freezes tuning and algorithm changes require warmup. Zero warmup and zero samples compile both sweep programs without execution. Zero-draw fits return checks and inference without a conditioned model. repeat distinguishes timing attempts. GET polls call_id."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.post("/{workspace_id}/simulate", response_model=ActionPoll, operation_id="simulate")
async def simulate(
    workspace_id: str,
    request: Request,
    body: Annotated[SimulateRequest[RevisionSelector], Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Simulate input.dynamical_model_spec_ref using input.simulation. The model owns deterministic inputs and retained trajectory coordinates; authored initial states apply at simulation.start. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body requires data and report; report owns single or paired numerical evidence, calculated summaries and causal findings; messages retain traces. GET polls call_id."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.post("/{workspace_id}/data_diff", response_model=ActionPoll, operation_id="data_diff")
async def data_diff(
    workspace_id: str,
    request: Request,
    body: Annotated[DataDiffRequest[RevisionSelector], Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Compare input.left_ref and input.right_ref, each a nonempty list of {revision, replicate_index} references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require exact Git hashes. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.post("/{workspace_id}/model_diff", response_model=ActionPoll, operation_id="model_diff")
async def model_diff(
    workspace_id: str,
    request: Request,
    body: Annotated[ModelDiffRequest[RevisionSelector], Body(discriminator="action")],
    temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)],
) -> Response:
    """Compare input.before_ref and input.after_ref as saved DynamicalModelSpec documents. Return changes using the same partial DynamicalModelSpec contract as edit_model: omissions are unchanged and null map entries delete identities. Retain the patch in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return _http_response(await calls.submit_action(workspace_id, body, temporal_client), request)


@router.get(
    "/{workspace_id}/timeline", response_model=TimelineResponse, response_class=JSONResponse
)
async def get_timeline(
    workspace_id: str, temporal_client: Annotated[TemporalClientProvider, Depends(temporal_client)]
) -> TimelineResponse:
    """Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id."""
    result = await calls.get_timeline(workspace_id, temporal_client)
    if isinstance(result, StudyError):
        raise_study_error(result)
    return result
