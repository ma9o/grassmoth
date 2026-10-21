"""HTTP adapter to the public nof1 study library."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Response
from fastapi.responses import JSONResponse

from nof1_causal_service.api.clients import study_clients
from nof1_causal_service.api.policy import raise_study_error
from nof1_causal_service.api.responses import MessagePackResponse
from nof1_causal_service.public.contracts import (
    ActionId,
    ActionPoll,
    CallId,
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    RevisionSelector,
    SimulateRequest,
    SourceFolder,
    TimelineResponse,
)
from nof1_causal_service.study import interface as studies
from nof1_causal_service.study.interface import ActionResult, StudyError, TemporalClientProvider

router = APIRouter(prefix="/api/studies", default_response_class=MessagePackResponse)


def _http_response(result: ActionResult | StudyError) -> Response:
    if isinstance(result, StudyError):
        raise_study_error(result)
    return Response(content=result.payload, media_type=MessagePackResponse.media_type)


@router.get(
    "/{workspace_id}/{action}/{call_id}", response_model=ActionPoll, operation_id="poll_action"
)
async def poll_action(
    workspace_id: str,
    action: ActionId,
    call_id: CallId,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures."""
    return _http_response(await studies.poll_action(workspace_id, action, call_id, clients))


@router.post(
    "/{workspace_id}/edit_question", response_model=ActionPoll, operation_id="edit_question"
)
async def edit_question(
    workspace_id: str,
    body: Annotated[EditQuestionRequest, Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/edit_model", response_model=ActionPoll, operation_id="edit_model")
async def edit_model(
    workspace_id: str,
    body: Annotated[EditModelRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Merge input.dynamical_model_spec into input.parent_ref: start empty for a question revision, or retain omitted fields from a model revision and its pinned question. Null entity entries delete their IDs. Prune constructs outside the outcome ancestry with warnings, then validate and evaluate data-independent model checks. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/prepare_data", response_model=ActionPoll, operation_id="prepare_data")
async def prepare_data(
    workspace_id: str,
    body: Annotated[
        PrepareDataRequest[RevisionSelector, SourceFolder], Body(discriminator="action")
    ],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Prepare observations from uploaded tables using the selected model's definitions.

    The source is a folder under ``data/{workspace_id}/``, including subfolders.
    CSV and Parquet tables must supply a date or datetime timestamp column and
    are concatenated in captured file order. Source bytes and revision selectors
    are pinned before computing call identity.

    Args:
        workspace_id: Study workspace containing the source folder and revision
            history.
        body: Preparation request selecting a model, source folder, and computed
            rules or semantic extraction instructions for its observation IDs.
        clients: Provider of the shared Temporal connection used to dispatch
            or retrieve the preparation workflow.

    Returns:
        Current call status or a cached HTTP response for the same call. Polling
        by call ID exposes progress messages and, on success, the prepared
        observations and available profiles.
    """
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/fit", response_model=ActionPoll, operation_id="fit")
async def fit(
    workspace_id: str,
    body: Annotated[FitRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Fit target.new model and data, or continue target.fit_ref at its last retained state. Counts request additional draws; zero warmup freezes tuning and algorithm changes require warmup. Zero warmup and zero samples compile both sweep programs without execution. Zero-draw fits return checks and inference without a conditioned model. repeat distinguishes timing attempts. GET polls call_id."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/simulate", response_model=ActionPoll, operation_id="simulate")
async def simulate(
    workspace_id: str,
    body: Annotated[SimulateRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Simulate input.dynamical_model_spec_ref using input.simulation. The model owns deterministic inputs and retained trajectory coordinates; authored initial states apply at simulation.start. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body requires data and report; report owns single or paired numerical evidence, calculated summaries and causal findings; messages retain traces. GET polls call_id."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/data_diff", response_model=ActionPoll, operation_id="data_diff")
async def data_diff(
    workspace_id: str,
    body: Annotated[DataDiffRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Compare input.left_ref and input.right_ref, each a nonempty list of {revision, replicate_index} references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require explicit gitrefs. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.post("/{workspace_id}/model_diff", response_model=ActionPoll, operation_id="model_diff")
async def model_diff(
    workspace_id: str,
    body: Annotated[ModelDiffRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> Response:
    """Compare input.before_ref and input.after_ref as saved DynamicalModelSpec documents. Return changes using the same partial DynamicalModelSpec contract as edit_model: omissions are unchanged and null map entries delete identities. Retain the patch in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return _http_response(await studies.submit_action(workspace_id, body, clients))


@router.get(
    "/{workspace_id}/timeline", response_model=TimelineResponse, response_class=JSONResponse
)
async def get_timeline(
    workspace_id: str, clients: Annotated[TemporalClientProvider, Depends(study_clients)]
) -> TimelineResponse:
    """Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id."""
    result = await studies.get_timeline(workspace_id, clients)
    if isinstance(result, StudyError):
        raise_study_error(result)
    return result
