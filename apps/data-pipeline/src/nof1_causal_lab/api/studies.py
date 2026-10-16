"""Content-named scientific calls and their slim attempt journal over HTTP."""

from __future__ import annotations

import asyncio
import pathlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast
from uuid import uuid4

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Response,
)
from fastapi.responses import JSONResponse

from nof1_causal_lab.actions.call_logs import collect_call_log
from nof1_causal_lab.actions.contracts.call_state import (
    CallProgress,
    CompletedCall,
    PendingCall,
    RunningCall,
)
from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.polling import (
    ActionPoll,
    FailedPoll,
    RunningAction,
    RunningPoll,
)
from nof1_causal_lab.actions.contracts.requests import (
    ActionInput,
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
    call_identity,
)
from nof1_causal_lab.actions.input_resolution import resolve_action_inputs
from nof1_causal_lab.api.clients import TemporalClientProvider, study_clients
from nof1_causal_lab.api.policy import _require_actions_enabled, _safe_workspace_id, actions_enabled
from nof1_causal_lab.api.responses import MessagePackResponse
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ActionId, CallId, GitOid, RevisionSelector
from nof1_causal_lab.artifacts.observations.preparation import SourceFolder
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.queries.action_outputs import completed_call_msgpack
from nof1_causal_lab.study.records import (
    ActionMessage,
    Applied,
    Attempt,
    RecordDependency,
    StudyRevision,
    record_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from temporalio.client import WorkflowHandle

    from nof1_causal_lab.actions.temporal.study.workflow import StudyWorkflow

router = APIRouter(prefix="/api/studies", default_response_class=MessagePackResponse)

_CALL_PROGRESS_TYPE = cast("type[CallProgress | None]", CallProgress | None)
_CALL_STATE_TYPE = cast("type[CallProgress]", CallProgress)


async def _study_handle(
    workspace_id: str, clients: TemporalClientProvider
) -> WorkflowHandle[StudyWorkflow, None]:
    from temporalio.common import WorkflowIDConflictPolicy

    from nof1_causal_lab.actions.temporal.client import STUDY_TASK_QUEUE, study_workflow_id
    from nof1_causal_lab.actions.temporal.study.messages import StudyInit
    from nof1_causal_lab.actions.temporal.study.workflow import StudyWorkflow

    return await (await clients.get()).start_workflow(
        StudyWorkflow.run,
        StudyInit(
            workspace_id=workspace_id, initial_seq=StudyRepository(workspace_id).latest_seq()
        ),
        id=study_workflow_id(workspace_id),
        task_queue=STUDY_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _running_action(
    workspace_id: str, clients: TemporalClientProvider
) -> RunningAction | None:
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.temporal.client import RUNNING_ACTION_MEMO, study_workflow_id

    try:
        description = (
            await (await clients.get())
            .get_workflow_handle(study_workflow_id(workspace_id))
            .describe()
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return None
        raise
    if description.status != WorkflowExecutionStatus.RUNNING:
        return None
    running = await description.memo_value(RUNNING_ACTION_MEMO, None, type_hint=RunningCall)
    if running is None:
        return None
    log = await asyncio.to_thread(
        collect_call_log,
        workspace_id,
        seq=running.seq,
        attempt_id=running.attempt_id,
        messages=running.messages,
        at=datetime.now(UTC),
    )
    return RunningAction(
        call_id=call_identity(running.request),
        action=running.request.action,
        request=running.request,
        messages=log.messages,
    )


def _stage_sources(
    workspace_id: str,
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
    contents: Mapping[str, bytes],
) -> None:
    if not isinstance(request, PrepareDataRequest):
        return
    for filename, identity in request.input.source.hashes.items():
        path = (
            pathlib.Path(data_module.scratch_dir(workspace_id))
            / "source-files"
            / identity
            / filename
        )
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{uuid4().hex}.partial")
        partial.write_bytes(contents[filename])
        partial.replace(path)


def _saved_response(workspace_id: str, revision: StudyRevision) -> Response:
    return Response(
        content=completed_call_msgpack(workspace_id, revision),
        media_type=MessagePackResponse.media_type,
    )


async def _dispatch_action(
    workspace_id: str,
    body: ActionInput,
    clients: TemporalClientProvider,
) -> ActionPoll | Response:
    """Resolve and deduplicate a call; saved failures and successes share the same identity."""
    from temporalio.client import WorkflowUpdateStage

    from nof1_causal_lab.actions.temporal.study.messages import ActionRequest

    workspace_id = _safe_workspace_id(workspace_id)
    if not actions_enabled() and not storage.exists(
        storage.join(data_module.study_dir(workspace_id), "history.git")
    ):
        raise HTTPException(403, "This read-only facade answers saved calls only")
    repository = StudyRepository(workspace_id)
    try:
        request, contents = await asyncio.to_thread(resolve_action_inputs, repository, body)
    except StudyLookupError as exc:
        raise HTTPException(422, str(exc)) from exc
    saved = await asyncio.to_thread(repository.saved_call, request)
    if saved is not None:
        return await asyncio.to_thread(_saved_response, workspace_id, saved)
    _require_actions_enabled()
    identity = call_identity(request)
    handle = await _study_handle(workspace_id, clients)
    progress = cast(
        "CallProgress | None",
        await handle.query("call_progress", identity, result_type=_CALL_PROGRESS_TYPE),
    )
    if progress is None:
        await asyncio.to_thread(_stage_sources, workspace_id, request, contents)
        attempt_id = uuid4()
        await handle.start_update(
            "execute_action",
            ActionRequest(request=request, attempt_id=attempt_id),
            id=str(attempt_id),
            result_type=_CALL_STATE_TYPE,
            wait_for_stage=WorkflowUpdateStage.ACCEPTED,
        )
        progress = cast(
            "CallProgress | None",
            await handle.query("call_progress", identity, result_type=_CALL_PROGRESS_TYPE),
        )
    if progress is None:
        raise RuntimeError("Accepted call has no workflow progress")
    return await _progress_response(workspace_id, progress)


async def _progress_response(workspace_id: str, progress: CallProgress) -> ActionPoll | Response:
    if isinstance(progress, PendingCall):
        return RunningPoll(call_id=call_identity(progress.request), action=progress.request.action)
    if isinstance(progress, RunningCall):
        log = await asyncio.to_thread(
            collect_call_log,
            workspace_id,
            seq=progress.seq,
            attempt_id=progress.attempt_id,
            messages=progress.messages,
            at=datetime.now(UTC),
        )
        return RunningPoll(
            call_id=call_identity(progress.request),
            action=progress.request.action,
            messages=log.messages,
        )
    if progress.commit_id is not None:
        return await asyncio.to_thread(
            _saved_response, workspace_id, StudyRepository(workspace_id).record(progress.commit_id)
        )
    attempt = progress.attempt
    if attempt.request is None or isinstance(attempt.outcome, Applied):
        raise RuntimeError("An unpublished completion must own a failed call")
    log = await asyncio.to_thread(
        collect_call_log,
        workspace_id,
        seq=progress.seq,
        attempt_id=progress.attempt_id,
        messages=progress.messages,
        at=datetime.now(UTC),
        failure=attempt.outcome,
    )
    return FailedPoll(
        call_id=call_identity(attempt.request),
        action=attempt.action,
        commit_id=None,
        messages=log.messages,
    )


@router.get(
    "/{workspace_id}/{action}/{call_id}", response_model=ActionPoll, operation_id="poll_action"
)
async def poll_action(
    workspace_id: str,
    action: ActionId,
    call_id: CallId,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures."""
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.temporal.client import study_workflow_id

    workspace_id = _safe_workspace_id(workspace_id)
    if storage.exists(storage.join(data_module.study_dir(workspace_id), "history.git")):
        saved = await asyncio.to_thread(StudyRepository(workspace_id).call, call_id)
        if saved is not None:
            if saved.record.attempt.action != action:
                raise HTTPException(404, "Unknown action call")
            return await asyncio.to_thread(_saved_response, workspace_id, saved)
    if not actions_enabled():
        raise HTTPException(404, "Unknown action call")
    try:
        handle = (await clients.get()).get_workflow_handle(study_workflow_id(workspace_id))
        progress = cast(
            "CallProgress | None",
            await handle.query("call_progress", call_id, result_type=_CALL_PROGRESS_TYPE),
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(404, "Unknown action call") from exc
        raise
    if progress is None:
        raise HTTPException(404, "Unknown action call")
    actual_action = (
        progress.attempt.action if isinstance(progress, CompletedCall) else progress.request.action
    )
    if actual_action != action:
        raise HTTPException(404, "Unknown action call")
    return await _progress_response(workspace_id, progress)


@router.post(
    "/{workspace_id}/edit_question", response_model=ActionPoll, operation_id="edit_question"
)
async def edit_question(
    workspace_id: str,
    body: Annotated[EditQuestionRequest, Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/edit_model", response_model=ActionPoll, operation_id="edit_model")
async def edit_model(
    workspace_id: str,
    body: Annotated[EditModelRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Merge input.dynamical_model_spec into input.parent_ref: start empty for a question revision, or retain omitted fields from a model revision and its pinned question. Null entity entries delete their IDs. Prune constructs outside the outcome ancestry with warnings, then validate and evaluate data-independent model checks. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/prepare_data", response_model=ActionPoll, operation_id="prepare_data")
async def prepare_data(
    workspace_id: str,
    body: Annotated[
        PrepareDataRequest[RevisionSelector, SourceFolder], Body(discriminator="action")
    ],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
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
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/fit", response_model=ActionPoll, operation_id="fit")
async def fit(
    workspace_id: str,
    body: Annotated[FitRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Fit target.new model and data, or continue target.fit_ref at its last retained state. Counts request additional draws; zero warmup freezes tuning and algorithm changes require warmup. Zero warmup and zero samples compile both sweep programs without execution. Zero-draw fits return checks and inference without a conditioned model. repeat distinguishes timing attempts. GET polls call_id."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/simulate", response_model=ActionPoll, operation_id="simulate")
async def simulate(
    workspace_id: str,
    body: Annotated[SimulateRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Simulate input.dynamical_model_spec_ref using input.simulation. The model owns deterministic inputs and retained trajectory coordinates; authored initial states apply at simulation.start. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body requires data and report; report owns single or paired numerical evidence, calculated summaries and causal findings; messages retain traces. GET polls call_id."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/data_diff", response_model=ActionPoll, operation_id="data_diff")
async def data_diff(
    workspace_id: str,
    body: Annotated[DataDiffRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Compare input.left_ref and input.right_ref, each a nonempty list of {revision, replicate_index} references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require explicit gitrefs. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/model_diff", response_model=ActionPoll, operation_id="model_diff")
async def model_diff(
    workspace_id: str,
    body: Annotated[ModelDiffRequest[RevisionSelector], Body(discriminator="action")],
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Compare input.before_ref and input.after_ref as saved DynamicalModelSpec documents. Return changes using the same partial DynamicalModelSpec contract as edit_model: omissions are unchanged and null map entries delete identities. Retain the patch in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body."""
    return await _dispatch_action(workspace_id, body, clients)


class TimelineRecord(Value):
    """A replayable call log, excluding every scientific result and check payload."""

    seq: int
    ts: str
    messages: tuple[ActionMessage, ...]
    trace_ids: tuple[str, ...]
    attempt: Attempt[
        ActionId, ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid], None
    ]


class TimelineRevision(Value):
    """A lightweight attempt record linked to its call identity and Git publication."""

    call_id: CallId | None
    commit_id: GitOid
    parent_ids: tuple[GitOid, ...]
    record: TimelineRecord


class TimelineResponse(Value):
    """Journal attempts, declared input dependencies, and the currently running action."""

    attempts: tuple[TimelineRevision, ...]
    dependencies: tuple[RecordDependency, ...]
    running: RunningAction | None


@router.get(
    "/{workspace_id}/timeline", response_model=TimelineResponse, response_class=JSONResponse
)
async def get_timeline(
    workspace_id: str, clients: Annotated[TemporalClientProvider, Depends(study_clients)]
) -> TimelineResponse:
    """Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id."""
    workspace_id = _safe_workspace_id(workspace_id)
    records = await asyncio.to_thread(StudyRepository(workspace_id).attempts)
    attempts = []
    for revision in records:
        record = revision.record
        outcome = record.attempt.outcome
        summary = (
            Applied[None](
                result=None,
                effects=ActionEffects(
                    produced=outcome.effects.produced, retracted=outcome.effects.retracted
                ),
            )
            if isinstance(outcome, Applied)
            else outcome
        )
        attempts.append(
            TimelineRevision(
                call_id=call_identity(record.attempt.request)
                if record.attempt.request is not None
                else None,
                commit_id=revision.commit_id,
                parent_ids=revision.parent_ids,
                record=TimelineRecord(
                    seq=record.seq,
                    ts=record.ts,
                    messages=record.messages,
                    trace_ids=record.trace_ids,
                    attempt=Attempt[
                        ActionId,
                        ScientificActionRequest
                        | DataDiffRequest[GitOid]
                        | ModelDiffRequest[GitOid],
                        None,
                    ](
                        action=record.attempt.action,
                        request=record.attempt.request,
                        outcome=summary,
                    ),
                ),
            )
        )
    return TimelineResponse(
        attempts=tuple(attempts),
        dependencies=tuple(record_dependencies(records)),
        running=await _running_action(workspace_id, clients) if actions_enabled() else None,
    )
