"""Internal study operations using the production Temporal lifecycle and retained results."""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from grassmoth.public import GitOid
from grassmoth_server.actions.call_logs import collect_call_log
from grassmoth_server.actions.contracts.call_state import (
    CallProgress,
    CompletedCall,
    PendingCall,
    RunningCall,
)
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.identity import ActionId, CallId
from grassmoth_server.actions.contracts.polling import (
    ActionResult,
    FailedPoll,
    RunningAction,
    RunningPoll,
    StudyError,
)
from grassmoth_server.actions.contracts.requests import (
    ActionInput,
    DataDiffRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    call_identity,
)
from grassmoth_server.actions.contracts.timeline import (
    TimelineRecord,
    TimelineResponse,
    TimelineRevision,
)
from grassmoth_server.actions.input_resolution import resolve_action_inputs
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
from grassmoth_server.study.records import Applied, Attempt, StudyRevision, record_dependencies

if TYPE_CHECKING:
    from collections.abc import Mapping

    from temporalio.client import Client, WorkflowHandle

    from grassmoth_server.actions.temporal.study.workflow import StudyWorkflow


def actions_enabled() -> bool:
    """Read the deployment's permission to submit new study actions."""
    return os.environ.get("READ_ONLY_FACADE") != "1"


def parse_workspace_id(value: str) -> str | StudyError:
    """Parse an external workspace name at its single input boundary."""
    workspace_id = value.strip()
    if (
        not workspace_id
        or len(workspace_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id) is None
    ):
        return StudyError(code="invalid_request", detail="Invalid workspaceId format")
    return workspace_id


def _read_only() -> StudyError:
    return StudyError(code="read_only", detail="This read-only facade answers saved calls only")


class TemporalClientProvider:
    """One lazily connected Temporal client owned by the facade."""

    def __init__(self) -> None:
        """Initialize lazy Temporal connection state protected by an asynchronous lock."""
        self._lock = asyncio.Lock()
        self._client: Client | None = None

    async def get(self) -> Client:
        """Connect to Temporal once and share the resulting client across concurrent requests."""
        async with self._lock:
            if self._client is None:
                from grassmoth_server.actions.temporal.client import connect_client

                self._client = await connect_client()
            return self._client


_CALL_PROGRESS_TYPE = cast("type[CallProgress | None]", CallProgress | None)
_CALL_STATE_TYPE = cast("type[CallProgress]", CallProgress)


async def _study_handle(
    workspace_id: str, clients: TemporalClientProvider
) -> WorkflowHandle[StudyWorkflow, None]:
    from temporalio.common import WorkflowIDConflictPolicy

    from grassmoth_server.actions.temporal.client import STUDY_TASK_QUEUE, study_workflow_id
    from grassmoth_server.actions.temporal.study.messages import StudyInit
    from grassmoth_server.actions.temporal.study.workflow import StudyWorkflow

    return await (await clients.get()).start_workflow(
        StudyWorkflow.run,
        StudyInit(
            workspace_id=workspace_id, initial_seq=StudyRepository(workspace_id).latest_seq()
        ),
        id=study_workflow_id(workspace_id),
        task_queue=STUDY_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _running_actions(
    workspace_id: str, clients: TemporalClientProvider
) -> tuple[RunningAction, ...]:
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from grassmoth_server.actions.temporal.client import RUNNING_ACTION_MEMO, study_workflow_id

    try:
        description = (
            await (await clients.get())
            .get_workflow_handle(study_workflow_id(workspace_id))
            .describe()
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return ()
        raise
    if description.status != WorkflowExecutionStatus.RUNNING:
        return ()
    running = await description.memo_value(
        RUNNING_ACTION_MEMO, (), type_hint=tuple[RunningCall, ...]
    )
    logs = await asyncio.gather(
        *(
            asyncio.to_thread(
                collect_call_log,
                workspace_id,
                seq=call.seq,
                attempt_id=call.attempt_id,
                messages=call.messages,
                at=datetime.now(UTC),
            )
            for call in running
        )
    )
    return tuple(
        RunningAction(
            call_id=call_identity(call.request),
            action=call.request.action,
            request=call.request,
            messages=log.messages,
        )
        for call, log in zip(running, logs, strict=True)
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


def _saved_response(workspace_id: str, revision: StudyRevision) -> ActionResult:
    return ActionResult(payload=completed_call_msgpack(workspace_id, revision))


async def submit_action(
    workspace_id: str,
    body: ActionInput,
    clients: TemporalClientProvider,
) -> ActionResult | StudyError:
    """Resolve and deduplicate a call; saved failures and successes share the same identity."""
    from temporalio.client import WorkflowUpdateStage

    from grassmoth_server.actions.temporal.study.messages import ActionRequest

    parsed = parse_workspace_id(workspace_id)
    if isinstance(parsed, StudyError):
        return parsed
    workspace_id = parsed
    if not actions_enabled() and not storage.exists(
        storage.join(data_module.study_dir(workspace_id), "history.git")
    ):
        return _read_only()
    repository = StudyRepository(workspace_id)
    try:
        request, contents = await asyncio.to_thread(resolve_action_inputs, repository, body)
    except StudyLookupError as exc:
        return StudyError(code="invalid_inputs", detail=str(exc))
    saved = await asyncio.to_thread(repository.saved_call, request)
    if saved is not None:
        return await asyncio.to_thread(_saved_response, workspace_id, saved)
    if not actions_enabled():
        return _read_only()
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


async def _progress_response(workspace_id: str, progress: CallProgress) -> ActionResult:
    if isinstance(progress, PendingCall):
        return ActionResult.from_poll(
            RunningPoll(call_id=call_identity(progress.request), action=progress.request.action)
        )
    if isinstance(progress, RunningCall):
        log = await asyncio.to_thread(
            collect_call_log,
            workspace_id,
            seq=progress.seq,
            attempt_id=progress.attempt_id,
            messages=progress.messages,
            at=datetime.now(UTC),
        )
        return ActionResult.from_poll(
            RunningPoll(
                call_id=call_identity(progress.request),
                action=progress.request.action,
                messages=log.messages,
            )
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
    return ActionResult.from_poll(
        FailedPoll(
            call_id=call_identity(attempt.request),
            action=attempt.action,
            commit_id=None,
            messages=log.messages,
        )
    )


async def poll_action(
    workspace_id: str,
    action: ActionId,
    call_id: CallId,
    clients: TemporalClientProvider,
) -> ActionResult | StudyError:
    """Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures."""
    from temporalio.service import RPCError, RPCStatusCode

    from grassmoth_server.actions.temporal.client import study_workflow_id

    parsed = parse_workspace_id(workspace_id)
    if isinstance(parsed, StudyError):
        return parsed
    workspace_id = parsed
    if storage.exists(storage.join(data_module.study_dir(workspace_id), "history.git")):
        saved = await asyncio.to_thread(StudyRepository(workspace_id).call, call_id)
        if saved is not None:
            if saved.record.attempt.action != action:
                return StudyError(code="unknown_call", detail="Unknown action call")
            return await asyncio.to_thread(_saved_response, workspace_id, saved)
    if not actions_enabled():
        return StudyError(code="unknown_call", detail="Unknown action call")
    try:
        handle = (await clients.get()).get_workflow_handle(study_workflow_id(workspace_id))
        progress = cast(
            "CallProgress | None",
            await handle.query("call_progress", call_id, result_type=_CALL_PROGRESS_TYPE),
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return StudyError(code="unknown_call", detail="Unknown action call")
        raise
    if progress is None:
        return StudyError(code="unknown_call", detail="Unknown action call")
    actual_action = (
        progress.attempt.action if isinstance(progress, CompletedCall) else progress.request.action
    )
    if actual_action != action:
        return StudyError(code="unknown_call", detail="Unknown action call")
    return await _progress_response(workspace_id, progress)


async def get_timeline(
    workspace_id: str, clients: TemporalClientProvider
) -> TimelineResponse | StudyError:
    """Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id."""
    parsed = parse_workspace_id(workspace_id)
    if isinstance(parsed, StudyError):
        return parsed
    workspace_id = parsed
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
        running=await _running_actions(workspace_id, clients) if actions_enabled() else (),
    )
