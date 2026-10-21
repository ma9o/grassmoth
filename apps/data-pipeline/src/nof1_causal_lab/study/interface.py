"""Public study operations using the production Temporal lifecycle and retained results."""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from pydantic import TypeAdapter

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
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    call_identity,
)
from nof1_causal_lab.actions.contracts.timeline import (
    TimelineRecord,
    TimelineResponse,
    TimelineRevision,
)
from nof1_causal_lab.actions.input_resolution import resolve_action_inputs
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ActionId, CallId, GitOid
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.codec import pack_result, unpack_result
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.queries.action_outputs import completed_call_msgpack
from nof1_causal_lab.study.records import (
    Applied,
    Attempt,
    StudyRevision,
    record_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from temporalio.client import Client, WorkflowHandle

    from nof1_causal_lab.actions.temporal.study.workflow import StudyWorkflow


class StudyError(Value):
    """An expected rejection at the public study boundary."""

    code: Literal["invalid_request", "read_only", "invalid_inputs", "unknown_call"]
    detail: str


_POLL: TypeAdapter[ActionPoll] = TypeAdapter(ActionPoll)


class ActionResult(Value):
    """An intact action envelope; decoding is optional for callers forwarding saved bytes."""

    payload: bytes

    @classmethod
    def from_poll(cls, poll: ActionPoll) -> ActionResult:
        """Encode a live status using the same numerical codec as saved results."""
        return cls(payload=pack_result(poll))

    def decode(self) -> ActionPoll:
        """Read the typed result without executing or rebuilding scientific outputs."""
        return _POLL.validate_python(unpack_result(self.payload))


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
                from nof1_causal_lab.actions.temporal.client import connect_client

                self._client = await connect_client()
            return self._client


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


def _saved_response(workspace_id: str, revision: StudyRevision) -> ActionResult:
    return ActionResult(payload=completed_call_msgpack(workspace_id, revision))


async def submit_action(
    workspace_id: str,
    body: ActionInput,
    clients: TemporalClientProvider,
) -> ActionResult | StudyError:
    """Resolve and deduplicate a call; saved failures and successes share the same identity."""
    from temporalio.client import WorkflowUpdateStage

    from nof1_causal_lab.actions.temporal.study.messages import ActionRequest

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

    from nof1_causal_lab.actions.temporal.client import study_workflow_id

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
        running=await _running_action(workspace_id, clients) if actions_enabled() else None,
    )
