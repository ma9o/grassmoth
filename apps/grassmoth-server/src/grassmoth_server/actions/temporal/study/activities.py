"""The study's execution and publication I/O, outside the workflow sandbox."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from temporalio import activity
from temporalio.exceptions import ApplicationError

from grassmoth.public import AggregatedCompileError, IncompleteModelError
from grassmoth_server.actions.call_logs import collect_call_snapshot
from grassmoth_server.actions.contracts.requests import PrepareDataRequest, call_identity
from grassmoth_server.actions.edit_model import edit_model
from grassmoth_server.actions.errors import ActionExecutionError
from grassmoth_server.actions.input_resolution import input_state
from grassmoth_server.actions.messages import (
    comparison_messages,
    edit_messages,
    fit_check_messages,
    inference_messages,
    preparation_messages,
    simulation_messages,
)
from grassmoth_server.actions.runners import run_action
from grassmoth_server.actions.temporal.extraction.activities import MEASUREMENT_ACTIVITIES
from grassmoth_server.actions.temporal.extraction.source_data import read_source_data_activity
from grassmoth_server.actions.temporal.llm.activities import LLM_SUBROUTINE_ACTIVITIES
from grassmoth_server.actions.temporal.study.messages import (
    ActionInput,
    AttemptPublication,
    ChecksResult,
    EditModelActivityInput,
    EvaluateChecksInput,
    ReadInputsInput,
)
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import (
    ActionAttempt,
    ActionBase,
    Applied,
    EditAttempt,
    Rejected,
    StudyRevision,
    action_attempt,
)
from grassmoth_server.study.state import StudyState
from grassmoth_server.study.sweep import collect_completed_runs


@activity.defn
async def run_action_activity(activity_input: ActionInput) -> ActionAttempt:
    """Run a scientific action and translate expected input failures into rejected attempts.

    Args:
        activity_input: Pinned request, selected study state, and destination workspace.

    Returns:
        An applied attempt or an input/scientific rejection suitable for publication.

    Raises:
        ApplicationError: Execution failed with retained diagnostics; Temporal must
            not retry this scientific failure.
    """
    try:
        result = await run_action(activity_input.workspace_id, activity_input.request)
    except StudyLookupError as exc:
        return action_attempt(
            activity_input.request,
            Rejected(code="INPUT_UNAVAILABLE", subject="inputs", detail=str(exc)),
        )
    except (IncompleteModelError, AggregatedCompileError) as exc:
        return action_attempt(
            activity_input.request,
            Rejected(code="model_execution", subject="model", detail=str(exc)),
        )
    except ActionExecutionError as exc:
        raise ApplicationError(
            str(exc), exc.diagnostics, type=type(exc).__name__, non_retryable=True
        ) from exc
    return action_attempt(activity_input.request, result)


@activity.defn
async def edit_model_activity(activity_input: EditModelActivityInput) -> EditAttempt:
    """Execute a model edit and wrap its applied or rejected outcome as an edit attempt."""
    result = edit_model(activity_input.workspace_id, activity_input.request)
    return EditAttempt(request=activity_input.request, outcome=result, action="edit_model")


@activity.defn
async def evaluate_model_checks_activity(activity_input: EvaluateChecksInput) -> ChecksResult:
    """Read the action's complete checks and produce its completion messages."""
    from grassmoth.public import EditModelOutput, FitOutput
    from grassmoth_server.actions.contracts.requests import FitRequest

    store = ArtifactStore(activity_input.workspace_id)
    if isinstance(activity_input.request, FitRequest):
        fit = await asyncio.to_thread(
            lambda: store.read_result(activity_input.applied.result, FitOutput)
        )
        at = datetime.now(UTC)
        return ChecksResult(
            reports={},
            messages=(*fit_check_messages(fit.checks, at), *inference_messages(fit.inference, at)),
        )
    if isinstance(activity_input.request, PrepareDataRequest):
        raise TypeError("Model checks require an edit or fit request")
    result = await asyncio.to_thread(
        lambda: store.read_result(activity_input.applied.result, EditModelOutput)
    )
    return ChecksResult(
        reports={},
        messages=edit_messages(
            result.checks, result.identification, result.pruning, datetime.now(UTC)
        ),
    )


@activity.defn
async def evaluate_data_checks_activity(activity_input: EvaluateChecksInput) -> ChecksResult:
    """Read the prepared result and produce its completion messages."""
    from grassmoth.public import PrepareDataOutput

    store = ArtifactStore(activity_input.workspace_id)
    result = await asyncio.to_thread(
        lambda: store.read_result(activity_input.applied.result, PrepareDataOutput)
    )
    return ChecksResult(
        reports={},
        messages=preparation_messages(result.profile, result.extraction, datetime.now(UTC)),
    )


@activity.defn
async def read_inputs_activity(activity_input: ReadInputsInput) -> ActionBase:
    """Read pinned input state or locate an already saved identical call."""
    repository = StudyRepository(activity_input.workspace_id)
    saved = repository.saved_call(activity_input.request)
    return ActionBase(
        state=StudyState()
        if saved is not None
        else input_state(repository, activity_input.request),
        saved=saved,
    )


@activity.defn
async def journal_activity(activity_input: AttemptPublication) -> StudyRevision:
    """Publish the owned record and traces; replay retries use its persisted identity."""
    return await asyncio.to_thread(_journal_attempt, activity_input)


def _journal_attempt(activity_input: AttemptPublication) -> StudyRevision:
    """Read large saved reports and publish their journal entry outside the event loop."""
    journal = StudyRepository(activity_input.workspace_id)
    request = activity_input.record.attempt.request
    assert request is not None
    identity = call_identity(request)
    existing = journal.call(identity)
    if existing is not None:
        return existing
    record = activity_input.record
    messages = activity_input.messages
    if isinstance(record.attempt.outcome, Applied) and record.attempt.action in {
        "simulate",
        "data_diff",
    }:
        from grassmoth.public import DataDiffOutput, SimulateOutput
        from grassmoth_server.study.queries.results import read_result

        store = ArtifactStore(activity_input.workspace_id)
        result = read_result(store, record.attempt.action, record.attempt.outcome.result)
        at = datetime.fromisoformat(record.ts)
        announcements = (
            simulation_messages(result.report, at)
            if isinstance(result, SimulateOutput)
            else comparison_messages(result.report, at)
            if isinstance(result, DataDiffOutput)
            else ()
        )
        messages = (*messages, *announcements)
    snapshot = collect_call_snapshot(activity_input.workspace_id, identity)
    return journal.append(record, messages=messages, snapshot=snapshot)


@activity.defn
async def collect_completed_runs_activity(workspace_id: str) -> None:
    """Remove scratch run directories whose completed attempts have been retained in history."""
    collect_completed_runs(workspace_id)


ALL_ACTIVITIES = [
    run_action_activity,
    edit_model_activity,
    journal_activity,
    read_inputs_activity,
    collect_completed_runs_activity,
    read_source_data_activity,
    *MEASUREMENT_ACTIVITIES,
    *LLM_SUBROUTINE_ACTIVITIES,
]
