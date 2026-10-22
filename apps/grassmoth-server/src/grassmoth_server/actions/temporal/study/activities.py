"""The study's execution and publication I/O, outside the workflow sandbox."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from temporalio import activity
from temporalio.exceptions import ApplicationError

from grassmoth.public import AggregatedCompileError, IncompleteModelError
from grassmoth_server.actions.contracts.requests import PrepareDataRequest
from grassmoth_server.actions.edit_model import edit_model
from grassmoth_server.actions.edit_question import edit_question
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
    CompleteResultInput,
    EditModelActivityInput,
    EditQuestionActivityInput,
    EvaluateChecksInput,
    ReadInputsInput,
)
from grassmoth_server.storage.traces import collect_run_traces
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import (
    ActionAttempt,
    ActionBase,
    Applied,
    EditAttempt,
    EditQuestionAttempt,
    Rejected,
    StudyRevision,
    applied_attempt,
    failed_attempt,
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
        result = await run_action(
            activity_input.workspace_id, activity_input.request, activity_input.state
        )
    except StudyLookupError as exc:
        return failed_attempt(
            activity_input.request,
            Rejected(code="INPUT_UNAVAILABLE", subject="inputs", detail=str(exc)),
        )
    except (IncompleteModelError, AggregatedCompileError) as exc:
        return failed_attempt(
            activity_input.request,
            Rejected(code="model_execution", subject="model", detail=str(exc)),
        )
    except ActionExecutionError as exc:
        raise ApplicationError(
            str(exc), exc.diagnostics, type=type(exc).__name__, non_retryable=True
        ) from exc
    return applied_attempt(
        activity_input.request, Applied(result=result.result, effects=result.effects)
    )


@activity.defn
async def edit_model_activity(activity_input: EditModelActivityInput) -> EditAttempt:
    """Execute a model edit and wrap its applied or rejected outcome as an edit attempt."""
    result = edit_model(activity_input.workspace_id, activity_input.request)
    return EditAttempt(request=activity_input.request, outcome=result, action="edit_model")


@activity.defn
async def edit_question_activity(
    activity_input: EditQuestionActivityInput,
) -> EditQuestionAttempt:
    """Save the question and return the attempt that owns its publication effect."""
    result = edit_question(activity_input.workspace_id, activity_input.request)
    return EditQuestionAttempt(
        request=activity_input.request, outcome=result, action="edit_question"
    )


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
        return ChecksResult(reports={}, messages=fit_check_messages(fit.checks, datetime.now(UTC)))
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
async def complete_result_activity(activity_input: CompleteResultInput) -> ActionAttempt:
    """Save the complete typed result and return its small publication reference."""
    from grassmoth_server.actions.output_builder import complete_attempt

    return await asyncio.to_thread(
        complete_attempt, activity_input.workspace_id, activity_input.attempt
    )


@activity.defn
async def journal_activity(activity_input: AttemptPublication) -> StudyRevision:
    """Publish the owned record and traces; replay retries use its persisted identity."""
    return await asyncio.to_thread(_journal_attempt, activity_input)


def _journal_attempt(activity_input: AttemptPublication) -> StudyRevision:
    """Read large saved reports and publish their journal entry outside the event loop."""
    journal = StudyRepository(activity_input.workspace_id)
    existing = journal.read_attempt(activity_input.record.seq)
    if existing is not None:
        return existing
    record = activity_input.record
    messages = record.messages
    if isinstance(record.attempt.outcome, Applied):
        from grassmoth.public import DataDiffOutput, FitOutput, SimulateOutput
        from grassmoth_server.study.queries.results import read_result

        result = read_result(
            ArtifactStore(activity_input.workspace_id),
            record.attempt.action,
            record.attempt.outcome.result,
        )
        if messages:
            announcements = (
                inference_messages(result.inference, messages[-1].timestamp)
                if isinstance(result, FitOutput)
                else simulation_messages(result.report, messages[-1].timestamp)
                if isinstance(result, SimulateOutput)
                else comparison_messages(result.report, messages[-1].timestamp)
                if isinstance(result, DataDiffOutput)
                else ()
            )
            messages = messages[:-1] + announcements + messages[-1:]
    logs = collect_run_traces(activity_input.workspace_id, record.seq)
    record = record.with_logs(
        messages=messages,
        trace_ids=tuple(path.removeprefix("traces/").removesuffix(".json") for path in logs),
    )
    from pydantic import TypeAdapter

    from grassmoth_server.actions.contracts.progress import ProgressEvent
    from grassmoth_server.actions.progress import read_events

    if record.attempt_id is not None:
        logs["progress.json"] = TypeAdapter(tuple[ProgressEvent, ...]).dump_json(
            tuple(read_events(activity_input.workspace_id, record.attempt_id))
        )
    return journal.append(record, logs=logs)


@activity.defn
async def collect_completed_runs_activity(workspace_id: str) -> None:
    """Remove scratch run directories whose completed attempts have been retained in history."""
    collect_completed_runs(workspace_id)


ALL_ACTIVITIES = [
    complete_result_activity,
    run_action_activity,
    edit_question_activity,
    edit_model_activity,
    journal_activity,
    read_inputs_activity,
    collect_completed_runs_activity,
    read_source_data_activity,
    *MEASUREMENT_ACTIVITIES,
    *LLM_SUBROUTINE_ACTIVITIES,
]
