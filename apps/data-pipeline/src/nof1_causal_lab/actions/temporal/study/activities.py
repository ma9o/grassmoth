"""The study's execution and publication I/O, outside the workflow sandbox."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.contracts.effects import ActionReportName
from nof1_causal_lab.actions.contracts.requests import PrepareDataRequest
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.edit_question import edit_question
from nof1_causal_lab.actions.errors import ActionExecutionError
from nof1_causal_lab.actions.input_resolution import input_state
from nof1_causal_lab.actions.messages import (
    comparison_messages,
    edit_messages,
    fit_check_messages,
    inference_messages,
    preparation_messages,
    simulation_messages,
)
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.actions.temporal.extraction.activities import MEASUREMENT_ACTIVITIES
from nof1_causal_lab.actions.temporal.extraction.source_data import read_source_data_activity
from nof1_causal_lab.actions.temporal.llm.activities import LLM_SUBROUTINE_ACTIVITIES
from nof1_causal_lab.actions.temporal.study.messages import (
    ActionInput,
    AttemptPublication,
    ChecksResult,
    CompleteResultInput,
    EditModelActivityInput,
    EditQuestionActivityInput,
    EvaluateChecksInput,
    ReadInputsInput,
)
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.storage.traces import collect_run_traces
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import (
    ActionAttempt,
    ActionBase,
    Applied,
    Rejected,
    StagedActionAttempt,
    StagedEditAttempt,
    StagedEditQuestionAttempt,
    StudyRevision,
    applied_attempt,
    failed_staged_attempt,
)
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.sweep import collect_completed_runs


@activity.defn
async def run_action_activity(activity_input: ActionInput) -> StagedActionAttempt:
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
        return failed_staged_attempt(
            activity_input.request,
            Rejected(code="INPUT_UNAVAILABLE", subject="inputs", detail=str(exc)),
        )
    except (IncompleteModelError, AggregatedCompileError) as exc:
        return failed_staged_attempt(
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
async def edit_model_activity(activity_input: EditModelActivityInput) -> StagedEditAttempt:
    """Execute a model edit and wrap its applied or rejected outcome as an edit attempt."""
    result = edit_model(activity_input.workspace_id, activity_input.request)
    return StagedEditAttempt(request=activity_input.request, outcome=result, action="edit_model")


@activity.defn
async def edit_question_activity(
    activity_input: EditQuestionActivityInput,
) -> StagedEditQuestionAttempt:
    """Save the question and return the attempt that owns its publication effect."""
    result = edit_question(activity_input.workspace_id, activity_input.request)
    return StagedEditQuestionAttempt(
        request=activity_input.request, outcome=result, action="edit_question"
    )


@activity.defn
async def evaluate_model_checks_activity(
    activity_input: EvaluateChecksInput[ModelEditResult | None],
) -> ChecksResult:
    """Run model checks off the event loop and persist their reports.

    Args:
        activity_input: Successful edit or fit, its request, and selected input state.

    Returns:
        References to the retained check reports and their completion messages.

    Raises:
        TypeError: A preparation request is incorrectly routed to model checks.
        AssertionError: An edit reached evaluation without its pruning result.
    """
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks

    if isinstance(activity_input.request, PrepareDataRequest):
        raise TypeError("Model checks require an edit or fit request")
    action = activity_input.request.action
    evaluated = await asyncio.to_thread(
        lambda: evaluate_model_checks(
            activity_input.workspace_id,
            activity_input.state,
            activity_input.applied,
            action=action,
        )
    )
    store = ArtifactStore(activity_input.workspace_id)
    from nof1_causal_lab.artifacts.reports.inference import FitCheckReport

    if isinstance(evaluated, FitCheckReport):
        return ChecksResult(
            reports={"checks": store.write_report(evaluated)},
            messages=fit_check_messages(evaluated, datetime.now(UTC)),
        )
    checks, identification = evaluated
    assert isinstance(activity_input.applied.result, ModelEditResult)
    reports: dict[ActionReportName, GitOid] = {
        "checks": store.write_report(checks),
        "identification": store.write_report(identification),
    }
    return ChecksResult(
        reports=reports,
        messages=edit_messages(
            checks, identification, activity_input.applied.result, datetime.now(UTC)
        ),
    )


@activity.defn
async def evaluate_data_checks_activity(
    activity_input: EvaluateChecksInput[DataPreparationResult],
) -> ChecksResult:
    """Persist the prepared panel's data profile and produce its completion messages."""
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks

    profile = await asyncio.to_thread(
        evaluate_data_checks,
        activity_input.workspace_id,
        activity_input.state,
        activity_input.applied,
    )
    return ChecksResult(
        reports={"data-profile": ArtifactStore(activity_input.workspace_id).write_report(profile)},
        messages=preparation_messages(profile, activity_input.applied.result, datetime.now(UTC)),
    )


@activity.defn
async def read_inputs_activity(activity_input: ReadInputsInput) -> ActionBase:
    """Capture the publication head and input state, or locate an already saved identical call."""
    repository = StudyRepository(activity_input.workspace_id)
    saved = repository.saved_call(activity_input.request)
    return ActionBase(
        commit_id=repository.head(),
        state=StudyState()
        if saved is not None
        else input_state(repository, activity_input.request),
        saved=saved,
    )


@activity.defn
async def complete_result_activity(activity_input: CompleteResultInput) -> ActionAttempt:
    """Save the complete typed result and return its small publication reference."""
    from nof1_causal_lab.actions.output_builder import complete_attempt

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
        return journal.append(existing.record, parent_id=activity_input.parent_id)
    record = activity_input.record
    messages = record.messages
    if isinstance(record.attempt.outcome, Applied):
        from nof1_causal_lab.actions.contracts.bodies import (
            DataDiffOutput,
            FitOutput,
            SimulateOutput,
        )
        from nof1_causal_lab.study.queries.results import read_result

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

    from nof1_causal_lab.actions.contracts.progress import ProgressEvent
    from nof1_causal_lab.actions.progress import read_events

    if record.attempt_id is not None:
        logs["progress.json"] = TypeAdapter(tuple[ProgressEvent, ...]).dump_json(
            tuple(read_events(activity_input.workspace_id, record.attempt_id))
        )
    return journal.append(record, parent_id=activity_input.parent_id, logs=logs)


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
