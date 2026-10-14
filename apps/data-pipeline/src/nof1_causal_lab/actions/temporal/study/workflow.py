"""Serialize scientific attempts and publish their correlated outcomes atomically."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ChildWorkflowError

from nof1_causal_lab.actions.contracts.effects import ActionEffects

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.bodies import PrepareDataInput
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.observations.preparation import FileSourceRef

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.contracts.call_state import (
        CallProgress,
        CompletedCall,
        PendingCall,
        RunningCall,
    )
    from nof1_causal_lab.actions.contracts.requests import (
        DataDiffRequest,
        EditModelRequest,
        EditQuestionRequest,
        FitRequest,
        ModelDiffRequest,
        PrepareDataRequest,
        call_identity,
    )
    from nof1_causal_lab.actions.temporal.activity_policy import temporal_failure
    from nof1_causal_lab.actions.temporal.client import MODEL_CHECKS_TASK_QUEUE, RUNNING_ACTION_MEMO
    from nof1_causal_lab.actions.temporal.extraction.messages import (
        MeasurementsWorkflowInput,
        ReadSourceDataInput,
    )
    from nof1_causal_lab.actions.temporal.extraction.source_data import read_source_data_activity
    from nof1_causal_lab.actions.temporal.study.activities import (
        complete_result_activity,
        evaluate_model_checks_activity,
        run_action_activity,
    )
    from nof1_causal_lab.actions.temporal.study.messages import (
        ActionInput,
        ActionRequest,
        AttemptPublication,
        ChecksResult,
        CompleteResultInput,
        EditModelActivityInput,
        EditQuestionActivityInput,
        EvaluateChecksInput,
        ReadInputsInput,
        StudyInit,
    )
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
    from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
    from nof1_causal_lab.study.records import (
        ActionAttempt,
        ActionBase,
        ActionMessage,
        Applied,
        AttemptRecord,
        Rejected,
        StagedEditAttempt,
        StagedEditQuestionAttempt,
        StagedPrepareAttempt,
        StudyRevision,
        failed_attempt,
    )
    from nof1_causal_lab.study.state import validate_lineage
_RUN_ACTION_TIMEOUT = timedelta(hours=4)
_WRITE_TIMEOUT = timedelta(minutes=5)
_CHECK_TIMEOUT = timedelta(hours=1)
_JOURNAL_TIMEOUT = timedelta(minutes=1)
_RUN_COLLECTION_TIMEOUT = timedelta(minutes=10)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
    non_retryable_error_types=[
        "ActionExecutionError",
        "IncompleteModelError",
        "ModelFitError",
        "StudyLookupError",
    ],
)
_JOURNAL_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)


@workflow.defn
class StudyWorkflow:
    """Workspace-scoped workflow serializing scientific calls and tracking their progress."""

    @workflow.init
    def __init__(self, init: StudyInit) -> None:
        """Bind the workspace and starting sequence, then initialize call tracking and serialization."""
        self._workspace_id = init.workspace_id
        self._seq = init.initial_seq
        self._closed = False
        self._lock = asyncio.Lock()
        self._attempts: dict[UUID, CallProgress] = {}
        self._calls: dict[str, UUID] = {}
        self._messages: tuple[ActionMessage, ...] = ()

    @workflow.run
    async def run(self, init: StudyInit) -> None:
        """Keep the initialized study workflow alive until its close signal arrives."""
        del init
        await workflow.wait_condition(lambda: self._closed)

    @workflow.update
    async def execute_action(self, request: ActionRequest) -> CallProgress:
        """Deduplicate identical calls and execute each new attempt under the workspace lock."""
        identity = call_identity(request.request)
        existing = self.call_progress(identity)
        if existing is not None:
            return existing
        self._calls[identity] = request.attempt_id
        self._attempts[request.attempt_id] = PendingCall(
            attempt_id=request.attempt_id, request=request.request
        )
        async with self._lock:
            try:
                await self._execute_action(request)
            except (ActivityError, ChildWorkflowError) as exc:
                failure = temporal_failure(exc)
                self._attempts[request.attempt_id] = CompletedCall(
                    attempt_id=request.attempt_id,
                    seq=self._seq,
                    commit_id=None,
                    attempt=failed_attempt(request.request, failure),
                    messages=(
                        *self._messages,
                        ActionMessage(
                            subject=request.request.action,
                            detail=failure.error_message,
                            timestamp=workflow.now(),
                            severity="error",
                            code=_error_label(failure.error_type),
                        ),
                    ),
                )
            finally:
                workflow.upsert_memo({RUNNING_ACTION_MEMO: None})
        return self._attempts[request.attempt_id]

    async def _execute_action(self, request: ActionRequest) -> None:
        action = request.request
        self._seq += 1
        self._messages = (
            ActionMessage(
                subject=request.request.action,
                detail=f"Started {action.action.replace('_', ' ')}.",
                timestamp=workflow.now(),
                severity="info",
                code=f"{action.action.upper()}_STARTED",
            ),
        )
        self._report_progress(request)
        try:
            base = await workflow.execute_activity(
                "read_inputs_activity",
                ReadInputsInput(workspace_id=self._workspace_id, request=action),
                result_type=ActionBase,
                start_to_close_timeout=_JOURNAL_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except ActivityError as exc:
            failure = temporal_failure(exc)
            outcome = (
                Rejected(code="INPUT_UNAVAILABLE", subject="inputs", detail=failure.error_message)
                if failure.error_type == "StudyLookupError"
                else failure
            )
            await self._journal(self._seq, request, None, failed_attempt(action, outcome))
            return
        if base.saved is not None:
            saved = base.saved
            self._attempts[request.attempt_id] = CompletedCall(
                attempt_id=request.attempt_id,
                seq=saved.record.seq,
                commit_id=saved.commit_id,
                attempt=saved.record.attempt,
                messages=saved.record.messages,
            )
            return
        seq = self._seq
        lineage = (
            validate_lineage(base.state, action.action)
            if not isinstance(action, (DataDiffRequest, ModelDiffRequest))
            else (None if base.state.has("question") else "Set the study question first")
        )
        rejection = (
            Rejected(code="INPUT_UNAVAILABLE", subject="inputs", detail=lineage)
            if lineage is not None
            else None
        )
        if rejection is not None:
            await self._journal(seq, request, base, failed_attempt(action, rejection))
            return
        try:
            if isinstance(action, EditQuestionRequest):
                attempt = await workflow.execute_activity(
                    "edit_question_activity",
                    EditQuestionActivityInput(workspace_id=self._workspace_id, request=action),
                    result_type=StagedEditQuestionAttempt,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif isinstance(action, EditModelRequest):
                attempt = await workflow.execute_activity(
                    "edit_model_activity",
                    EditModelActivityInput(workspace_id=self._workspace_id, request=action),
                    result_type=StagedEditAttempt,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif isinstance(action, PrepareDataRequest):
                attempt = await self._prepare_files(seq, request, action.input)
            else:
                attempt = await workflow.execute_activity(
                    run_action_activity,
                    ActionInput(workspace_id=self._workspace_id, request=action, state=base.state),
                    start_to_close_timeout=_RUN_ACTION_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            if isinstance(
                action, (EditModelRequest, FitRequest, PrepareDataRequest)
            ) and isinstance(attempt.outcome, Applied):
                result = attempt.outcome
                self._messages = (
                    *self._messages,
                    ActionMessage(
                        subject=request.request.action,
                        detail="Evaluating data findings."
                        if attempt.action == "prepare_data"
                        else "Evaluating model findings.",
                        timestamp=workflow.now(),
                        severity="info",
                        code="DATA_CHECKS_STARTED"
                        if attempt.action == "prepare_data"
                        else "MODEL_CHECKS_STARTED",
                    ),
                )
                self._report_progress(request)
                if isinstance(action, PrepareDataRequest):
                    assert isinstance(result.result, DataPreparationResult)
                    evaluated = await workflow.execute_activity(
                        "evaluate_data_checks_activity",
                        EvaluateChecksInput[DataPreparationResult](
                            workspace_id=self._workspace_id,
                            state=base.state,
                            applied=Applied(result=result.result, effects=result.effects),
                            request=action,
                        ),
                        result_type=ChecksResult,
                        task_queue=MODEL_CHECKS_TASK_QUEUE,
                        start_to_close_timeout=_CHECK_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                else:
                    assert result.result is None or isinstance(result.result, ModelEditResult)
                    evaluated = await workflow.execute_activity(
                        evaluate_model_checks_activity,
                        EvaluateChecksInput[ModelEditResult | None](
                            workspace_id=self._workspace_id,
                            state=base.state,
                            applied=Applied(result=result.result, effects=result.effects),
                            request=action,
                        ),
                        task_queue=MODEL_CHECKS_TASK_QUEUE,
                        start_to_close_timeout=_CHECK_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                self._messages = (*self._messages, *evaluated.messages)
                attempt = attempt.revised(
                    outcome=result.revised(
                        effects=result.effects.revised(
                            reports={**result.effects.reports, **evaluated.reports}
                        )
                    )
                )
            completed = await workflow.execute_activity(
                complete_result_activity,
                CompleteResultInput(workspace_id=self._workspace_id, attempt=attempt),
                start_to_close_timeout=_CHECK_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except (ActivityError, ChildWorkflowError) as exc:
            completed = failed_attempt(action, temporal_failure(exc))
        await self._journal(seq, request, base, completed)

    async def _prepare_files(
        self, seq: int, request: ActionRequest, preparation: PrepareDataInput[GitOid, FileSourceRef]
    ) -> StagedPrepareAttempt:
        assert isinstance(request.request, PrepareDataRequest)
        memo = {"workspace_id": self._workspace_id, "seq": seq, "action": "prepare_data"}
        raw_data = await workflow.execute_activity(
            read_source_data_activity,
            ReadSourceDataInput(workspace_id=self._workspace_id, source=preparation.source),
            start_to_close_timeout=_WRITE_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            summary="Read source tables",
        )
        if isinstance(raw_data, Rejected):
            return StagedPrepareAttempt(
                action="prepare_data", request=request.request, outcome=raw_data
            )
        extracted = await workflow.execute_child_workflow(
            "MeasurementsWorkflow",
            MeasurementsWorkflowInput(
                workspace_id=self._workspace_id,
                seq=seq,
                attempt_id=request.attempt_id,
                raw_data_revision=raw_data.revision,
                preparation=preparation,
            ),
            id=f"measurements-{self._workspace_id}-{seq:06d}",
            result_type=Applied[DataPreparationResult],
            execution_timeout=_RUN_ACTION_TIMEOUT,
            static_summary="Prepare observations",
            memo=memo,
        )
        return StagedPrepareAttempt(
            action="prepare_data",
            request=request.request,
            outcome=Applied(
                result=extracted.result,
                effects=ActionEffects(produced=(raw_data, *extracted.effects.produced)),
            ),
        )

    @workflow.signal
    def close(self) -> None:
        """Signal that the study workflow may finish waiting and close."""
        self._closed = True

    @workflow.query
    def action_progress(self, attempt_id: UUID) -> CallProgress | None:
        """Return the latest state for an attempt, or ``None`` if this workflow has not seen it."""
        return self._attempts.get(attempt_id)

    @workflow.query
    def call_progress(self, identity: str) -> CallProgress | None:
        """Resolve a call identity to its latest attempt state, or ``None`` for an unknown call."""
        attempt_id = self._calls.get(identity)
        return self._attempts.get(attempt_id) if attempt_id is not None else None

    def _report_progress(self, request: ActionRequest) -> None:
        self._attempts[request.attempt_id] = RunningCall(
            attempt_id=request.attempt_id,
            seq=self._seq,
            request=request.request,
            messages=self._messages,
        )
        workflow.upsert_memo(
            {
                RUNNING_ACTION_MEMO: RunningCall(
                    attempt_id=request.attempt_id,
                    seq=self._seq,
                    request=request.request,
                    messages=self._messages,
                )
            }
        )

    async def _journal(
        self,
        seq: int,
        request: ActionRequest,
        base: ActionBase | None,
        attempt: ActionAttempt,
    ) -> None:
        outcome = attempt.outcome
        label = (
            "ACTION_COMPLETED"
            if isinstance(outcome, Applied)
            else outcome.code
            if isinstance(outcome, Rejected)
            else _error_label(outcome.error_type)
        )
        messages = (
            *self._messages,
            ActionMessage(
                subject=outcome.subject
                if isinstance(outcome, Rejected)
                else request.request.action,
                detail=outcome.detail
                if isinstance(outcome, Rejected)
                else "Action completed."
                if isinstance(outcome, Applied)
                else outcome.error_message,
                timestamp=workflow.now(),
                severity="info" if isinstance(outcome, Applied) else "error",
                code=label,
            ),
        )
        revision = await workflow.execute_activity(
            "journal_activity",
            AttemptPublication(
                workspace_id=self._workspace_id,
                parent_id=base.commit_id if base is not None else None,
                record=AttemptRecord(
                    seq=seq,
                    attempt_id=request.attempt_id,
                    ts=workflow.now().isoformat(),
                    attempt=attempt,
                    messages=messages,
                ),
            ),
            result_type=StudyRevision,
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        self._attempts[request.attempt_id] = CompletedCall(
            attempt_id=request.attempt_id,
            seq=revision.record.seq,
            commit_id=revision.commit_id,
            attempt=revision.record.attempt,
            messages=revision.record.messages,
        )
        try:
            await workflow.execute_activity(
                "collect_completed_runs_activity",
                self._workspace_id,
                start_to_close_timeout=_RUN_COLLECTION_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except ActivityError as exc:
            workflow.logger.warning("run scratch collection failed after seq %d: %s", seq, exc)


def _error_label(error_type: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])", "_", error_type)
    return "ACTION_FAILED_" + re.sub(r"[^A-Za-z0-9]+", "_", words).strip("_").upper()
