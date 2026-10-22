"""Tests for replay-safe study workflow helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
from temporalio.exceptions import ApplicationError, FailureError

from grassmoth_server.actions.contracts.progress import StepError, StepEvent
from grassmoth_server.actions.temporal.activity_policy import (
    EVENT_RETRY,
    EVENT_TIMEOUT,
    emit_progress,
    temporal_failure,
)
from grassmoth_server.actions.temporal.study.messages import ActionRequest
from grassmoth_server.actions.temporal.study.workflow import StudyWorkflow

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from grassmoth_server.actions.contracts.call_state import CallProgress

pytestmark = pytest.mark.contract


async def _execute_action(machine: StudyWorkflow, request: ActionRequest) -> CallProgress:
    return await cast("Awaitable[CallProgress]", StudyWorkflow.execute_action(machine, request))


@pytest.mark.parametrize("fails", [False, True])
def test_openrouter_transport_is_scoped_to_the_worker_lifetime(monkeypatch, fails):
    from grassmoth_server import config
    from grassmoth_server.actions.temporal import worker
    from grassmoth_server.llm import openrouter as openrouter_client

    temporal_client = Mock()
    transport = Mock()
    pending_finished = not fails

    def close_transport(*_args):
        assert pending_finished
        return False

    transport_scope = MagicMock(
        __aenter__=AsyncMock(return_value=transport),
        __aexit__=AsyncMock(side_effect=close_transport),
    )
    factory = Mock(return_value=transport_scope)
    monkeypatch.setattr(openrouter_client, "create_openrouter_client", factory)
    monkeypatch.setattr(config, "configure_jax_persistent_cache", Mock())
    monkeypatch.setattr(worker, "connect_client", AsyncMock(return_value=temporal_client))

    builders = {}
    for name in (
        "build_worker",
        "build_openrouter_worker",
        "build_model_checks_worker",
    ):
        run = AsyncMock(
            side_effect=RuntimeError("worker failed")
            if fails and name == "build_openrouter_worker"
            else None
        )
        builders[name] = Mock(return_value=Mock(run=run))
        monkeypatch.setattr(worker, name, builders[name])

    if fails:

        async def pending_worker():
            nonlocal pending_finished
            try:
                await asyncio.Event().wait()
            finally:
                pending_finished = True

        builders["build_worker"].return_value.run = pending_worker
        with pytest.raises(ExceptionGroup) as failure:
            asyncio.run(worker.run_worker())
        assert len(failure.value.exceptions) == 1
        assert str(failure.value.exceptions[0]) == "worker failed"
    else:
        asyncio.run(worker.run_worker())

    factory.assert_called_once_with()
    builders["build_openrouter_worker"].assert_called_once_with(temporal_client, transport)
    transport_scope.__aexit__.assert_awaited_once()


class _WrappedError(FailureError):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.__cause__ = cause


def test_temporal_failure_unwraps_application_error_and_copies_diagnostics() -> None:
    diagnostics = {"reason": "invalid"}
    error = _WrappedError(
        _WrappedError(
            ApplicationError(
                "model failed",
                diagnostics,
                type="ModelFitError",
                non_retryable=True,
            )
        )
    )

    failure = temporal_failure(error)
    diagnostics["local"] = True

    assert failure.error_type == "ModelFitError"
    assert failure.error_message == "model failed"
    assert failure.details == ('{"reason": "invalid"}',)


def test_temporal_failure_uses_deepest_untyped_cause() -> None:
    failure = temporal_failure(_WrappedError(ValueError("bad input")))
    assert failure.error_type == "ValueError"
    assert failure.error_message == "bad input"
    assert failure.details == ()


def test_emit_progress_uses_shared_activity_policy(monkeypatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def execute_activity(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    from grassmoth_server.actions.temporal import activity_policy as workflow_support

    monkeypatch.setattr(workflow_support.workflow, "execute_activity", execute_activity)
    step = StepEvent(
        attempt_id=uuid4(),
        status="failed",
        error=StepError(type="ValueError", message="bad input"),
    )

    asyncio.run(emit_progress("workspace-1", step))

    (activity_name, payload), kwargs = calls[0]
    assert activity_name == "emit_progress_event_activity"
    assert payload.workspace_id == "workspace-1"
    assert payload.event == step
    assert kwargs == {
        "start_to_close_timeout": EVENT_TIMEOUT,
        "retry_policy": EVENT_RETRY,
    }


@pytest.mark.parametrize(("kind", "attempts"), [("fit", 1), ("simulate", 3)])
def test_failed_fit_has_one_compute_attempt_and_only_journals_the_failure(
    monkeypatch, kind, attempts
):
    from datetime import UTC, datetime

    from temporalio.exceptions import ActivityError, RetryState

    from grassmoth_server.actions.temporal.study import workflow as study_workflow
    from grassmoth_server.actions.temporal.study.messages import StudyInit
    from grassmoth_server.study.records import ActionBase
    from grassmoth_server.study.state import StudyState
    from tests.support.git import git_oid

    payload = (
        {"target": {"kind": "continue", "fit_ref": git_oid(1)}}
        if kind == "fit"
        else {
            "dynamical_model_spec_ref": git_oid(1),
            "simulation": {"start": "2026-01-01", "horizon": "1d"},
        }
    )
    request = ActionRequest.model_validate({"request": {"action": kind, "input": payload}})
    base = ActionBase(state=StudyState())
    error = ActivityError(
        "provider timeout",
        scheduled_event_id=1,
        started_event_id=2,
        identity="worker",
        activity_type="run_action_activity",
        activity_id="2",
        retry_state=RetryState.MAXIMUM_ATTEMPTS_REACHED,
    )
    execute = AsyncMock(side_effect=[base, error])
    monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", Mock())
    monkeypatch.setattr(study_workflow, "validate_lineage", lambda *_: None)
    runner = study_workflow.StudyWorkflow(StudyInit(workspace_id="test"))
    journal = AsyncMock()
    monkeypatch.setattr(runner, "_journal", journal)
    asyncio.run(_execute_action(runner, request))
    assert (
        execute.call_count == 2
    )  # Read inputs, then compute; no checks or scientific publication.
    assert execute.call_args.kwargs["retry_policy"].maximum_attempts == attempts
    assert journal.await_args is not None
    assert journal.await_args.args[-1].outcome.status == "raised"


def test_continuation_calls_overlap_deduplicate_and_keep_their_own_progress(monkeypatch):
    from datetime import UTC, datetime

    from grassmoth_server.actions.contracts.call_state import CompletedCall, RunningCall
    from grassmoth_server.actions.temporal.client import RUNNING_ACTION_MEMO
    from grassmoth_server.actions.temporal.study import workflow as study_workflow
    from grassmoth_server.actions.temporal.study.messages import StudyInit
    from grassmoth_server.study.records import (
        ActionBase,
        FitAttempt,
        Raised,
        StudyRevision,
        failed_attempt,
    )
    from grassmoth_server.study.state import ArtifactRecord, StudyState
    from tests.support.git import git_oid

    async def scenario():
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        executions = []
        publications = []
        memos = []
        state = StudyState(
            current={
                "question": ArtifactRecord(
                    artifact_id="question",
                    revision=git_oid(1),
                )
            }
        )

        async def execute(name, payload, **_kwargs):
            name = name if isinstance(name, str) else name.__name__
            if name == "read_inputs_activity":
                await asyncio.sleep(0)
                return ActionBase(state=state)
            if name == "run_action_activity":
                index = payload.request.input.repeat
                executions.append(index)
                started[index].set()
                await release[index].wait()
                return FitAttempt(
                    action="fit",
                    request=payload.request,
                    outcome=Raised(error_type=f"Arm{index}", error_message=f"arm {index}"),
                )
            if name == "complete_result_activity":
                return failed_attempt(payload.attempt.request, payload.attempt.outcome)
            if name == "journal_activity":
                publications.append(payload.record)
                return StudyRevision(
                    record=payload.record,
                    parent_ids=(),
                    commit_id=git_oid(100 + payload.record.seq),
                )
            assert name == "collect_completed_runs_activity"
            return None

        monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
        monkeypatch.setattr(
            study_workflow.workflow, "now", lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        monkeypatch.setattr(study_workflow.workflow, "upsert_memo", memos.append)
        worker = study_workflow.StudyWorkflow(StudyInit(workspace_id="test"))
        requests = [
            ActionRequest.model_validate(
                {
                    "request": {
                        "action": "fit",
                        "input": {
                            "target": {"kind": "continue", "fit_ref": git_oid(7)},
                            "repeat": index,
                        },
                    }
                }
            )
            for index in range(2)
        ]
        async with asyncio.timeout(3):
            tasks = [asyncio.create_task(_execute_action(worker, request)) for request in requests]
            await asyncio.gather(*(event.wait() for event in started))
            assert len(memos[-1][RUNNING_ACTION_MEMO]) == 2
            duplicate = await _execute_action(worker, ActionRequest(request=requests[0].request))
            assert isinstance(duplicate, RunningCall)
            assert duplicate.seq == 1
            release[1].set()
            second = await tasks[1]
            assert isinstance(second, CompletedCall)
            assert second.seq == 2
            assert tuple(call.seq for call in memos[-1][RUNNING_ACTION_MEMO]) == (1,)
            release[0].set()
            first = await tasks[0]
            assert isinstance(first, CompletedCall)
            assert first.seq == 1
        assert executions == [0, 1]
        assert [record.seq for record in publications] == [2, 1]
        assert first.messages[-1].detail == "arm 0"
        assert second.messages[-1].detail == "arm 1"
        assert memos[-1][RUNNING_ACTION_MEMO] == ()

    asyncio.run(scenario())
