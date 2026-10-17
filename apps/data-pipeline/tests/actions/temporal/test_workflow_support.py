"""Tests for replay-safe study workflow helpers."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
from temporalio.exceptions import ApplicationError, FailureError

from nof1_causal_lab.actions.contracts.progress import StepError, StepEvent
from nof1_causal_lab.actions.temporal.activity_policy import (
    EVENT_RETRY,
    EVENT_TIMEOUT,
    emit_progress,
    temporal_failure,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("fails", [False, True])
def test_openrouter_transport_is_scoped_to_the_worker_lifetime(monkeypatch, fails):
    from nof1_causal_lab import config
    from nof1_causal_lab.actions.temporal import worker
    from nof1_causal_lab.llm import openrouter as openrouter_client

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
        "build_harness_worker",
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

    from nof1_causal_lab.actions.temporal import activity_policy as workflow_support

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

    from nof1_causal_lab.actions.temporal.study import workflow as study_workflow
    from nof1_causal_lab.actions.temporal.study.messages import ActionRequest, StudyInit
    from nof1_causal_lab.study.records import ActionBase
    from nof1_causal_lab.study.state import StudyState
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
    base = ActionBase(commit_id=git_oid(1), state=StudyState())
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
    monkeypatch.setattr(study_workflow, "validate_lineage", lambda *_: None)
    runner = study_workflow.StudyWorkflow(StudyInit(workspace_id="test"))
    journal = AsyncMock()
    monkeypatch.setattr(runner, "_journal", journal)
    monkeypatch.setattr(runner, "_report_progress", Mock())
    asyncio.run(runner._execute_action(request))
    assert (
        execute.call_count == 2
    )  # Read inputs, then compute; no checks or scientific publication.
    assert execute.call_args.kwargs["retry_policy"].maximum_attempts == attempts
    assert journal.await_args is not None
    assert journal.await_args.args[-1].outcome.status == "raised"
