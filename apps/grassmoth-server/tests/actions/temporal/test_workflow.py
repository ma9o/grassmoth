"""Content-named calls, atomic outcomes, and recovery from Git.

Computed CSV preparation makes no model calls. Model edits supply their definition directly;
fit and simulation readiness failures exercise the real execution boundary.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from grassmoth.actions.contracts.bodies import EditModelOutput, PrepareDataOutput
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth_server.actions.contracts.call_state import CallProgress, RunningCall
from grassmoth_server.actions.contracts.identity import RevisionSelector
from grassmoth_server.actions.contracts.inputs import (
    EditModelInput,
    EditQuestionInput,
    FitInput,
    ModelDiffInput,
    NewFitTarget,
    SimulateInput,
)
from grassmoth_server.actions.contracts.polling import (
    ActionPoll,
    ActionResult,
    RunningPoll,
    StudyError,
)
from grassmoth_server.actions.contracts.preparation import SourceFolder
from grassmoth_server.actions.contracts.requests import (
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from grassmoth_server.actions.contracts.timeline import TimelineResponse
from grassmoth_server.actions.temporal.study.workflow import StudyWorkflow
from grassmoth_server.llm.openrouter import create_openrouter_client
from grassmoth_server.study import calls as study_calls
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_model
from grassmoth_server.study.records import ActionMessage, Raised, Rejected
from grassmoth_test_support.helpers import graph_constructs

pytestmark = [pytest.mark.workflow, pytest.mark.timeout(60, method="thread")]
_QUESTION = "does exercise improve sleep?"
_PREPARATION: dict[str, Any] = {
    "dynamical_model_spec_ref": "latest",
    "source": "input",
    "extraction": {
        "indicator:sleep": {
            "kind": "computed",
            "how_to_measure": "Read the steps column",
            "source_columns": ["steps"],
        }
    },
}


async def _submit(workspace, request, clients) -> ActionPoll:
    response = await study_calls.submit_action(workspace, request, clients)
    assert isinstance(response, ActionResult), response
    return response.decode()


async def _timeline(workspace, clients) -> TimelineResponse:
    response = await study_calls.get_timeline(workspace, clients)
    assert isinstance(response, TimelineResponse), response
    return response


def _proposed_model() -> dict[str, Any]:
    return {
        "constructs": {
            "construct:sleep": {
                "name": "sleep",
                "description": "sleep quality",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
            "construct:unmeasured_outcome": {
                "name": "unmeasured_outcome",
                "description": "Downstream response outside the measured test states.",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
        },
        "edges": {
            "edge:test-outcome-0": {
                "cause": "construct:sleep",
                "effect": "construct:unmeasured_outcome",
                "description": "Test state affects an unmeasured downstream response",
            }
        },
    }


def _measured_model() -> dict[str, Any]:
    model = _proposed_model()
    model["measurement_clock"] = "1d"
    graph_constructs(model)[0]["indicators"] = {
        "indicator:sleep": {
            "observation": {
                "name": "sleep_steps_proxy",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            "construct_polarity": "positive",
        }
    }
    return model


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    import grassmoth_server.llm.openrouter as openrouter_client
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    input_root = Path(data_module.input_dir(workspace_id))
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "observations.csv").write_text(
        "timestamp,steps\n2026-01-01T08:00:00,1000\n2026-01-02T08:00:00,2000\n"
    )

    async def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("Computed CSV preparation must not call an LLM")

    monkeypatch.setattr(openrouter_client, "call_model", unexpected_model_call)
    return workspace_id


def test_study_workflow_journey(machine_env, monkeypatch):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from grassmoth_server.actions.temporal.client import pydantic_data_converter
        from grassmoth_server.actions.temporal.worker import (
            build_model_checks_worker,
            build_openrouter_worker,
            build_worker,
        )
        from grassmoth_server.study import calls as study_api

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                create_openrouter_client() as transport,
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client, transport),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("READ_ONLY_FACADE", "0")
                clients = study_calls.TemporalClientProvider()
                monkeypatch.setattr(clients, "get", AsyncMock(return_value=env.client))
                handle = await study_calls._study_handle(workspace_id, clients)

                async def execute(request):
                    response = await _submit(workspace_id, request, clients)
                    if isinstance(response, RunningPoll):
                        progress = await handle.query(
                            "call_progress",
                            response.call_id,
                            result_type=cast("type[CallProgress]", CallProgress),
                        )
                        await handle.get_update_handle(str(progress.attempt_id)).result()
                        response = await study_api.poll_action(
                            workspace_id, response.action, response.call_id, clients
                        )
                    polled = response.decode() if isinstance(response, ActionResult) else response
                    assert not isinstance(polled, StudyError)
                    assert polled.status != "running"
                    assert polled.commit_id is not None
                    record = StudyRepository(workspace_id).record(polled.commit_id)
                    assert polled.status == (
                        "success" if record.record.attempt.outcome.status == "applied" else "failed"
                    )
                    lifecycle = [
                        message for message in polled.messages if isinstance(message, ActionMessage)
                    ]
                    assert lifecycle[0].code == f"{request.action.upper()}_STARTED"
                    assert lifecycle[-1].severity == (
                        "info" if polled.status == "success" else "error"
                    )
                    return record

                def state(record=None):
                    repository = StudyRepository(workspace_id)
                    return repository.state(record.commit_id if record is not None else None)

                rejected = await execute(
                    EditModelRequest[GitOid](
                        input=EditModelInput[GitOid](
                            parent_ref=GitOid("0" * 40), dynamical_model_spec=DynamicalModelSpec()
                        )
                    )
                )
                assert isinstance(rejected.record.attempt.outcome, Rejected)
                assert rejected.record.attempt.outcome.code == "INPUT_UNAVAILABLE"
                assert "0" * 40 in rejected.record.attempt.outcome.detail

                question = QuestionSpec(text=_QUESTION, outcome="construct:unmeasured_outcome")
                root = await execute(
                    EditQuestionRequest(input=EditQuestionInput(question=question))
                )
                assert root.record.attempt.outcome.status == "applied"
                assert state(root).has("question")
                again = await execute(
                    EditQuestionRequest(input=EditQuestionInput(question=question))
                )
                assert again == root

                # Every edit defines the question's nodes; without them it is an invalid request.
                invalid = await execute(
                    EditModelRequest[GitOid](
                        input=EditModelInput[GitOid](
                            parent_ref=StudyRepository(workspace_id).question().revision,
                            dynamical_model_spec=DynamicalModelSpec(),
                        )
                    )
                )
                assert invalid.record.attempt.outcome.status == "rejected"
                assert invalid.record.attempt.outcome.code == "MODEL_SPEC_INVALID"
                edited = await execute(
                    EditModelRequest[GitOid](
                        input=EditModelInput[GitOid](
                            parent_ref=StudyRepository(workspace_id).question().revision,
                            dynamical_model_spec=DynamicalModelSpec.model_validate(
                                _measured_model()
                            ).materialized(),
                        )
                    )
                )
                assert edited.record.attempt.outcome.status == "applied", edited
                prepared = await execute(
                    PrepareDataRequest[RevisionSelector, SourceFolder](input=_PREPARATION)
                )
                assert prepared.record.attempt.outcome.status == "applied", prepared
                assert state(prepared).has("panel")
                assert set(state(prepared).current) == {"question", "model", "raw_data", "panel"}
                store = ArtifactStore(workspace_id)
                assert (
                    store.read_result(
                        prepared.record.attempt.outcome.result, PrepareDataOutput
                    ).profile
                    is not None
                )
                saved_model = store.read_result(
                    edited.record.attempt.outcome.result, EditModelOutput
                )
                assert saved_model.checks.specification
                model_revision = state(edited).current["model"].revision
                before = state().current

                # Both numerical actions reject incomplete definitions before any numerical work.
                for request in (
                    FitRequest[GitOid](
                        input=FitInput[GitOid](
                            target=NewFitTarget(
                                dynamical_model_spec_ref=model_revision,
                                data_ref=DataRef(
                                    revision=before["panel"].revision, replicate_index=0
                                ),
                            ),
                        )
                    ),
                    SimulateRequest[GitOid](
                        input=SimulateInput[GitOid](
                            simulation=SimulationSpec(start="2026-01-01", horizon="1d"),
                            dynamical_model_spec_ref=model_revision,
                        )
                    ),
                ):
                    raised = await execute(request)
                    outcome = raised.record.attempt.outcome
                    if isinstance(request, FitRequest):
                        assert isinstance(outcome, Raised), raised
                        assert outcome.error_type == "ModelFitError"
                    else:
                        assert isinstance(outcome, Rejected), raised
                        assert outcome.code == "model_execution"
                    assert state().current == before

                status = await _timeline(workspace_id, clients)
                previous_state = state()
                previous_run_id = handle.first_execution_run_id
                await handle.terminate()
                handle = await study_calls._study_handle(workspace_id, clients)
                assert handle.first_execution_run_id != previous_run_id
                recovered = await _timeline(workspace_id, clients)
                assert state() == previous_state
                assert recovered.attempts == status.attempts

                store = ArtifactStore(workspace_id)
                measured = read_model(store, model_revision)
                revised = measured.with_entities(
                    edges=(measured.edges[0].revised(description="Sleep shapes the response"),)
                )
                rewritten = await execute(
                    EditModelRequest[GitOid](
                        input=EditModelInput[GitOid](
                            parent_ref=model_revision,
                            dynamical_model_spec=revised,
                        )
                    )
                )
                assert rewritten.record.attempt.outcome.status == "applied", rewritten
                assert (
                    rewritten.record.seq == max(item.record.seq for item in recovered.attempts) + 1
                )
                records = StudyRepository(workspace_id).attempts()
                assert [record.record.attempt.outcome.status for record in records] == [
                    "rejected",
                    "applied",
                    "rejected",
                    "applied",
                    "applied",
                    "raised",
                    "rejected",
                    "applied",
                ]
                assert [record.record.attempt.action for record in records] == [
                    "edit_model",
                    "edit_question",
                    "edit_model",
                    "edit_model",
                    "prepare_data",
                    "fit",
                    "simulate",
                    "edit_model",
                ]
                assert records[3].record.attempt.action == "edit_model"
                assert records[3].record.attempt.request is not None
                assert (
                    records[3].record.attempt.request.input.dynamical_model_spec
                    == DynamicalModelSpec.model_validate(_measured_model()).materialized()
                )
                assert records[5].record.attempt.action == "fit"
                assert records[5].record.attempt.request is not None
                assert isinstance(records[5].record.attempt.request.input.target, NewFitTarget)
                assert (
                    records[5].record.attempt.request.input.target.dynamical_model_spec_ref
                    == model_revision
                )
                assert all("move" not in record.model_dump() for record in records)
                assert read_model(store, state(rewritten).current["model"].revision) == revised
                assert state(rewritten).current["question"] == state(root).current["question"]
                assert (
                    store.read_result(edited.record.attempt.outcome.result, EditModelOutput)
                    == saved_model
                )
                from grassmoth_server.actions.contracts.requests import ModelDiffRequest

                head = StudyRepository(workspace_id).head()
                comparison_request = ModelDiffRequest[GitOid](
                    reasoning="Check the revised edge before deciding which model to fit.",
                    input=ModelDiffInput[GitOid](
                        before_ref=state(edited).current["model"].revision,
                        after_ref=state(rewritten).current["model"].revision,
                    ),
                )
                compared = await execute(comparison_request)
                assert compared.record.attempt.action == "model_diff"
                assert compared.record.attempt.outcome.status == "applied"
                assert StudyRepository(workspace_id).head() == head
                saved = await _submit(
                    workspace_id,
                    ModelDiffRequest[RevisionSelector](
                        input=ModelDiffInput[RevisionSelector](
                            before_ref=comparison_request.input.before_ref,
                            after_ref=comparison_request.input.after_ref,
                        )
                    ),
                    clients,
                )
                completion = saved
                assert completion.status == "success"
                assert completion.action == "model_diff"
                assert completion.body.changes == revised.changes_from(measured)
                assert compared.record.attempt.request == comparison_request
                reverse = await execute(
                    ModelDiffRequest[GitOid](
                        input=ModelDiffInput[GitOid](
                            before_ref=state(rewritten).current["model"].revision,
                            after_ref=state(edited).current["model"].revision,
                        )
                    )
                )
                assert reverse.record.attempt.outcome.status == "applied"
                assert reverse.record.attempt.request.reasoning is None
                assert StudyRepository(workspace_id).head() == head
                timeline = await _timeline(workspace_id, clients)
                assert {
                    (link.source_seq, link.argument)
                    for link in timeline.dependencies
                    if link.seq == compared.record.seq
                } == {(edited.record.seq, "before"), (rewritten.record.seq, "after")}
                await handle.signal(StudyWorkflow.close)
                await handle.result()
        finally:
            await env.shutdown()

    asyncio.run(scenario())


def test_timeline_reports_only_attempts_a_live_workflow_executes(machine_env, monkeypatch):
    from grassmoth_server.actions.temporal.extraction import source_data as source_data_activity

    workspace_id = machine_env
    read_sources = source_data_activity.read_source_data_activity

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from grassmoth_server.actions.temporal.client import (
            RUNNING_ACTION_MEMO,
            pydantic_data_converter,
        )
        from grassmoth_server.actions.temporal.worker import (
            build_model_checks_worker,
            build_openrouter_worker,
            build_worker,
        )

        # Source reading waits here, so preparation stays in flight until released.
        release = asyncio.Event()

        from temporalio import activity

        from grassmoth_server.actions.temporal.extraction.messages import ReadSourceDataInput
        from grassmoth_server.actions.temporal.study import activities
        from grassmoth_server.study.state import ArtifactRecord

        @activity.defn(name="read_source_data_activity")
        async def held_read_sources(payload: ReadSourceDataInput) -> ArtifactRecord | Rejected:
            await release.wait()
            return await read_sources(payload)

        monkeypatch.setattr(
            activities,
            "ALL_ACTIVITIES",
            [
                held_read_sources if item is read_sources else item
                for item in activities.ALL_ACTIVITIES
            ],
        )
        from grassmoth_server.actions.temporal import worker

        monkeypatch.setattr(worker, "ALL_ACTIVITIES", activities.ALL_ACTIVITIES)
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                create_openrouter_client() as transport,
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client, transport),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("READ_ONLY_FACADE", "0")
                clients = study_calls.TemporalClientProvider()
                monkeypatch.setattr(clients, "get", AsyncMock(return_value=env.client))
                handle = await study_calls._study_handle(workspace_id, clients)
                created = await _submit(
                    workspace_id,
                    EditQuestionRequest(
                        input=EditQuestionInput(
                            question=QuestionSpec(
                                text=_QUESTION, outcome="construct:unmeasured_outcome"
                            )
                        )
                    ),
                    clients,
                )
                if isinstance(created, RunningPoll):
                    progress = await handle.query(
                        "call_progress",
                        created.call_id,
                        result_type=cast("type[CallProgress]", CallProgress),
                    )
                    await handle.get_update_handle(str(progress.attempt_id)).result()
                assert (await _timeline(workspace_id, clients)).running == ()
                model = await _submit(
                    workspace_id,
                    EditModelRequest[RevisionSelector](
                        input=EditModelInput[RevisionSelector](
                            parent_ref=StudyRepository(workspace_id).question().revision,
                            dynamical_model_spec=DynamicalModelSpec.model_validate(
                                _measured_model()
                            ).materialized(),
                        )
                    ),
                    clients,
                )
                if isinstance(model, RunningPoll):
                    progress = await handle.query(
                        "call_progress",
                        model.call_id,
                        result_type=cast("type[CallProgress]", CallProgress),
                    )
                    await handle.get_update_handle(str(progress.attempt_id)).result()

                preparing = await _submit(
                    workspace_id,
                    PrepareDataRequest[RevisionSelector, SourceFolder](input=_PREPARATION),
                    clients,
                )
                while not (
                    running_calls := await study_calls._running_actions(workspace_id, clients)
                ):
                    await asyncio.sleep(0.05)
                (running,) = running_calls
                assert isinstance(preparing, RunningPoll)
                assert running.call_id == preparing.call_id
                assert running.action == "prepare_data"
                assert [
                    message.code
                    for message in running.messages
                    if isinstance(message, ActionMessage)
                ] == ["PREPARE_DATA_STARTED"]

                # A terminated workflow can never finish the attempt its memo still names.
                await handle.terminate()
                memo = await (await handle.describe()).memo_value(
                    RUNNING_ACTION_MEMO, type_hint=tuple[RunningCall, ...]
                )
                assert memo[0].request == running.request
                assert (await _timeline(workspace_id, clients)).running == ()
        finally:
            release.set()
            await env.shutdown()

    asyncio.run(scenario())
