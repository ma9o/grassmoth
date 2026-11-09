"""HTTP orchestration contracts for rebuilding the non-fit development study."""

from __future__ import annotations

import httpx
import pytest
from pydantic import TypeAdapter

from grassmoth_server.actions.contracts.polling import ActionResult, FailedPoll, RunningPoll
from grassmoth_server.actions.contracts.requests import ActionInput
from grassmoth_server.actions.contracts.timeline import TimelineResponse, TimelineRevision
from grassmoth_server_test_support.action_examples import examples
from grassmoth_server_test_support.action_values import oid
from scripts.dev.rebuild_france1940simple import Recipe, Study, rebuild

pytestmark = pytest.mark.contract
REQUEST = TypeAdapter(ActionInput)


def test_rebuild_pins_http_outputs_and_skips_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {entry.name: entry for entry in examples()}
    selected = tuple(
        values[name]
        for name in (
            "question",
            "authored",
            "prepared",
            "simulated",
            "dataCompared",
            "modelCompared",
        )
    )
    timeline = TimelineResponse(
        attempts=tuple(
            TimelineRevision.model_validate(
                {
                    "call_id": entry.poll.call_id,
                    "commit_id": entry.revision.commit_id,
                    "parent_ids": entry.revision.parent_ids,
                    "record": {
                        "seq": entry.revision.record.seq,
                        "ts": entry.revision.record.ts,
                        "attempt": {
                            "action": entry.request.action,
                            "request": entry.request,
                            "outcome": {
                                "status": "applied",
                                "result": None,
                                "effects": entry.revision.record.attempt.outcome.effects,
                            },
                        },
                    },
                }
            )
            for entry in selected
            if entry.revision is not None
            and entry.revision.record.attempt.outcome.status == "applied"
        ),
        dependencies=(),
        running=(),
    )
    requests: list[ActionInput] = []
    uploads: list[bytes] = []
    polls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/upload":
            uploads.append(request.read())
            return httpx.Response(200, json="REBUILD/tmp/input/historical_inputs.csv")
        if request.url.path.endswith("/timeline"):
            return httpx.Response(200, content=timeline.model_dump_json())
        if request.method == "POST":
            assert request.headers["content-type"] == "application/json"
            submitted = REQUEST.validate_json(request.content)
            requests.append(submitted)
            result = (
                (
                    values["question"].poll
                    if submitted.input.parent_ref is None
                    else values["authored"].poll
                )
                if submitted.action == "edit_model"
                else next(entry.poll for entry in selected if entry.poll.action == submitted.action)
            )
            if submitted.action == "edit_model" and submitted.input.parent_ref is None:
                return httpx.Response(
                    200,
                    content=ActionResult.from_poll(
                        RunningPoll(call_id=result.call_id, action=result.action)
                    ).payload,
                )
            return httpx.Response(200, content=ActionResult.from_poll(result).payload)
        polls.append(request.url.path)
        return httpx.Response(200, content=ActionResult.from_poll(values["question"].poll).payload)

    monkeypatch.setattr("scripts.dev.rebuild_france1940simple.time.sleep", lambda _: None)
    with httpx.Client(base_url="http://study", transport=httpx.MockTransport(handle)) as client:
        rebuild(Study(client, "REBUILD", 30), Recipe.read())

    assert len(uploads) == 1
    assert b"historical_inputs.csv" in uploads[0]
    assert b"1936-01-01,48,3.2" in uploads[0]
    assert polls == [f"/api/studies/REBUILD/edit_model/{values['question'].poll.call_id}"]
    assert [request.action for request in requests] == [
        "edit_model",
        "edit_model",
        "prepare_data",
        "simulate",
        "data_diff",
        "model_diff",
    ]
    _, model, data, simulation, comparison, model_comparison = requests
    assert model.action == "edit_model"
    assert model.input.parent_ref == oid(101)
    assert data.action == "prepare_data"
    assert data.input.dynamical_model_spec_ref == oid(104)
    assert data.input.source == "tmp/input"
    assert simulation.action == "simulate"
    assert simulation.input.dynamical_model_spec_ref == oid(104)
    assert simulation.input.num_replicates == 100
    assert comparison.action == "data_diff"
    assert comparison.input.left_ref[0].revision == oid(105)
    assert comparison.input.left_ref[0].replicate_index == 0
    assert comparison.input.right_ref[0].revision == values["simulated"].poll.commit_id
    assert comparison.input.right_ref[0].replicate_index is None
    assert model_comparison.action == "model_diff"
    assert model_comparison.input.before_ref == oid(101)
    assert model_comparison.input.after_ref == oid(104)


def test_failed_action_stops_rebuild() -> None:
    failed = next(entry.poll for entry in examples() if entry.name == "editFailed")
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/upload":
            return httpx.Response(200, json="REBUILD/tmp/input/historical_inputs.csv")
        return httpx.Response(
            200,
            content=ActionResult.from_poll(
                FailedPoll(
                    action="edit_model",
                    call_id=failed.call_id,
                    commit_id=failed.commit_id,
                    messages=failed.messages,
                )
            ).payload,
        )

    with (
        httpx.Client(base_url="http://study", transport=httpx.MockTransport(handle)) as client,
        pytest.raises(RuntimeError, match="SyntheticProposalError"),
    ):
        rebuild(Study(client, "REBUILD", 30), Recipe.read())
    assert paths == ["/api/upload", "/api/studies/REBUILD/edit_model"]


def test_timeout_keeps_the_call_available_to_resume() -> None:
    question = Recipe.read().initial_model
    saved = next(entry.poll for entry in examples() if entry.name == "question")
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            200,
            content=ActionResult.from_poll(
                RunningPoll(action=question.action, call_id=saved.call_id)
            ).payload,
        )

    with (
        httpx.Client(base_url="http://study", transport=httpx.MockTransport(handle)) as client,
        pytest.raises(TimeoutError, match=f"{saved.call_id}.*server continues"),
    ):
        Study(client, "REBUILD", 0).run(question)
    assert paths == ["/api/studies/REBUILD/edit_model"]
