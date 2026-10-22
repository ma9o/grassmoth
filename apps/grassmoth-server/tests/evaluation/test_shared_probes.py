"""Probe arms share a saved anchor through actions; warmup is explicit for every treatment."""

from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from evaluation.declarations import Experiment
from evaluation.run import assigned_settings
from tests.support.experiments import declaration

pytestmark = pytest.mark.contract


def test_probe_assigns_positive_warmup():
    payload = {**declaration(), "probe_samples": 16, "probe_warmup": 32}
    experiment = Experiment.model_validate(payload)
    settings = assigned_settings(experiment, experiment.arms[0], 13)
    assert settings.num_warmup == 32
    assert settings.num_samples_per_chain == 16


def test_probe_arms_continue_one_anchor_in_its_workspace(tmp_path, monkeypatch):
    from evaluation import run as runner
    from evaluation.client import ActionClient
    from evaluation.recipes import PreparedCase
    from grassmoth_server.public.contracts import ContinueFitTarget, DataRef, NewFitTarget
    from tests.support.git import git_oid

    experiment = Experiment.model_validate({**declaration(), "probe_samples": 16})
    prepared = PreparedCase(
        model_ref=git_oid(1),
        data_ref=DataRef(revision=git_oid(2), replicate_index=0),
        model_digest="model",
        data_digest="data",
        recipe_digest="recipe",
        provenance="test",
        findings=(),
    )
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(runner, "prepare_case", prepare)
    submitted = []
    checkpoint = SimpleNamespace(
        settings=experiment.base.revised(seed=13), content_digest=lambda: "checkpoint"
    )

    def submit(client, request):
        submitted.append((client.workspace, request))
        return SimpleNamespace(
            action="fit",
            call_id=request.call_id,
            commit_id=git_oid(len(submitted) + 2),
            body=SimpleNamespace(
                checks=SimpleNamespace(preflight=()),
                inference=SimpleNamespace(evidence=SimpleNamespace(checkpoint=checkpoint)),
            ),
        )

    monkeypatch.setattr(ActionClient, "call", submit)
    # Scheduling is the contract under test; reporting needs actual numerical results.
    monkeypatch.setattr(runner, "_controls", Mock(side_effect=RuntimeError("calls complete")))
    with pytest.raises(RuntimeError, match="calls complete"):
        runner.run(
            experiment,
            Path(__file__).parents[2] / "evaluation/cases",
            "http://isolated.invalid",
            tmp_path / "run",
            None,
        )
    prepare.assert_called_once()
    assert len(submitted) == 3
    assert len({workspace for workspace, _ in submitted}) == 1
    assert isinstance(submitted[0][1].input.target, NewFitTarget)
    assert all(
        request.input.target == ContinueFitTarget(fit_ref=git_oid(3))
        for _, request in submitted[1:]
    )
    assert [request.input.settings.num_particles for _, request in submitted[1:]] == [8, 16]


def test_lost_fit_acknowledgement_retains_identity_without_resubmission(tmp_path, monkeypatch):
    from evaluation.client import ActionClient
    from grassmoth.artifacts.observations.reference import DataRef
    from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
    from grassmoth_server.actions.contracts.requests import FitRequest, call_identity
    from grassmoth_server.public.contracts import StudyError
    from tests.support.git import git_oid

    request = FitRequest(
        input=FitInput(
            target=NewFitTarget(
                dynamical_model_spec_ref=git_oid(1),
                data_ref=DataRef(revision=git_oid(2), replicate_index=0),
            )
        )
    )
    client = ActionClient("http://isolated.invalid", "probe", tmp_path)
    transport = Mock(
        poll=Mock(return_value=StudyError(code="unknown_call", detail="not saved")),
        submit=Mock(side_effect=HTTPError("submission", 500, "response lost", Message(), None)),
    )
    monkeypatch.setattr(client, "transport", transport)
    with pytest.raises(HTTPError, match="response lost"):
        client.call(request)
    identity = call_identity(request)
    assert (tmp_path / (identity + ".request.json")).read_text() == request.model_dump_json(
        round_trip=True
    )
    assert client.reused[identity] is False
    assert [call[0] for call in transport.mock_calls] == ["poll", "submit"]
