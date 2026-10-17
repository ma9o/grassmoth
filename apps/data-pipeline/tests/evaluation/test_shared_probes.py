"""Shared preparation owns one frozen ledger; warmup is explicit for every treatment."""

from email.message import Message
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from evaluation.declarations import Experiment
from evaluation.run import assigned_settings
from evaluation.workspaces import copy_anchor_workspace, ledger_hashes

from tests.evaluation.test_experiments import declaration

pytestmark = pytest.mark.contract


def test_probe_assigns_positive_warmup_and_copies_only_idle_durable_state(tmp_path):
    payload = {**declaration(), "probe_samples": 16, "probe_warmup": 32}
    experiment = Experiment.model_validate(payload)
    settings = assigned_settings(experiment, experiment.arms[0], 13)
    assert settings.num_warmup == 32
    assert settings.num_samples_per_chain == 16
    origin = tmp_path / "anchor"
    (origin / "study/history.git").mkdir(parents=True)
    (origin / "study/history.git/HEAD").write_text("ref: refs/heads/main\n")
    (origin / "store").mkdir()
    (origin / "store/blob").write_bytes(b"native-array")
    (origin / "scratch").mkdir()
    (origin / "scratch/live").write_text("not durable")
    client = Mock(workspace="anchor")
    hashes = copy_anchor_workspace(tmp_path, client, "arm")
    assert hashes == ledger_hashes(origin) == ledger_hashes(tmp_path / "arm")
    assert not (tmp_path / "arm/scratch").exists()
    client.require_idle.assert_called_once()
    with pytest.raises(FileExistsError):
        copy_anchor_workspace(tmp_path, client, "arm")
    client.require_idle = Mock(side_effect=ValueError("running"))
    with pytest.raises(ValueError, match="running"):
        copy_anchor_workspace(tmp_path, client, "blocked")
    assert not (tmp_path / "blocked").exists()


def test_lost_fit_acknowledgement_retains_identity_without_resubmission(tmp_path, monkeypatch):
    from evaluation.client import ActionClient

    from nof1_causal_lab.actions.contracts.bodies import FitInput, NewFitTarget
    from nof1_causal_lab.actions.contracts.requests import FitRequest, call_identity
    from nof1_causal_lab.artifacts.observations.reference import DataRef
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
    reads = Mock(
        side_effect=[
            HTTPError("lookup", 404, "not saved", Message(), None),
            HTTPError("submission", 500, "response lost", Message(), None),
        ]
    )
    monkeypatch.setattr(client, "_read", reads)
    with pytest.raises(HTTPError, match="response lost"):
        client.call(request)
    identity = call_identity(request)
    assert (tmp_path / (identity + ".request.json")).read_text() == request.model_dump_json(
        round_trip=True
    )
    assert client.reused[identity] is False
    assert [call.args[0].get_method() for call in reads.call_args_list] == ["GET", "POST"]
