"""Shared preparation owns one frozen ledger; warmup is explicit for every treatment."""

from unittest.mock import Mock

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
