"""Only saved call refs authorize automatic cleanup of temporary work."""

from pathlib import Path

import pytest

from grassmoth.public import DynamicalModelSpec, GitOid
from grassmoth_server.actions.contracts.inputs import EditModelInput
from grassmoth_server.actions.contracts.requests import EditModelRequest, call_identity
from grassmoth_server.storage import paths
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.records import AttemptRecord, Raised, action_attempt
from grassmoth_server.study.sweep import cleanup_saved_calls, collect_completed_runs, sweep_cli
from grassmoth_test_support.git import git_oid

pytestmark = pytest.mark.contract


def test_failed_publication_preserves_tmp_and_startup_recovers_saved_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_DATA_URI", str(tmp_path))
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=git_oid(1), dynamical_model_spec=DynamicalModelSpec()
        )
    )
    directory = Path(paths.call_dir("study", call_identity(request)))
    sources = directory / "sources" / "tmp" / "input"
    sources.mkdir(parents=True)
    (sources / "source.csv").write_text("timestamp,value\n2026-01-01,1\n")
    (directory / "profiles").mkdir()
    (directory / "profiles" / "fit.json").write_text('{"elapsed":1}')
    unfinished = Path(paths.call_dir("study", "orphan"))
    unfinished.mkdir()
    record = AttemptRecord(
        seq=3,
        ts="2026-01-01T00:00:00Z",
        attempt=action_attempt(request, Raised(error_type="Failed", error_message="No result")),
    )
    repository = StudyRepository("study")
    import grassmoth_server.study.persistence.history as history

    with monkeypatch.context() as failing:
        failing.setattr(
            history, "write_blob", lambda *_args: (_ for _ in ()).throw(OSError("disk failed"))
        )
        with pytest.raises(OSError, match="disk failed"):
            repository.append(record)
    assert collect_completed_runs("study") == 0
    assert directory.exists()
    saved = repository.append(record)
    assert (
        repository.read_file(saved.commit_id, "sources/tmp/input/source.csv")
        == (sources / "source.csv").read_bytes()
    )
    assert repository.read_file(saved.commit_id, "profiles/fit.json") == b'{"elapsed":1}'
    cleanup_saved_calls()
    cleanup_saved_calls()
    assert not directory.exists()
    assert unfinished.exists()
    assert repository.append(record) == saved
    monkeypatch.setattr("sys.argv", ["sweep", "study", "--offline"])
    sweep_cli()
    assert not unfinished.exists()
