"""Format-32 publication, exact result ownership and restart contracts."""

import shutil
import subprocess
from pathlib import Path

import pygit2
import pytest

from grassmoth.public import DynamicalModelSpec, GitOid
from grassmoth_server.actions.contracts.inputs import EditModelInput
from grassmoth_server.actions.contracts.requests import EditModelRequest, call_identity
from grassmoth_server.actions.edit_model import edit_model
from grassmoth_server.storage import paths
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
from grassmoth_server.study.records import Applied, AttemptRecord, Raised, action_attempt
from grassmoth_server.study.sweep import collect_completed_runs
from grassmoth_server_test_support.action import applied_record, question_root
from grassmoth_test_support.helpers import make_model
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_DATA_URI", str(tmp_path / "data"))
    return "storage"


def _edit(workspace, clock, seq):
    repository = StudyRepository(workspace)
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=repository.state().current["model"].revision,
            dynamical_model_spec=revise_fixture(
                make_model(["unmeasured_outcome"]), measurement_clock=clock
            ),
        )
    )
    result = edit_model(workspace, request)
    assert isinstance(result, Applied)
    return applied_record(workspace, result, request=request, seq=seq)


def test_format_and_single_publication_root(workspace):
    store = ArtifactStore(workspace)
    assert Path(store.repo.path) == Path(paths.history_dir(workspace))
    assert store.repo.config.get_int("grassmoth.format") == 33
    root = question_root(workspace)
    published = StudyRepository(workspace).append(_edit(workspace, "2d", 2))
    repo = store.repo
    assert set(repo.references) == {
        "refs/question",
        *(
            f"refs/calls/{call_identity(r.record.attempt.request).removeprefix('call:')}"
            for r in (root, published)
            if r.record.attempt.request is not None
        ),
    }
    commit = repo[published.commit_id]
    assert {entry.name for entry in commit.tree} == {
        "call.json",
        "log.json",
        "result.msgpack",
        "records",
    }
    model = commit.tree["records/model"].peel(pygit2.Tree)
    assert {entry.name for entry in model} == {"meta.json", "result.msgpack"}
    assert model["result.msgpack"].id == commit.tree["result.msgpack"].id
    repo.config["grassmoth.format"] = 31
    with pytest.raises(ValueError, match="format 33"):
        ArtifactStore(workspace)


def test_out_of_order_publication_and_first_saved_call_wins_after_cleanup(workspace):
    repository = StudyRepository(workspace)
    question_root(workspace)
    first, last = _edit(workspace, "2d", 2), _edit(workspace, "10d", 10)
    last_saved = repository.append(last)
    first_saved = repository.append(first)
    assert [r.record.seq for r in repository.attempts()] == [1, 2, 10]
    assert repository.latest_seq() == 10
    assert repository.head() == last_saved.commit_id
    assert repository.state().current["model"] == last.attempt.outcome.effects.produced[0]
    failed_request = revise_fixture(
        last.attempt.request,
        input=revise_fixture(
            last.attempt.request.input,
            dynamical_model_spec=DynamicalModelSpec(measurement_clock="11d"),
        ),
    )
    repository.append(
        AttemptRecord(
            seq=11,
            ts=last.ts,
            attempt=action_attempt(
                failed_request, Raised(error_type="Failure", error_message="failed")
            ),
        )
    )
    assert repository.latest_seq() == 11
    collect_completed_runs(workspace)
    assert not Path(paths.call_dir(workspace, call_identity(first.attempt.request))).exists()
    assert (
        repository.append(revise_fixture(first, seq=100, ts="2026-03-01T00:00:00Z")) == first_saved
    )
    assert repository.latest_seq() == 11


def test_packed_repository_alone_serves_calls_and_timeline(workspace, tmp_path, monkeypatch):
    repository = StudyRepository(workspace)
    question_root(workspace)
    repository.append(_edit(workspace, "2d", 2))
    expected = [(r, completed_call_msgpack(workspace, r)) for r in repository.attempts()]
    subprocess.run(["git", "--git-dir", str(repository.path), "repack", "-ad"], check=True)
    subprocess.run(["git", "--git-dir", str(repository.path), "pack-refs", "--all"], check=True)
    clone_root = tmp_path / "copy"
    shutil.copytree(repository.path, clone_root / workspace / "history.git")
    monkeypatch.setattr(paths, "_DATA_URI", str(clone_root))
    copied = StudyRepository(workspace)
    assert copied.attempts() == [r for r, _ in expected]
    for revision, body in expected:
        assert completed_call_msgpack(workspace, copied.record(revision.commit_id)) == body
        assert copied.read_file(revision.commit_id, "result.msgpack") == repository.read_file(
            revision.commit_id, "result.msgpack"
        )
    assert set((clone_root / workspace).iterdir()) == {clone_root / workspace / "history.git"}


def test_concurrent_publication_indexes_only_law_arrays_and_keeps_first_owner(
    workspace, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import numpy as np

    from grassmoth.public import EditModelOutput, NumericalArray
    from grassmoth_server.actions.contracts.effects import ActionEffects
    from grassmoth_server.study.persistence import codec, history
    from grassmoth_test_support.git import git_oid

    law = NumericalArray.from_numpy(np.array([1.0, 2.0]))
    output = EditModelOutput(
        dynamical_model_spec=DynamicalModelSpec.model_validate(
            {
                "distributions": {
                    "distribution:law": {"distribution": "Delta", "params": {"v": {"npy": law.npy}}}
                }
            }
        ),
        checks={"specification": (), "question": {"findings": ()}},
        identification={"outcome": None, "treatments": {}},
        pruning={},
    )
    store = ArtifactStore(workspace)
    records = []
    for seq in (2, 10):
        request = EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=git_oid(seq), dynamical_model_spec=DynamicalModelSpec()
            )
        )
        revision = store.write_result(
            revise_fixture(
                output,
                dynamical_model_spec=revise_fixture(
                    output.dynamical_model_spec, measurement_clock=f"{seq}d"
                ),
            ),
            call_id=call_identity(request),
        )
        records.append(
            AttemptRecord(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                attempt=action_attempt(request, Applied(result=revision, effects=ActionEffects())),
            )
        )
    repository = StudyRepository(workspace)
    with monkeypatch.context() as failure:
        failure.setattr(
            history, "write_blob", lambda *_: (_ for _ in ()).throw(OSError("full disk"))
        )
        with pytest.raises(OSError, match="full disk"):
            repository.append(records[0])
    assert not tuple(store.repo.references)

    def cannot_decode(*_args):
        pytest.fail("Publication must use the law index collected by encoding")

    monkeypatch.setattr(codec, "unpack_result", cannot_decode)
    later_saved = Event()

    def earlier():
        assert later_saved.wait(10)
        return StudyRepository(workspace).append(records[0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(earlier)
        last = executor.submit(StudyRepository(workspace).append, records[1]).result()
        owner = str(store.repo.references[f"refs/arrays/{law.identity}"].target)
        later_saved.set()
        first.result()
    assert [r.record.seq for r in repository.attempts()] == [2, 10]
    assert repository.head() == last.commit_id
    assert str(store.repo.references[f"refs/arrays/{law.identity}"].target) == owner
    assert [r for r in store.repo.references if r.startswith("refs/arrays/")] == [
        f"refs/arrays/{law.identity}"
    ]
