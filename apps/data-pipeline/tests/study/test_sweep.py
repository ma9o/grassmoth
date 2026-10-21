"""Lifecycle collection for scratch runs, telemetry, and caches."""

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from nof1_causal_service.storage import backend as storage
from nof1_causal_service.storage import paths as data_module
from nof1_causal_service.study.persistence.history import StudyRepository
from nof1_causal_service.study.records import AttemptRecord, EditAttempt, Raised
from nof1_causal_service.study.sweep import collect_completed_runs, sweep_workspace

pytestmark = pytest.mark.contract


def test_remote_file_metadata_is_normalized_at_storage_boundary(monkeypatch):
    modified = datetime(2026, 1, 1, tzinfo=UTC)
    fs = Mock(info=Mock(return_value={"size": 42, "LastModified": modified, "name": "file"}))
    monkeypatch.setattr(storage, "is_remote", lambda: True)
    monkeypatch.setattr(storage, "get_fs", lambda: fs)

    assert storage.file_info("s3://bucket/file") == storage.FileInfo(42, modified.timestamp())
    fs.info.assert_called_once_with("s3://bucket/file")


def _workspace(monkeypatch, tmp_path) -> str:
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "sweep-test"


def _run_file(workspace_id: str, run_id: str) -> str:
    path = storage.join(data_module.scratch_run_dir(workspace_id, run_id), "context.json")
    storage.write_text(path, "{}")
    return path


def test_offline_run_collection_removes_completed_and_abandoned_runs(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    StudyRepository(workspace_id).append(
        AttemptRecord(
            seq=3,
            ts="2026-07-15T00:00:00Z",
            attempt=EditAttempt(
                action="edit_model",
                request=None,
                outcome=Raised(error_type="SavedError", error_message="failed"),
            ),
        )
    )
    old_run = _run_file(workspace_id, "seq-000001")
    raised_run = _run_file(workspace_id, "seq-000003")
    unjournaled_run = _run_file(workspace_id, "seq-000004")

    assert collect_completed_runs(workspace_id) == 1
    assert storage.exists(old_run)
    assert not storage.exists(raised_run)
    assert storage.exists(unjournaled_run)

    result = sweep_workspace(workspace_id, now_seconds=1_000, collect_runs=True)

    assert result.removed_runs == 2
    assert not storage.exists(old_run)
    assert not storage.exists(raised_run)
    assert not storage.exists(unjournaled_run)


def test_default_sweep_does_not_infer_run_liveness(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    run = _run_file(workspace_id, "seq-000001")

    result = sweep_workspace(workspace_id, now_seconds=1_000)

    assert result.removed_runs == 0
    assert storage.exists(run)


def test_sweep_expires_events_and_cache_and_bounds_cache_size(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    events = data_module.scratch_events_dir(workspace_id)
    old_event = storage.join(events, f"{100 * 1_000_000_000:020d}-old.json")
    new_event = storage.join(events, f"{900 * 1_000_000_000:020d}-new.json")
    storage.write_text(old_event, "{}")
    storage.write_text(new_event, "{}")

    old_cache = storage.join(data_module.cache_dir(workspace_id), "old.bin")
    first_cache = storage.join(data_module.cache_dir(workspace_id), "first.bin")
    second_cache = storage.join(data_module.cache_dir(workspace_id), "second.bin")
    for path in (old_cache, first_cache, second_cache):
        storage.write_text(path, "12345")
    os.utime(Path(old_cache), (100, 100))
    os.utime(Path(first_cache), (800, 800))
    os.utime(Path(second_cache), (900, 900))

    result = sweep_workspace(
        workspace_id,
        now_seconds=1_000,
        event_retention_seconds=500,
        cache_retention_seconds=500,
        cache_max_bytes=5,
    )

    assert result.removed_events == 1
    assert not storage.exists(old_event)
    assert storage.exists(new_event)
    assert result.removed_cache_files == 2
    assert result.removed_cache_bytes == 10
    assert not storage.exists(old_cache)
    assert not storage.exists(first_cache)
    assert storage.exists(second_cache)
