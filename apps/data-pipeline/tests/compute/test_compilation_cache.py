"""Cache persistence contracts do not execute scientific numerical kernels."""

import hashlib
import io
import json
import os
import tarfile
from unittest.mock import Mock

import jax
import pytest

from nof1_causal_lab.compute.compilation_cache import (
    FIT_CACHE_POLICY,
    CacheLimit,
    CachePolicy,
    local_compilation_cache,
    persist_cache,
    stage_cache,
)

pytestmark = pytest.mark.contract
AUTOTUNE = "xla_gpu_per_fusion_autotune_cache_dir"


def _file(root, name, content, modified):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, ns=(modified, modified))


def test_separate_selection_bounds_round_trip_and_unchanged_skip_commit(tmp_path):
    working = tmp_path / "working"
    policy = CachePolicy(
        executables=CacheLimit(files=2, bytes=9), autotune=CacheLimit(files=1, bytes=5)
    )
    _file(working, "old-cache", b"1234", 1)
    _file(working, "new-cache", b"12345", 3)
    _file(working, "too-big-cache", b"0123456789", 4)
    _file(working, "overflow-cache", b"12345", 2)
    _file(working, "ignored-atime", b"ignored", 5)
    _file(working, f"{AUTOTUNE}/old.opaque", b"123", 1)
    _file(working, f"{AUTOTUNE}/nested/new.opaque", b"1234", 2)
    (working / "link-cache").symlink_to(working / "new-cache")
    archive = tmp_path / "volume" / "selected.tar"
    commit = Mock()
    persisted = persist_cache(archive, working, (), policy, commit)
    assert (persisted.files, persisted.bytes) == (3, 13)
    assert persisted.archive_bytes == archive.stat().st_size <= policy.archive_bytes
    assert min(persisted.copy_seconds, persisted.selection_seconds, persisted.commit_seconds) >= 0
    commit.assert_called_once_with()
    staged, transfer = stage_cache(archive, tmp_path / "fresh", policy)
    assert [entry.path for entry in staged] == [
        "new-cache",
        "old-cache",
        f"{AUTOTUNE}/nested/new.opaque",
    ]
    assert (transfer.files, transfer.bytes, transfer.archive_bytes) == (
        3,
        13,
        persisted.archive_bytes,
    )
    for entry in staged:
        assert (tmp_path / "fresh" / entry.path).read_bytes() == (working / entry.path).read_bytes()
        assert (tmp_path / "fresh" / entry.path).stat().st_mtime_ns == entry.modified_ns
    commit.reset_mock()
    unchanged = persist_cache(archive, tmp_path / "fresh", staged, policy, commit)
    assert (unchanged.files, unchanged.bytes, unchanged.copy_seconds, unchanged.commit_seconds) == (
        0,
        0,
        0,
        0,
    )
    commit.assert_not_called()


def _bundle(path, entries, content=b"abc"):
    manifest = json.dumps(entries).encode()
    with tarfile.open(path, "w") as archive:
        for name, data in [("manifest.json", manifest), ("files/0", content)]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("defect", ["path", "size", "digest", "files", "archive"])
def test_invalid_bundles_fail_before_use_without_fallback(tmp_path, defect):
    archive = tmp_path / "bundle.tar"
    policy = FIT_CACHE_POLICY.revised(executables=CacheLimit(files=1, bytes=4))
    entry = {
        "path": "one-cache",
        "size": 3,
        "modified_ns": 1,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    if defect == "path":
        entry["path"] = f"{AUTOTUNE}/../../escape"
    elif defect == "size":
        entry["size"] = 5
    elif defect == "digest":
        entry["sha256"] = "0" * 64
    entries = [entry, {**entry, "path": "two-cache"}] if defect == "files" else [entry]
    _bundle(archive, entries)
    if defect == "archive":
        with archive.open("r+b") as target:
            target.truncate(policy.archive_bytes + 1)
    with pytest.raises(ValueError, match="cache"):
        stage_cache(archive, tmp_path / "local", policy)
    assert not (tmp_path / "escape").exists()


def test_local_lifecycle_isolated_namespaces_failure_cleanup_and_accounting(tmp_path, monkeypatch):
    # Backend discovery is separate from this I/O contract.
    monkeypatch.setattr(
        "nof1_causal_lab.compute.compilation_cache.cache_namespace", lambda _policy: "a" * 64
    )
    persistent = tmp_path / "volume"
    persistent.mkdir()
    (persistent / "unrelated.tar").write_bytes(b"never opened or inventoried")
    before = {key: value for key, value in jax.config.values.items() if "cache" in key}
    reports = []
    commit = Mock()
    with local_compilation_cache(persistent, commit=commit, report=reports.append) as working:
        assert not working.is_relative_to(persistent)
        assert jax.config.jax_compilation_cache_dir == str(working)
        _file(working, "compiled-cache", b"binary", 100)
    assert not working.exists()
    assert [(r.phase, r.files, r.bytes) for r in reports] == [("stage", 0, 0), ("persist", 1, 6)]
    assert {key: jax.config.values[key] for key in before} == before
    reports.clear()
    with local_compilation_cache(persistent, commit=commit, report=reports.append) as working:
        assert (working / "compiled-cache").read_bytes() == b"binary"
    assert [(r.phase, r.files) for r in reports] == [("stage", 1), ("persist", 0)]
    commit.assert_called_once_with()
    with (
        pytest.raises(RuntimeError, match="fit failed"),
        local_compilation_cache(persistent, commit=commit) as working,
    ):
        raise RuntimeError("fit failed")
    assert not working.exists()
    assert {key: jax.config.values[key] for key in before} == before
    commit.assert_called_once_with()
    assert (persistent / "unrelated.tar").read_bytes() == b"never opened or inventoried"


def test_commit_failure_propagates_and_restores_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nof1_causal_lab.compute.compilation_cache.cache_namespace", lambda _policy: "a" * 64
    )
    original = jax.config.jax_compilation_cache_dir
    with (
        pytest.raises(OSError, match="commit failed"),
        local_compilation_cache(
            tmp_path, commit=Mock(side_effect=OSError("commit failed"))
        ) as working,
    ):
        _file(working, "one-cache", b"binary", 1)
    assert jax.config.jax_compilation_cache_dir == original
