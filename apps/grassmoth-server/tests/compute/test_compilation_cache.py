"""Cache persistence contracts do not execute scientific numerical kernels."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from grassmoth_server.compute.compilation_cache import (
    FIT_CACHE_POLICY,
    CacheLimit,
    CachePolicy,
    local_compilation_cache,
    persist_cache,
    stage_cache,
)

pytestmark = pytest.mark.contract
AUTOTUNE = "xla_gpu_per_fusion_autotune_cache_dir"
KERNELS = "xla_gpu_kernel_cache_file"


@pytest.mark.parametrize("package", ["grassmoth", "grassmoth_server"])
def test_source_identity_tracks_both_workspace_packages(tmp_path, monkeypatch, package):
    from grassmoth_server.actions.fit import runtime

    library = tmp_path / "packages/grassmoth/src/grassmoth"
    service = tmp_path / "apps/grassmoth-server/src/grassmoth_server"
    for root in (library, service):
        root.mkdir(parents=True)
        (root / "__init__.py").write_text('"""Package source."""\n')
    monkeypatch.setattr(runtime, "files", lambda _package: library)
    monkeypatch.setattr(runtime, "__file__", str(service / "actions/fit/runtime.py"))
    before = runtime.numerical_source_digest()
    owner = library if package == "grassmoth" else service
    (owner / "new_owner.py").write_text("VALUE = 1\n")
    added = runtime.numerical_source_digest()
    assert added != before
    (owner / "new_owner.py").write_text("VALUE = 2\n")
    assert runtime.numerical_source_digest() not in {before, added}


@pytest.mark.parametrize("output_only", [False, True])
def test_jax_persistent_cache_reuses_executable_with_runtime_values_across_processes(
    tmp_path, output_only
):
    script = textwrap.dedent(
        """
        import json, os
        import jax
        import jax.numpy as jnp
        from jax._src import compiler
        from grassmoth_server.compute.compilation_cache import prepare_program
        from grassmoth.artifacts.model.expressions import StateExpression
        from grassmoth.models.ssm.dynamics.expression import ExpressionComponent

        template = ExpressionComponent(
            target=0, edge_owned=False, expression=StateExpression(construct_id='construct:test'),
            evaluate_fn=lambda values, params: values,
        )
        offset = jnp.asarray(float(os.environ['OFFSET']))
        enabled = jnp.asarray(os.environ['ENABLED'] == '1')
        power = int(os.environ['POWER'])
        value = jnp.ones((2,), dtype=jnp.float32)
        output_only = os.environ['OUTPUT_ONLY'] == '1'

        def program(carry):
            values = carry if output_only else carry[1]
            result = jax.lax.cond(enabled, lambda: values + offset, lambda: values - offset)
            return template, result ** power

        compiled = []
        original = compiler.backend_compile_and_load
        def observe(*args, **kwargs):
            assert os.environ['FORBID_COMPILE'] == '0', 'warm program invoked XLA compilation'
            compiled.append(True)
            return original(*args, **kwargs)
        compiler.backend_compile_and_load = observe
        leaf = jax.ShapeDtypeStruct((2,), jnp.float32)
        compile_program, operands, shapes, memory = prepare_program(
            program, leaf if output_only else (template, leaf),
        )
        executable = compile_program()
        actual = executable(operands, value if output_only else (template, value))
        assert actual[0].evaluate_fn is template.evaluate_fn
        print(json.dumps({'value': actual[1].tolist(), 'shape': shapes[1].shape,
                          'bytes': memory.device_bytes, 'compiled': len(compiled)}))
        """
    )
    results = []
    for offset, enabled, power, forbid_compile in ((1, 0, 2, 0), (3, 1, 2, 1), (3, 1, 3, 0)):
        process = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
                "OFFSET": str(offset),
                "ENABLED": str(enabled),
                "POWER": str(power),
                "FORBID_COMPILE": str(forbid_compile),
                "OUTPUT_ONLY": str(int(output_only)),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout))
    assert [result["value"] for result in results] == [[0.0, 0.0], [16.0, 16.0], [64.0, 64.0]]
    assert [result["compiled"] for result in results] == [1, 0, 1]
    assert all(result["shape"] == [2] and result["bytes"] > 0 for result in results)
    assert not tuple(tmp_path.glob("grassmoth-*-cache"))


def test_jax_compilation_preserves_output_trees_and_weak_scalar_types(tmp_path):
    from grassmoth_server.compute.compilation_cache import compile_program

    def increment(value):
        return jax.tree.map(lambda leaf: leaf + 1, value)

    scalar = jnp.asarray(2.0)
    with local_compilation_cache(tmp_path, commit=lambda: None):
        for value in ({"a": scalar}, {"b": scalar}, (scalar,), scalar):
            actual = compile_program(increment, value).call(value)
            assert jax.tree.structure(actual) == jax.tree.structure(value)
            for leaf in jax.tree.leaves(actual):
                assert jax.typeof(leaf) == jax.typeof(scalar)
                assert leaf == 3


def _file(root, name, content, modified):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, ns=(modified, modified))


def test_separate_selection_bounds_round_trip_and_unchanged_skip_commit(tmp_path):
    working = tmp_path / "working"
    policy = CachePolicy(
        executables=CacheLimit(files=2, bytes=9),
        autotune=CacheLimit(files=1, bytes=5),
        kernels=CacheLimit(files=1, bytes=6),
    )
    _file(working, "old-cache", b"1234", 1)
    _file(working, "new-cache", b"12345", 3)
    _file(working, "too-big-cache", b"0123456789", 4)
    _file(working, "overflow-cache", b"12345", 2)
    _file(working, "ignored-atime", b"ignored", 5)
    _file(working, f"{AUTOTUNE}/old.opaque", b"123", 1)
    _file(working, f"{AUTOTUNE}/nested/new.opaque", b"1234", 2)
    _file(working, KERNELS, b"kernel", 2)
    (working / "link-cache").symlink_to(working / "new-cache")
    archive = tmp_path / "volume" / "selected.tar"
    commit = Mock()
    persisted = persist_cache(archive, working, (), policy, commit)
    assert (persisted.files, persisted.bytes) == (4, 19)
    assert persisted.archive_bytes == archive.stat().st_size <= policy.archive_bytes()
    assert min(persisted.copy_seconds, persisted.selection_seconds, persisted.commit_seconds) >= 0
    commit.assert_called_once_with()
    staged, transfer = stage_cache(archive, tmp_path / "fresh", policy)
    assert [entry.path for entry in staged] == [
        "new-cache",
        "old-cache",
        KERNELS,
        f"{AUTOTUNE}/nested/new.opaque",
    ]
    assert (transfer.files, transfer.bytes, transfer.archive_bytes) == (
        4,
        19,
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
            target.truncate(policy.archive_bytes() + 1)
    with pytest.raises(ValueError, match="cache"):
        stage_cache(archive, tmp_path / "local", policy)
    assert not (tmp_path / "escape").exists()


def test_local_lifecycle_isolated_namespaces_failure_cleanup_and_accounting(tmp_path, monkeypatch):
    # Backend discovery is separate from this I/O contract.
    monkeypatch.setattr(
        "grassmoth_server.compute.compilation_cache.accelerator_partition", lambda: "a" * 64
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
        assert jax.config.jax_persistent_cache_enable_xla_caches == f"{AUTOTUNE},{KERNELS}"
        _file(working, "compiled-cache", b"binary", 100)
    assert not working.exists()
    assert [(r.phase, r.files, r.bytes) for r in reports] == [
        ("stage", 0, 0),
        ("persist", 1, 6),
    ]
    assert {key: jax.config.values[key] for key in before} == before
    reports.clear()
    first_path = working
    with local_compilation_cache(persistent, commit=commit, report=reports.append) as working:
        assert working == first_path
        assert (working / "compiled-cache").read_bytes() == b"binary"
    assert [(r.phase, r.files) for r in reports] == [
        ("stage", 1),
        ("persist", 0),
    ]
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
        "grassmoth_server.compute.compilation_cache.accelerator_partition", lambda: "a" * 64
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


def test_kernel_cache_path_does_not_change_executable_keys_between_contexts(tmp_path, monkeypatch):
    from jax._src import cache_key

    keys = []
    get_key = cache_key.get

    def observe(*args, **kwargs):
        key = get_key(*args, **kwargs)
        keys.append(key)
        return key

    monkeypatch.setattr(cache_key, "get", observe)

    def arithmetic(value):
        return value * value + 1

    for _ in range(2):
        jax.clear_caches()
        with local_compilation_cache(tmp_path, commit=lambda: None):
            jax.jit(arithmetic).lower(jax.ShapeDtypeStruct((4,), "float32")).compile()
    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_topology_writers_keep_separate_jax_binaries(tmp_path, monkeypatch):
    from grassmoth_server.compute import compilation_cache as cache

    commit = Mock()
    for topology in ("a" * 64, "b" * 64):
        monkeypatch.setattr(cache, "accelerator_partition", lambda topology=topology: topology)
        with local_compilation_cache(tmp_path, commit=commit) as working:
            assert not (working / "compiled-cache").exists()
            assert not (working / KERNELS).exists()
            _file(working, "compiled-cache", topology.encode(), 2)
            _file(working, KERNELS, topology.encode(), 2)
    assert commit.call_count == 2
    snapshots = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert set(snapshots) == {"a" * 64 + ".tar", "b" * 64 + ".tar"}
    for topology in ("a" * 64, "b" * 64):
        monkeypatch.setattr(cache, "accelerator_partition", lambda topology=topology: topology)
        with local_compilation_cache(tmp_path, commit=commit) as working:
            assert (working / "compiled-cache").read_bytes() == topology.encode()
            assert (working / KERNELS).read_bytes() == topology.encode()
    assert commit.call_count == 2
    assert snapshots == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert FIT_CACHE_POLICY.archive_bytes() <= 336 * 1024**2 + 128 * 1024 + 513 * 1024 + 10240


def test_accelerator_partition_matches_jax_fingerprint_component(monkeypatch):
    from grassmoth_server.compute import compilation_cache as cache

    devices = [object()]
    monkeypatch.setattr(cache.jax, "local_devices", lambda: devices)
    topology = SimpleNamespace(fingerprint=lambda: 123456789)
    describe = Mock(return_value=topology)
    monkeypatch.setattr(cache.xla_client, "get_topology_for_devices", describe)
    assert (
        cache.accelerator_partition() == hashlib.sha256((123456789).to_bytes(8, "big")).hexdigest()
    )
    describe.assert_called_once_with(devices)
