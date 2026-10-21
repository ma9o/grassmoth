"""Cache persistence contracts do not execute scientific numerical kernels."""

import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import textwrap
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from nof1_causal_service.compute.compilation_cache import (
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


def test_staging_preserves_public_and_user_created_jax_callables():
    from nof1_causal_service.compute import compilation_cache as cache

    public = (jnp.log, jnp.add)
    restored = pickle.loads(cache._program_bytes(public))
    assert all(left is right for left, right in zip(public, restored, strict=True))

    def programs(offset):
        generated = FunctionType(
            (lambda value, increment: value + increment).__code__,
            {},
            argdefs=(offset,),
        )
        return (
            jax.jit(lambda value: value + offset),
            jnp.frompyfunc(lambda value: value + offset, 1, 1),
            generated,
        )

    first = cache._program_bytes(programs(1))
    second = cache._program_bytes(programs(2))
    assert first != second
    for data, expected in ((first, 4), (second, 5)):
        for function in pickle.loads(data):
            assert function(jnp.asarray(3, dtype=jnp.float32)) == expected


@pytest.mark.parametrize("dependency", ["cuthbert", "cuthbertlib", "cd-dynamax"])
def test_staged_program_key_tracks_all_installed_dependency_versions(monkeypatch, dependency):
    from nof1_causal_service.compute import compilation_cache as cache

    monkeypatch.setattr(cache, "numerical_source_digest", lambda: "source")
    monkeypatch.setattr(cache, "cache_namespace", lambda _policy: "namespace")
    packages = [
        SimpleNamespace(name="jax", version="unchanged"),
        SimpleNamespace(name=dependency, version="1"),
    ]
    monkeypatch.setattr(cache, "distributions", lambda: iter(packages))
    layout = (jax.ShapeDtypeStruct((2,), "float32"),)
    original = cache._program_key(b"same structure", layout)
    packages.reverse()
    assert cache._program_key(b"same structure", layout) == original
    packages[0] = SimpleNamespace(name=dependency, version="2")
    assert cache._program_key(b"same structure", layout) != original


@pytest.mark.parametrize("output_only", [False, True])
def test_staged_program_reuses_structure_across_processes_and_rebinds_arrays(tmp_path, output_only):
    script = textwrap.dedent(
        """
        import json, os
        import jax
        import jax.numpy as jnp
        from nof1_causal_service.compute.compilation_cache import prepare_program
        from nof1_causal_lab.artifacts.model.expressions import StateExpression
        from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponent

        template = ExpressionComponent(
            target=0, edge_owned=False, expression=StateExpression(construct_id='construct:test'),
            evaluate_fn=lambda values, params: values,
        )

        def make_program(offset, power):
            def program(carry):
                assert os.environ['FORBID_TRACE'] == '0', 'cache hit retraced the program'
                if os.environ['OUTPUT_ONLY'] == '1':
                    return template, (carry + offset) ** power
                return jax.lax.cond(
                    True, lambda: (template, (carry[1] + offset) ** power), lambda: carry
                )
            return program

        output_only = os.environ['OUTPUT_ONLY'] == '1'
        example = jax.ShapeDtypeStruct((2,), jnp.float32)
        compile_program, arguments, shapes, memory = prepare_program(
            make_program(jnp.asarray(float(os.environ['OFFSET'])), int(os.environ['POWER'])),
            example if output_only else (template, example),
            static_tree=jax.tree.structure(template) if output_only else None,
        )
        executable = compile_program()
        value = jnp.ones((2,), dtype=jnp.float32)
        actual = executable(arguments, value if output_only else (template, value))
        if not output_only:
            executable(arguments, actual)
        # Returned static metadata must remain compatible with the live caller's model.
        jax.jit(lambda carry: jax.lax.cond(True, lambda: carry, lambda: (template, carry[1])))(actual)
        assert actual[0].evaluate_fn is template.evaluate_fn
        print(json.dumps({'value': actual[1].tolist(), 'shape': shapes[1].shape,
                          'bytes': memory.device_bytes}))
        """
    )
    results = []
    for offset, power, forbid_trace in ((1, 2, 0), (3, 2, 1), (3, 3, 0)):
        process = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
                "OFFSET": str(offset),
                "POWER": str(power),
                "FORBID_TRACE": str(forbid_trace),
                "OUTPUT_ONLY": str(int(output_only)),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout))
    assert [result["value"] for result in results] == [[4.0, 4.0], [16.0, 16.0], [64.0, 64.0]]
    assert all(result["shape"] == [2] and result["bytes"] > 0 for result in results)
    assert len(tuple(tmp_path.glob("nof1-shapes-*-cache"))) == 2
    assert len(tuple(tmp_path.glob("nof1-program-*-cache"))) == 2


def test_staged_key_keeps_input_tree_identity_with_identical_leaf_layouts(tmp_path):
    from nof1_causal_service.compute.compilation_cache import prepare_program

    def identity(value):
        return value

    leaf = jax.ShapeDtypeStruct((2,), jnp.float32)
    with local_compilation_cache(tmp_path, commit=lambda: None) as working:
        for argument in ({"a": leaf}, {"b": leaf}, (leaf,), {"a": leaf}):
            prepare_program(identity, argument)
        assert len(tuple(working.glob("nof1-shapes-*-cache"))) == 3


def test_restored_staged_outputs_preserve_weak_scalar_types(tmp_path):
    from nof1_causal_service.compute.compilation_cache import compile_staged

    def increment(value):
        return value + 1

    scalar = jnp.asarray(2.0)
    with local_compilation_cache(tmp_path, commit=lambda: None):
        cold = compile_staged(increment, scalar).call(scalar)
    with local_compilation_cache(tmp_path, commit=lambda: None):
        warm = compile_staged(increment, scalar).call(scalar)
    assert jax.typeof(cold) == jax.typeof(warm) == jax.typeof(scalar)
    assert cold == warm == 3


def _file(root, name, content, modified):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, ns=(modified, modified))


def test_separate_selection_bounds_round_trip_and_unchanged_skip_commit(tmp_path):
    working = tmp_path / "working"
    policy = CachePolicy(
        staged=CacheLimit(files=2, bytes=9),
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
    persisted = persist_cache(archive, working, (), policy, "accelerator", commit)
    assert (persisted.files, persisted.bytes) == (4, 19)
    assert persisted.archive_bytes == archive.stat().st_size <= policy.archive_bytes("accelerator")
    assert min(persisted.copy_seconds, persisted.selection_seconds, persisted.commit_seconds) >= 0
    commit.assert_called_once_with()
    staged, transfer = stage_cache(archive, tmp_path / "fresh", policy, "accelerator")
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
    unchanged = persist_cache(archive, tmp_path / "fresh", staged, policy, "accelerator", commit)
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
            target.truncate(policy.archive_bytes("accelerator") + 1)
    with pytest.raises(ValueError, match="cache"):
        stage_cache(archive, tmp_path / "local", policy, "accelerator")
    assert not (tmp_path / "escape").exists()


def test_local_lifecycle_isolated_namespaces_failure_cleanup_and_accounting(tmp_path, monkeypatch):
    # Backend discovery is separate from this I/O contract.
    monkeypatch.setattr(
        "nof1_causal_service.compute.compilation_cache.cache_namespace", lambda _policy: "a" * 64
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
        ("stage", 0, 0),
        ("persist", 0, 0),
        ("persist", 1, 6),
    ]
    assert {key: jax.config.values[key] for key in before} == before
    reports.clear()
    first_path = working
    with local_compilation_cache(persistent, commit=commit, report=reports.append) as working:
        assert working == first_path
        assert (working / "compiled-cache").read_bytes() == b"binary"
    assert [(r.phase, r.files) for r in reports] == [
        ("stage", 0),
        ("stage", 1),
        ("persist", 0),
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
        "nof1_causal_service.compute.compilation_cache.cache_namespace", lambda _policy: "a" * 64
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


def test_topology_writers_keep_separate_binaries_and_share_staged_artifacts(tmp_path, monkeypatch):
    from nof1_causal_service.compute import compilation_cache as cache

    monkeypatch.setattr(cache, "cache_namespace", lambda _policy: "namespace")
    commit = Mock()
    for topology in ("a" * 64, "b" * 64):
        monkeypatch.setattr(cache, "accelerator_partition", lambda topology=topology: topology)
        with local_compilation_cache(tmp_path, commit=commit) as working:
            assert not (working / "compiled-cache").exists()
            assert not (working / KERNELS).exists()
            if topology.startswith("a"):
                _file(working, "nof1-program-shared-cache", b"staged", 1)
            else:
                assert (working / "nof1-program-shared-cache").read_bytes() == b"staged"
            _file(working, "compiled-cache", topology.encode(), 2)
            _file(working, KERNELS, topology.encode(), 2)
    assert commit.call_count == 2  # One volume commit per fit, not per archive.
    snapshots = {path.name: path.read_bytes() for path in (tmp_path / "namespace").iterdir()}
    assert set(snapshots) == {"shared.tar", "a" * 64 + ".tar", "b" * 64 + ".tar"}
    for topology in ("a" * 64, "b" * 64):
        monkeypatch.setattr(cache, "accelerator_partition", lambda topology=topology: topology)
        with local_compilation_cache(tmp_path, commit=commit) as working:
            assert (working / "compiled-cache").read_bytes() == topology.encode()
            assert (working / KERNELS).read_bytes() == topology.encode()
            assert (working / "nof1-program-shared-cache").read_bytes() == b"staged"
    assert commit.call_count == 2
    assert snapshots == {
        path.name: path.read_bytes() for path in (tmp_path / "namespace").iterdir()
    }
    archive_limit = 400 * 1024**2 + 256 * 1024 + 529 * 1024 + 10240
    assert (
        sum(FIT_CACHE_POLICY.archive_bytes(p) for p in ("shared", "accelerator")) <= archive_limit
    )


@pytest.mark.parametrize(
    ("partition", "name"),
    [("shared", "binary-cache"), ("accelerator", "nof1-program-export-cache")],
)
def test_bundle_rejects_entries_owned_by_other_partition(tmp_path, partition, name):
    archive = tmp_path / "wrong.tar"
    _bundle(
        archive,
        [{"path": name, "size": 3, "modified_ns": 1, "sha256": hashlib.sha256(b"abc").hexdigest()}],
    )
    with pytest.raises(ValueError, match="another partition"):
        stage_cache(archive, tmp_path / "working", FIT_CACHE_POLICY, partition)


def test_accelerator_partition_matches_jax_fingerprint_component(monkeypatch):
    from nof1_causal_service.compute import compilation_cache as cache

    devices = [object()]
    monkeypatch.setattr(cache.jax, "local_devices", lambda: devices)
    topology = SimpleNamespace(fingerprint=lambda: 123456789)
    describe = Mock(return_value=topology)
    monkeypatch.setattr(cache.xla_client, "get_topology_for_devices", describe)
    assert (
        cache.accelerator_partition() == hashlib.sha256((123456789).to_bytes(8, "big")).hexdigest()
    )
    describe.assert_called_once_with(devices)
