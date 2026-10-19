"""Bounded cache bundles at the GPU boundary; compilation itself uses local disk."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import pickle
import platform
import shutil
import tarfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, gettempdir
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

import cloudpickle
import jax
from jax import export
from jax.core import Tracer
from jax.experimental.compilation_cache import compilation_cache
from pydantic import Field, TypeAdapter, field_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.compute.traced_memory import traced_memory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.compute.budget import MemoryDemand


def numerical_source_digest() -> str:
    """Identify the shipped numerical source, including uncommitted edits."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _program_structure(
    function: object, memo: Mapping[int, tuple[int, object]]
) -> tuple[bytes, tuple[jax.Array, ...]]:
    """Extract runtime arrays without tracing; NumPy structural masks stay static."""
    arrays: list[jax.Array] = []
    indices: dict[int, int] = {}

    def array_id(
        value: object,
    ) -> int | tuple[int, tuple[int, ...], str, bool] | jax.ShapeDtypeStruct | None:
        if id(value) in memo:
            return memo[id(value)][0]
        if isinstance(value, Tracer):
            # NumPyro retains unused initialization closures. Their expired
            # symbolic values have no runtime buffers. Preserve the abstract
            # value: using it in a live computation must still fail tracing.
            return jax.ShapeDtypeStruct(
                value.shape, value.dtype, weak_type=jax.typeof(value).weak_type
            )
        if isinstance(value, jax.Array):
            identity = id(value)
            if identity not in indices:
                indices[identity] = len(arrays)
                arrays.append(value)
            return (
                indices[identity],
                value.shape,
                str(value.dtype),
                bool(jax.typeof(value).weak_type),
            )
        return None

    buffer = io.BytesIO()
    pickler = cloudpickle.CloudPickler(buffer, protocol=5)
    pickler.persistent_id = array_id  # ty: ignore[invalid-assignment] # noqa: V101 - pickle supports per-instance callbacks, consumed internally.
    pickler.dump(function)
    return buffer.getvalue(), tuple(arrays)


def _program_key(structure: bytes, examples: object) -> str:
    """Scope staged programs to source, dependencies, compiler settings and layout."""
    leaves, tree = jax.tree.flatten(examples)
    layouts = tuple(
        (value.shape, str(value.dtype), bool(jax.typeof(value).weak_type)) for value in leaves
    )
    identity = (
        numerical_source_digest(),
        platform.python_version(),
        tuple(
            (name, version(name))
            for name in (
                "jax",
                "jaxlib",
                "numpy",
                "numpyro",
                "equinox",
                "dynestyx",
                "blackjax",
                "diffrax",
                "scipy",
                "cloudpickle",
                "flatbuffers",
            )
        ),
        cache_namespace(FIT_CACHE_POLICY),
        tuple(
            sorted(
                (name, str(value))
                for name, value in jax.config.values.items()
                if name != "jax_compilation_cache_dir"
            )
        ),
        layouts,
        tree,
    )
    digest = hashlib.sha256(structure)
    digest.update(cloudpickle.dumps(identity, protocol=5))
    return digest.hexdigest()


def _write_program(path: Path, data: bytes) -> None:
    """Publish a complete staged artifact under the existing bounded cache policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        try:
            target.write(data)
            target.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_program[Result](
    function: Callable[..., Result], *args: object, **kwargs: object
) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
    """Reuse an exported program and sizing metadata before any model tracing.

    Only locally constructed closures are unpickled for binding. Persistent
    artifacts share the executable cache's trusted directory, bounded transfer,
    source/version identity and atomic publication. Numerical JAX arrays are
    operands; structural Python values and NumPy masks participate in the key.
    """
    # Seed both serializers with the caller's static PyTree objects. In
    # particular, a carried Equinox expression and the rebuilt transition must
    # refer to the same static evaluator function. Fresh-process cache loads
    # resolve these references against that process's constructed input tree.
    static_buffer = io.BytesIO()
    static_pickler = cloudpickle.CloudPickler(static_buffer, protocol=5)
    static_pickler.dump(jax.tree.structure((args, kwargs)))
    static_memo = cast("dict[int, tuple[int, object]]", static_pickler.memo.copy())
    static_values = dict(static_memo.values())
    structure, arguments = _program_structure(function, static_memo)
    key = _program_key(static_buffer.getvalue() + structure, (args, kwargs))
    directory = jax.config.jax_compilation_cache_dir
    working = Path(directory) if directory and jax.config.jax_enable_compilation_cache else None
    metadata_path = working / f"nof1-shapes-{key}-cache" if working is not None else None
    program_path = working / f"nof1-program-{key}-cache" if working is not None else None

    def invoke(values: tuple[jax.Array, ...], *inputs: object, **options: object) -> Result:
        loader = pickle.Unpickler(io.BytesIO(structure))

        def bind(
            token: int | tuple[int, tuple[int, ...], str, bool] | jax.ShapeDtypeStruct,
        ) -> object:
            if isinstance(token, int):
                return static_values[token]
            return token if isinstance(token, jax.ShapeDtypeStruct) else values[token[0]]

        loader.persistent_load = bind  # ty: ignore[invalid-assignment] # noqa: V101 - pickle supports per-instance callbacks, consumed internally.
        bound = cast("Callable[..., Result]", loader.load())
        return bound(*inputs, **options)

    program = jax.jit(invoke)
    flat_args, input_tree = jax.tree.flatten((arguments, args, kwargs))

    def flat_program(*values: jax.Array) -> tuple[jax.Array, ...]:
        runtime, inputs, options = jax.tree.unflatten(input_tree, values)
        return cast(
            "tuple[jax.Array, ...]", tuple(jax.tree.leaves(program(runtime, *inputs, **options)))
        )

    flattened = jax.jit(flat_program)
    hit = metadata_path is not None and metadata_path.exists()
    if hit:
        assert metadata_path is not None
        loader = pickle.Unpickler(io.BytesIO(metadata_path.read_bytes()))
        loader.persistent_load = static_values.__getitem__  # ty: ignore[invalid-assignment] # noqa: V101 - pickle supports per-instance callbacks, consumed internally.
        shapes, memory = cast("tuple[Result, MemoryDemand]", loader.load())
    else:
        shapes = cast("Result", jax.eval_shape(program, arguments, *args, **kwargs))
        memory = traced_memory(flattened.trace(*flat_args).jaxpr)
        if metadata_path is not None:
            metadata = io.BytesIO()
            pickler = cloudpickle.CloudPickler(metadata, protocol=5)

            def static_id(value: object) -> int | None:
                entry = static_memo.get(id(value))
                return entry[0] if entry is not None else None

            pickler.persistent_id = static_id  # ty: ignore[invalid-assignment] # noqa: V101 - pickle supports per-instance callbacks, consumed internally.
            pickler.dump((shapes, memory))
            _write_program(metadata_path, metadata.getvalue())
    _LOGGER.info("Staged program shapes %s: %s", "hit" if hit else "miss", key)
    output_tree = jax.tree.structure(shapes)

    def compile_program() -> Compiled:
        hit = program_path is not None and program_path.exists()
        if hit:
            assert program_path is not None
            exported = export.deserialize(bytearray(program_path.read_bytes()))
        else:
            exported = export.export(flattened)(*flat_args)
            if program_path is not None:
                _write_program(program_path, bytes(exported.serialize()))
        _LOGGER.info("Staged program %s: %s", "hit" if hit else "miss", key)

        def restored(values: tuple[jax.Array, ...], *inputs: object, **options: object) -> Result:
            leaves = jax.tree.leaves((values, inputs, options))
            return cast("Result", jax.tree.unflatten(output_tree, exported.call(*leaves)))

        return jax.jit(restored).lower(arguments, *args, **kwargs).compile()

    return compile_program, arguments, shapes, memory


_AUTOTUNE = "xla_gpu_per_fusion_autotune_cache_dir"
_KERNELS = "xla_gpu_kernel_cache_file"
_MANIFEST_LIMIT = 256 * 1024
_LOGGER = logging.getLogger(__name__)
_CACHE_LOCK = Lock()


@contextmanager
def _local_directory() -> Iterator[Path]:
    """Serialize the process-global JAX configuration and its stable local path.

    XLA's kernel-cache filename participates in the executable key. A random
    temporary directory therefore invalidates otherwise identical programs.
    The lock lives outside the cleared directory and also excludes other local
    processes; interrupted runs cannot contribute uncommitted cache files.
    """
    root = Path(gettempdir()) / f"nof1-fit-jax-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    with _CACHE_LOCK, (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        working = root / "working"
        if working.exists():
            shutil.rmtree(working)
        working.mkdir(mode=0o700)
        try:
            yield working
        finally:
            shutil.rmtree(working)
            fcntl.flock(lock, fcntl.LOCK_UN)


class CacheLimit(Value):
    """An independent count and byte allowance for one cache family."""

    files: int = Field(gt=0)
    bytes: int = Field(gt=0)


class CachePolicy(Value):
    """Separate bounded persistence for executables, autotuning and GPU kernels."""

    executables: CacheLimit
    autotune: CacheLimit
    kernels: CacheLimit

    @property
    def archive_bytes(self) -> int:
        """Bound payload plus manifest, tar headers, padding and end records."""
        return (
            self.executables.bytes
            + self.autotune.bytes
            + self.kernels.bytes
            + _MANIFEST_LIMIT
            + 1024 * (self.executables.files + self.autotune.files + self.kernels.files)
            + 10240
        )


FIT_CACHE_MIN_COMPILE_SECONDS = 1.0
FIT_CACHE_POLICY = CachePolicy(
    executables=CacheLimit(files=64, bytes=256 * 1024**2),
    autotune=CacheLimit(files=128, bytes=16 * 1024**2),
    kernels=CacheLimit(files=1, bytes=128 * 1024**2),
)


class CacheEntry(Value):
    """An owned regular cache file; paths never escape the working directory."""

    path: str
    size: int = Field(ge=0)
    modified_ns: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def parse_path(cls, path: str) -> str:
        """Own canonical relative paths for the three compiler cache families."""
        parts = PurePosixPath(path).parts
        executable = len(parts) == 1 and path.endswith("-cache")
        autotune = len(parts) > 1 and parts[0] == _AUTOTUNE
        if (
            not (executable or autotune or path == _KERNELS)
            or ".." in parts
            or PurePosixPath(path).is_absolute()
            or str(PurePosixPath(path)) != path
            or "\\" in path
            or len(path) > 200
        ):
            raise ValueError("Invalid compilation cache entry path")
        return path

    @property
    def family(self) -> Literal["executable", "autotune", "kernels"]:
        """Project the family already established by path parsing."""
        if self.path == _KERNELS:
            return "kernels"
        return "autotune" if self.path.startswith(_AUTOTUNE + "/") else "executable"


_ENTRIES = TypeAdapter(tuple[CacheEntry, ...])


@dataclass(frozen=True)
class CacheTransfer:
    """Boundary costs, including the commit required to expose persisted entries."""

    phase: Literal["stage", "persist"]
    files: int
    bytes: int
    archive_bytes: int
    copy_seconds: float
    selection_seconds: float = 0.0
    commit_seconds: float = 0.0


def _report(transfer: CacheTransfer) -> None:
    _LOGGER.info(
        "Fit compilation cache: %s",
        json.dumps(
            {
                **asdict(transfer),
                "total_seconds": transfer.copy_seconds
                + transfer.selection_seconds
                + transfer.commit_seconds,
            },
            sort_keys=True,
        ),
    )


def cache_namespace(policy: CachePolicy) -> str:
    """Isolate compiler/device/options; JAX's own keys still validate each executable."""
    identity = {
        "bundle_version": 1,
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "devices": tuple(
            (device.platform, device.device_kind, device.client.platform_version)
            for device in jax.local_devices()
        ),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "x64": jax.config.x64_enabled,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "policy": policy.model_dump(),
        "min_compile_seconds": FIT_CACHE_MIN_COMPILE_SECONDS,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _selected(working: Path, policy: CachePolicy) -> tuple[CacheEntry, ...]:
    entries: list[CacheEntry] = []
    for candidates, limit in (
        (working.glob("*-cache"), policy.executables),
        ((working / _AUTOTUNE).rglob("*"), policy.autotune),
        ((working / _KERNELS,), policy.kernels),
    ):
        files = sorted(
            (path for path in candidates if not path.is_symlink() and path.is_file()),
            key=lambda path: (-path.stat().st_mtime_ns, path.as_posix()),
        )
        count, size = 0, 0
        for path in files:
            stat = path.stat()
            if count == limit.files:
                break
            if size + stat.st_size > limit.bytes:
                continue
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            entries.append(
                CacheEntry(
                    path=path.relative_to(working).as_posix(),
                    size=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                    sha256=digest,
                )
            )
            count += 1
            size += stat.st_size
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _parse_manifest(data: bytes, policy: CachePolicy) -> tuple[CacheEntry, ...]:
    entries = _ENTRIES.validate_json(data)
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("Duplicate compilation cache entry")
    for family, limit in (
        ("executable", policy.executables),
        ("autotune", policy.autotune),
        ("kernels", policy.kernels),
    ):
        selected = tuple(entry for entry in entries if entry.family == family)
        if len(selected) > limit.files or sum(entry.size for entry in selected) > limit.bytes:
            raise ValueError("Compilation cache bundle exceeds its staging policy")
    return entries


def _member(archive: tarfile.TarFile, name: str, size_limit: int) -> tarfile.TarInfo:
    member = archive.next()
    if member is None or not member.isfile() or member.name != name or member.size > size_limit:
        raise ValueError("Invalid compilation cache bundle member")
    return member


def stage_cache(
    archive_path: Path, working: Path, policy: CachePolicy
) -> tuple[tuple[CacheEntry, ...], CacheTransfer]:
    """Read one bounded bundle, never inventory the persistent volume."""
    start = perf_counter()
    if not archive_path.exists():
        return (), CacheTransfer("stage", 0, 0, 0, perf_counter() - start)
    archive_size = archive_path.stat().st_size
    if archive_size > policy.archive_bytes:
        raise ValueError("Compilation cache archive exceeds its staging policy")
    with tarfile.open(archive_path, "r|") as archive:
        manifest = archive.extractfile(_member(archive, "manifest.json", _MANIFEST_LIMIT))
        assert manifest is not None
        entries = _parse_manifest(manifest.read(), policy)
        for index, entry in enumerate(entries):
            member = _member(archive, f"files/{index}", entry.size)
            if member.size != entry.size:
                raise ValueError("Compilation cache size differs from manifest")
            source = archive.extractfile(member)
            assert source is not None
            destination = working / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            with destination.open("rb") as staged:
                if hashlib.file_digest(staged, "sha256").hexdigest() != entry.sha256:
                    raise ValueError("Compilation cache digest differs from manifest")
            os.utime(destination, ns=(entry.modified_ns, entry.modified_ns))
        if archive.next() is not None:
            raise ValueError("Unexpected compilation cache bundle member")
    return entries, CacheTransfer(
        "stage",
        len(entries),
        sum(entry.size for entry in entries),
        archive_size,
        perf_counter() - start,
    )


def persist_cache(
    archive_path: Path,
    working: Path,
    previous: tuple[CacheEntry, ...],
    policy: CachePolicy,
    commit: Callable[[], None],
) -> CacheTransfer:
    """Publish selected entries atomically, then commit; unchanged bundles do no I/O."""
    start = perf_counter()
    entries = _selected(working, policy)
    selection_seconds = perf_counter() - start
    if entries == previous:
        return CacheTransfer("persist", 0, 0, 0, 0.0, selection_seconds)
    start = perf_counter()
    manifest = _ENTRIES.dump_json(entries)
    if len(manifest) > _MANIFEST_LIMIT:
        raise ValueError("Compilation cache manifest exceeds its staging policy")
    with NamedTemporaryFile() as local:
        with tarfile.open(fileobj=local, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            for index, entry in enumerate(entries):
                info = tarfile.TarInfo(f"files/{index}")
                info.size = entry.size
                with (working / entry.path).open("rb") as source:
                    archive.addfile(info, source)
        archive_size = local.tell()
        local.seek(0)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=archive_path.parent, delete=False) as remote:
            temporary = Path(remote.name)
            try:
                shutil.copyfileobj(local, remote)
                remote.flush()
                Path(temporary).replace(archive_path)
            finally:
                temporary.unlink(missing_ok=True)
    copy_seconds = perf_counter() - start
    start = perf_counter()
    commit()
    return CacheTransfer(
        "persist",
        len(entries),
        sum(entry.size for entry in entries),
        archive_size,
        copy_seconds,
        selection_seconds,
        perf_counter() - start,
    )


@contextmanager
def local_compilation_cache(
    persistent_root: Path,
    *,
    commit: Callable[[], None],
    policy: CachePolicy = FIT_CACHE_POLICY,
    report: Callable[[CacheTransfer], None] = _report,
) -> Iterator[Path]:
    """Stage before fitting and persist on success; restore JAX settings on every exit."""
    archive_path = persistent_root / f"{cache_namespace(policy)}.tar"
    with _local_directory() as working:
        previous, staged = stage_cache(archive_path, working, policy)
        report(staged)
        settings = {
            "jax_compilation_cache_dir": str(working),
            "jax_enable_compilation_cache": True,
            "jax_persistent_cache_min_compile_time_secs": FIT_CACHE_MIN_COMPILE_SECONDS,
            "jax_persistent_cache_min_entry_size_bytes": 0,
            "jax_persistent_cache_enable_xla_caches": f"{_AUTOTUNE},{_KERNELS}",
            "jax_compilation_cache_max_size": -1,
        }
        old = {name: jax.config.values[name] for name in settings}
        compilation_cache.reset_cache()
        try:
            for name, value in settings.items():
                jax.config.update(name, value)
            yield working
            report(persist_cache(archive_path, working, previous, policy, commit))
        finally:
            compilation_cache.reset_cache()
            for name, value in old.items():
                jax.config.update(name, value)
