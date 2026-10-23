"""Bounded cache bundles at the GPU boundary; compilation itself uses local disk."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, gettempdir
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

import jax
from jax.experimental.compilation_cache import compilation_cache
from jaxlib import xla_client
from pydantic import Field, TypeAdapter, field_validator

from grassmoth.public import PreparedProgram, Value
from grassmoth_server.compute.traced_memory import traced_memory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from jax.stages import Compiled

    from grassmoth.public import MemoryDemand


def prepare_program[Result](
    function: Callable[..., Result],
    *args: object,
    **kwargs: object,
) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
    """Trace with JAX, lifting numerical closure values into runtime operands.

    JAX alone owns executable identity and persistent cache lookup. Its traced
    graph supplies the used closure constants, output tree and sizing evidence;
    no Python functions, shapes or exported programs are stored separately.
    """
    traced = jax.jit(function).trace(*args, **kwargs)
    constants, converted = traced.closure_convert()
    arguments = cast("tuple[jax.Array, ...]", tuple(constants))
    shapes = cast("Result", traced.out_info)

    def invoke(values: tuple[jax.Array, ...], *inputs: object, **options: object) -> Result:
        return cast("Result", converted(list(values), *inputs, **options))

    program = jax.jit(invoke)
    return (
        lambda: program.lower(arguments, *args, **kwargs).compile(),
        arguments,
        shapes,
        traced_memory(traced.jaxpr),
    )


def compile_program[Result](
    function: Callable[..., Result], *args: object, **kwargs: object
) -> PreparedProgram[Result]:
    """Bind numerical operands to an executable cached by JAX."""
    compile_function, operands, _, _ = prepare_program(function, *args, **kwargs)
    executable = compile_function()

    def call(*inputs: object, **options: object) -> Result:
        return cast("Result", executable(operands, *inputs, **options))

    return PreparedProgram(executable, call)


_AUTOTUNE = "xla_gpu_per_fusion_autotune_cache_dir"
_KERNELS = "xla_gpu_kernel_cache_file"
_LOGGER = logging.getLogger(__name__)
_CACHE_LOCK = Lock()

type CacheFamily = Literal["executable", "autotune", "kernels"]


def accelerator_partition() -> str:
    """Use the same topology fingerprint bytes as JAX's accelerator_config key."""
    topology = xla_client.get_topology_for_devices(jax.local_devices())
    return hashlib.sha256(topology.fingerprint().to_bytes(8, "big")).hexdigest()


_MANIFEST_LIMIT = 128 * 1024


@contextmanager
def _local_directory() -> Iterator[Path]:
    """Serialize the process-global JAX configuration and its stable local path.

    XLA's kernel-cache filename participates in the executable key. A random
    temporary directory therefore invalidates otherwise identical programs.
    The lock lives outside the cleared directory and also excludes other local
    processes; interrupted runs cannot contribute uncommitted cache files.
    """
    root = Path(gettempdir()) / f"grassmoth-fit-jax-{os.getuid()}"
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
    """Bound transfers of JAX executables and XLA accelerator caches."""

    executables: CacheLimit
    autotune: CacheLimit
    kernels: CacheLimit

    def families(self) -> tuple[tuple[CacheFamily, CacheLimit], ...]:
        """Enumerate the cache files produced by JAX and XLA."""
        return (
            ("executable", self.executables),
            ("autotune", self.autotune),
            ("kernels", self.kernels),
        )

    def archive_bytes(self) -> int:
        """Bound this partition's payload, manifest, headers and tar padding."""
        limits = tuple(limit for _, limit in self.families())
        return (
            sum(limit.bytes for limit in limits)
            + _MANIFEST_LIMIT
            + 1024 * sum(limit.files for limit in limits)
            + 10240
        )


FIT_CACHE_MIN_COMPILE_SECONDS = 0.0
FIT_CACHE_POLICY = CachePolicy(
    executables=CacheLimit(files=384, bytes=192 * 1024**2),
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
        """Own canonical relative paths for the compiler cache families."""
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
    def family(self) -> CacheFamily:
        """Project the family already established by path parsing."""
        if self.path == _KERNELS:
            return "kernels"
        return "autotune" if self.path.startswith(_AUTOTUNE + "/") else "executable"


_ENTRIES = TypeAdapter(tuple[CacheEntry, ...])


@dataclass(frozen=True)
class CacheTransfer:
    """Boundary costs, including the commit required to expose persisted entries."""

    phase: Literal["stage", "persist"]
    partition: str
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


def _selected(working: Path, policy: CachePolicy) -> tuple[CacheEntry, ...]:
    entries: list[CacheEntry] = []
    for family, limit in policy.families():
        candidates = (
            (working / _AUTOTUNE).rglob("*")
            if family == "autotune"
            else (working / _KERNELS,)
            if family == "kernels"
            else working.glob("*-cache")
        )
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
    families = policy.families()
    for family, limit in families:
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
        return (), CacheTransfer("stage", archive_path.stem, 0, 0, 0, perf_counter() - start)
    archive_size = archive_path.stat().st_size
    if archive_size > policy.archive_bytes():
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
        archive_path.stem,
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
        return CacheTransfer("persist", archive_path.stem, 0, 0, 0, 0.0, selection_seconds)
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
        archive_path.stem,
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
    archive_path = persistent_root / f"{accelerator_partition()}.tar"
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
