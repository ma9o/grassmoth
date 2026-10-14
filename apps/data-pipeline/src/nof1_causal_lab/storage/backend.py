"""Pluggable storage backend — local filesystem or Cloudflare R2.

Production (``DEPLOYMENT_ENV=production``) uses Cloudflare R2.
All other environments default to local filesystem.

Environment variables for R2::

    DEPLOYMENT_ENV=production
    R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID=...
    R2_SECRET_ACCESS_KEY=...
    R2_BUCKET=...
    R2_PREFIX=data              # key prefix inside bucket (default: "data")
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, TypedDict, overload

from pydantic import TypeAdapter

from nof1_causal_lab.json_types import JsonObject

if TYPE_CHECKING:
    from collections.abc import Iterator

    import fsspec
    import polars as pl
    from fsspec.spec import AbstractBufferedFile

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def is_remote() -> bool:
    """True when using Cloudflare R2 (remote) storage."""
    return os.getenv("DEPLOYMENT_ENV") == "production"


# ---------------------------------------------------------------------------
# Base URI & filesystem
# ---------------------------------------------------------------------------


def _find_local_data_dir() -> Path:
    """Find the repository ``data/`` directory by walking up from this file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data"
        if candidate.exists():
            return candidate
    return Path.cwd() / "data"


def get_base_uri() -> str:
    """Return the root URI for data storage.

    Local: absolute path like ``/Users/.../data``
    R2:    ``s3://<bucket>/<prefix>``
    """
    if is_remote():
        bucket = os.environ["R2_BUCKET"]
        prefix = os.getenv("R2_PREFIX", "data")
        return f"s3://{bucket}/{prefix}"
    return str(_find_local_data_dir())


@lru_cache(maxsize=1)
def get_fs() -> fsspec.AbstractFileSystem:
    """Return an fsspec filesystem for the active backend (cached)."""
    import fsspec as _fsspec

    if is_remote():
        remote: fsspec.AbstractFileSystem = _fsspec.filesystem(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            key=os.environ["R2_ACCESS_KEY_ID"],
            secret=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        return remote
    local: fsspec.AbstractFileSystem = _fsspec.filesystem("file")
    return local


def polars_storage_options() -> dict[str, str] | None:
    """Storage options dict for Polars cloud I/O, or *None* for local."""
    if not is_remote():
        return None
    return {
        "aws_endpoint_url": os.environ["R2_ENDPOINT_URL"],
        "aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "aws_region": "auto",
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def join(*parts: str) -> str:
    """Join path segments — works for both local paths and ``s3://`` URIs."""
    if not parts:
        return ""
    base = parts[0]
    for part in parts[1:]:
        base = f"{base.rstrip('/')}/{part.strip('/')}"
    return base


# ---------------------------------------------------------------------------
# I/O primitives
# ---------------------------------------------------------------------------


def exists(path: str) -> bool:
    """Check path existence through the configured local or remote storage backend."""
    if is_remote():
        present: bool = get_fs().exists(path)
        return present
    return Path(path).exists()


def makedirs(path: str) -> None:
    """Create directories. No-op for S3 (directories are implicit)."""
    if is_remote():
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def rm_tree(path: str) -> None:
    """Remove a directory tree or object prefix when it exists."""
    if is_remote():
        fs = get_fs()
        if fs.exists(path):
            fs.rm(path, recursive=True)
        return
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def rm_file(path: str) -> None:
    """Remove one file or object when it exists."""
    if is_remote():
        get_fs().rm(path)
        return
    Path(path).unlink(missing_ok=True)


def listdir(path: str) -> list[str]:
    """List entries in *path*. Returns full paths/URIs."""
    if is_remote():
        fs = get_fs()
        entries = fs.ls(path, detail=False) if fs.exists(path) else []
        return [f"s3://{e}" if not e.startswith("s3://") else e for e in entries]
    p = Path(path)
    if not p.is_dir():
        return []
    return [str(child) for child in p.iterdir()]


def walk_files(path: str) -> list[str]:
    """List every file below a directory or object prefix."""
    if is_remote():
        if not get_fs().exists(path):
            return []
        entries = get_fs().find(path, detail=False)
        return [f"s3://{entry}" if not entry.startswith("s3://") else entry for entry in entries]
    root = Path(path)
    if not root.is_dir():
        return []
    return [str(entry) for entry in root.rglob("*") if entry.is_file()]


@dataclass(frozen=True)
class FileInfo:
    """Backend-independent size and modification time for a stored file."""

    size: int
    modified_seconds: float


class _RemoteFileInfo(TypedDict):
    size: int
    LastModified: datetime


def file_info(path: str) -> FileInfo:
    """Normalize filesystem metadata at the storage boundary."""
    if is_remote():
        info = TypeAdapter(_RemoteFileInfo).validate_python(get_fs().info(path), strict=True)
        return FileInfo(size=info["size"], modified_seconds=info["LastModified"].timestamp())
    stat = Path(path).stat()
    return FileInfo(size=stat.st_size, modified_seconds=stat.st_mtime)


@overload
@contextmanager
def open_file(path: str, mode: Literal["rb", "wb", "ab"] = "rb") -> Iterator[IO[bytes]]: ...


@overload
@contextmanager
def open_file(path: str, mode: Literal["r", "w", "a"]) -> Iterator[IO[str]]: ...


@contextmanager
def open_file(path: str, mode: str = "rb") -> Iterator[IO[bytes] | IO[str] | AbstractBufferedFile]:
    """Open a file for reading or writing. Works for both local and remote."""
    if is_remote():
        with get_fs().open(path, mode) as f:
            yield f
    else:
        if "w" in mode:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open(mode) as f:
            yield f


def read_text(path: str) -> str:
    """Read text from the configured storage backend, decoding remote byte responses when needed."""
    if is_remote():
        with get_fs().open(path, "r") as f:
            payload: str | bytes = f.read()
            return payload.decode() if isinstance(payload, bytes) else payload
    return Path(path).read_text()


def write_text(path: str, content: str) -> None:
    """Write text through the configured backend, creating parent directories for local files."""
    if is_remote():
        with get_fs().open(path, "w") as f:
            f.write(content)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def read_json(path: str) -> JsonObject:
    """Read stored text and parse it as a JSON object."""
    return TypeAdapter[JsonObject](JsonObject).validate_json(read_text(path))


def read_parquet(path: str) -> pl.DataFrame:
    """Read a Polars DataFrame from a parquet path."""
    import polars as pl

    return pl.read_parquet(path, storage_options=polars_storage_options())
