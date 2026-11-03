"""Content-addressed remote fit outputs; the study remains the publication owner."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from pydantic import Field

from grassmoth.public import SamplerSpec, Value
from grassmoth_server.actions.contracts.identity import CallId

# Every stack shares this Volume; only temporary data is collected by the daily job.
ANALYSIS_CACHE_VOLUME = "grassmoth-fit-analysis"
ANALYSIS_CACHE_ROOT = Path("/analysis-cache")
ANALYSIS_RETENTION_SECONDS = 14 * 24 * 60 * 60


class AnalysisCall(Value):
    """Published-call lookup coordinates, independent of the cached content identity."""

    workspace_id: str
    call_id: CallId

    @property
    def identity(self) -> str:
        """Keep workspace and request names out of filesystem paths."""
        return hashlib.sha256((self.workspace_id + "\0" + self.call_id).encode()).hexdigest()


class AnalysisCacheEntry(Value):
    """An immutable pointer to verified MessagePack bytes produced by its remote fit."""

    call: AnalysisCall
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


class RetainedFit(Value):
    """Full fit location and checkpoint settings; absent settings mean no retained draw."""

    entry: AnalysisCacheEntry
    settings: SamplerSpec | None


def fit_profile_directory(root: Path, call: AnalysisCall) -> Path:
    """Call-scoped timing reports and compiled analysis beside cached fit bodies."""
    return root / "profiles" / call.identity


def cache_fit_result(
    root: Path, call: AnalysisCall, output: Mapping[str, object]
) -> AnalysisCacheEntry:
    """Stream an atomic cache body without an additional full encoded copy."""
    from grassmoth_server.study.persistence.codec import write_payload

    directory = root / "results"
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write_payload(output, stream.write)
            stream.flush()
            with temporary.open("rb") as saved:
                digest = hashlib.file_digest(saved, "sha256").hexdigest()
            entry = AnalysisCacheEntry(call=call, sha256=digest, size=temporary.stat().st_size)
            target = directory / (entry.sha256 + ".msgpack")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return entry


def retained_fit_result(root: Path, entry: AnalysisCacheEntry) -> Path:
    """Verify content against the immutable full-result pointer."""
    path = root / "results" / (entry.sha256 + ".msgpack")
    return verified_fit_content(path, entry)


def verified_fit_content(path: Path, entry: AnalysisCacheEntry) -> Path:
    """Own byte-length and digest verification for mounted and downloaded bodies."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != entry.size or digest != entry.sha256:
        raise ValueError("Analysis cache content does not match its identity")
    return path


def expire_analysis_cache(root: Path, *, now_seconds: float) -> tuple[int, int]:
    """Expire temporary cache data; pointer-backed scientific results are retained."""
    cutoff = now_seconds - ANALYSIS_RETENTION_SECONDS
    files = size = 0
    for profile in (root / "profiles").glob("*"):
        if profile.stat().st_mtime < cutoff:
            artifacts = tuple(path for path in profile.rglob("*") if path.is_file())
            files += len(artifacts)
            size += sum(path.stat().st_size for path in artifacts)
            shutil.rmtree(profile)
    for folder, suffix in (("calls", "*.json"), ("results", "*.msgpack")):
        for path in (
            *tuple((root / folder).glob(suffix)),
            *tuple((root / folder).glob(".pending-*")),
        ):
            stat = path.stat()
            if stat.st_mtime < cutoff:
                path.unlink()
                files += 1
                size += stat.st_size
    return files, size


def cleanup_analysis_cache() -> tuple[int, int]:
    """Run the daily retention policy against the mounted cache Volume."""
    import modal

    removed = expire_analysis_cache(ANALYSIS_CACHE_ROOT, now_seconds=time.time())
    modal.Volume.from_name(ANALYSIS_CACHE_VOLUME).commit()
    return removed
