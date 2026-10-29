"""Content-addressed remote fit outputs; the study remains the publication owner."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import Field

from grassmoth.public import Value
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
    """A call alias to verified MessagePack bytes produced by its remote fit."""

    call: AnalysisCall
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


def fit_profile_directory(root: Path, call: AnalysisCall) -> Path:
    """Call-scoped timing reports and compiled analysis beside cached fit bodies."""
    return root / "profiles" / call.identity


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def cache_fit_result(root: Path, call: AnalysisCall, output: bytes) -> AnalysisCacheEntry:
    """Cache the already encoded body atomically, without another numerical copy."""
    entry = AnalysisCacheEntry(
        call=call, sha256=hashlib.sha256(output).hexdigest(), size=len(output)
    )
    target = root / "results" / (entry.sha256 + ".msgpack")
    if not target.exists():
        _atomic_write(target, output)
    target.touch()
    _atomic_write(root / "calls" / (call.identity + ".json"), entry.model_dump_json().encode())
    return entry


def cached_fit_result(root: Path, call: AnalysisCall) -> Path:
    """Resolve a call alias and verify the complete cached file with bounded reads."""
    # Published bodies occupy a separate namespace, outside temporary-cache cleanup.
    retained = root / "retained"
    selected = retained if (retained / "calls" / (call.identity + ".json")).exists() else root
    entry = AnalysisCacheEntry.model_validate_json(
        (selected / "calls" / (call.identity + ".json")).read_bytes()
    )
    if entry.call != call:
        raise ValueError("Analysis cache call identity does not match")
    return retained_fit_result(selected, entry)


def retained_fit_result(root: Path, entry: AnalysisCacheEntry) -> Path:
    """Verify content against the immutable receipt, independent of mutable call aliases."""
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
    """Expire temporary cache data; receipt-backed scientific results are retained."""
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
