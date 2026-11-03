"""Transfer a served fit into Git and load retained full evidence for continuation."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from grassmoth.public import FitOutput, Value
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_VOLUME,
    AnalysisCacheEntry,
    retained_fit_result,
    verified_fit_content,
)
from grassmoth_server.study.persistence.codec import unpack_result


class RemoteFitResult(Value):
    """Remote completion coordinates; the study retains only the full-result pointer."""

    full_fit: AnalysisCacheEntry
    transfer: AnalysisCacheEntry
    law_arrays: tuple[str, ...]


def download_fit_result(root: Path, entry: AnalysisCacheEntry) -> Path:
    """Stream once to local disk and verify every explicit read against the immutable pointer."""
    import modal

    path = root / "results" / (entry.sha256 + ".msgpack")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                modal.Volume.from_name(ANALYSIS_CACHE_VOLUME).read_file_into_fileobj(
                    "retained/results/" + entry.sha256 + ".msgpack", stream
                )
                stream.flush()
                verified_fit_content(temporary, entry)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return retained_fit_result(root, entry)


def load_full_fit(root: Path, entry: AnalysisCacheEntry) -> FitOutput:
    """Read verified complete remote evidence into a local continuation call."""
    return FitOutput.model_validate(unpack_result(download_fit_result(root, entry).read_bytes()))
