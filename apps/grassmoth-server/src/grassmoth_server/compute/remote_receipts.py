"""Compact publication records for complete fit bodies retained on the Modal Volume."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal

from grassmoth.public import FitCheckReport, FitOutput, InferenceReport, SamplerSpec, Value
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_VOLUME,
    AnalysisCacheEntry,
    retained_fit_result,
    verified_fit_content,
)
from grassmoth_server.study.persistence.codec import unpack_result

if TYPE_CHECKING:
    from grassmoth.public import GitOid
    from grassmoth_server.study.persistence.store import ArtifactStore


class RemoteFitReceipt(Value):
    """Publication reports and the immutable identity of their complete remote evidence."""

    kind: Literal["remote_fit"] = "remote_fit"  # noqa: FIELD002 -- Storage discriminator is read from the decoded mapping before schema selection.
    cache: AnalysisCacheEntry
    checks: FitCheckReport
    inference: InferenceReport
    continuation: SamplerSpec | None
    has_model: bool
    array_ids: tuple[str, ...]


def receipt_for(
    result: FitOutput, entry: AnalysisCacheEntry, array_ids: tuple[str, ...]
) -> RemoteFitReceipt:
    """Keep native checks and convergence reports; numerical evidence stays on the Volume."""
    checkpoint = result.inference.evidence.checkpoint
    return RemoteFitReceipt(
        cache=entry,
        checks=result.checks,
        continuation=None if checkpoint is None else checkpoint.settings,
        has_model=result.dynamical_model_spec is not None,
        array_ids=array_ids,
        inference=result.inference.revised(
            evidence=type(result.inference.evidence)(),
            detail=type(result.inference.detail)(),
            core=result.inference.core.revised(
                inference_metadata=result.inference.core.inference_metadata.revised(
                    sampler_diagnostics=None
                ),
                posterior_marginals=(),
                prior_densities={},
                frequency_refresh=(),
                regional_movement=(),
                loo_diagnostics=None,
            ),
        ),
    )


def download_fit_result(root: Path, entry: AnalysisCacheEntry) -> Path:
    """Stream once to local disk and verify every explicit read against the receipt."""
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


def complete_remote_body(payload: bytes, root: Path) -> bytes:
    """Resolve a storage receipt only for an explicit complete-result read."""
    decoded = unpack_result(payload)
    if decoded.get("kind") != "remote_fit":
        return payload
    receipt = RemoteFitReceipt.model_validate(decoded)
    return download_fit_result(root, receipt.cache).read_bytes()


def load_remote_fit(root: Path, entry: AnalysisCacheEntry) -> FitOutput:
    """Parse verified remote evidence at the storage edge for local continuation."""
    return FitOutput.model_validate(unpack_result(download_fit_result(root, entry).read_bytes()))


def published_fit_report(store: ArtifactStore, revision: GitOid) -> FitOutput | RemoteFitReceipt:
    """Read publication reports without retrieving remote numerical evidence."""
    import pygit2

    from grassmoth_server.study.persistence.git_objects import read_blob

    payload = unpack_result(read_blob(store.repo, pygit2.Oid(hex=revision)))
    if payload.get("kind") == "remote_fit":
        return RemoteFitReceipt.model_validate(payload)
    return FitOutput.model_validate(payload)
