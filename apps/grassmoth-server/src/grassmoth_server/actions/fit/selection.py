"""Resolve continuation lineage without treating a fitted posterior as a new prior."""

from pathlib import Path

from grassmoth.public import GitOid, SamplerSpec, Value
from grassmoth_server.actions.contracts.inputs import FitTarget, NewFitTarget
from grassmoth_server.compute.analysis_cache import AnalysisCacheEntry
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.call import SavedApplied
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Rejected


class FitContinuation(Value):
    """A full retained fit and the sampler settings needed to continue it."""

    full_fit: AnalysisCacheEntry
    settings: SamplerSpec

    @property
    def retained_host_bytes(self) -> int:
        """Reserve the cached body's decode overlap without guessing its old controls.

        codec.unpack_result can hold the entire wire payload, decoded binary
        chunks, and joined large-buffer restorations simultaneously: three times
        the uncompressed shared-buffer MessagePack size bounds those byte stores.
        NumericalArray.values views restored bytes without copying. Keep this
        conservative loading reserve throughout initialization and sampling.
        """
        return 3 * self.full_fit.size


def prior_fit(store: ArtifactStore, revision: GitOid) -> FitContinuation | Rejected:
    """Select only fits with retained full evidence for continuation."""
    repository = StudyRepository(store.workspace_id, repository_path=Path(store.repo.path))
    call = repository.saved_record(revision)
    outcome = call.outcome
    if call.request.action != "fit" or not isinstance(outcome, SavedApplied):
        raise StudyLookupError("Continuing requires a successful fit revision")
    if outcome.result.full_fit is None:
        return Rejected(
            code="LOCAL_FIT_CONTINUATION_UNAVAILABLE",
            subject="inputs",
            detail="Local fits retain only their served result and cannot be continued. Start a new fit.",
        )
    full_fit = outcome.result.full_fit
    if full_fit.settings is None:
        return Rejected(
            code="FIT_CONTINUATION_UNAVAILABLE",
            subject="inputs",
            detail="The selected fit has no last retained draw to continue",
        )
    return FitContinuation(full_fit=full_fit.entry, settings=full_fit.settings)


def fit_source(store: ArtifactStore, target: FitTarget[GitOid]) -> NewFitTarget[GitOid]:
    """Walk fit parents to the original model and data references."""
    repository = StudyRepository(store.workspace_id, repository_path=Path(store.repo.path))
    return repository.fit_source(target)
