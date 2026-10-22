"""Resolve continuation lineage without treating a fitted posterior as a new prior."""

from pathlib import Path

from grassmoth.public import FitOutput, GitOid
from grassmoth_server.actions.contracts.inputs import FitTarget, NewFitTarget
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied


def prior_fit(store: ArtifactStore, revision: GitOid) -> FitOutput:
    """Read an applied fit by its immutable journal revision."""
    repository = StudyRepository(store.workspace_id, repository_path=Path(store.repo.path))
    attempt = repository.record(revision).record.attempt
    if attempt.action != "fit" or not isinstance(attempt.outcome, Applied):
        raise StudyLookupError("Continuing requires a successful fit revision")
    result = store.read_result(attempt.outcome.result, FitOutput)
    if result.inference.evidence.checkpoint is None:
        raise StudyLookupError("The selected fit has no last retained draw to continue")
    return result


def fit_source(store: ArtifactStore, target: FitTarget[GitOid]) -> NewFitTarget[GitOid]:
    """Walk fit parents to the original model and data references."""
    repository = StudyRepository(store.workspace_id, repository_path=Path(store.repo.path))
    return repository.fit_source(target)
