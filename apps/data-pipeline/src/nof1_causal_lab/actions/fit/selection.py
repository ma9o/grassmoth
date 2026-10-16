"""Resolve continuation lineage without treating a fitted posterior as a new prior."""

from pathlib import Path

from nof1_causal_lab.actions.contracts.bodies import FitOutput, FitTarget, NewFitTarget
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied


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
