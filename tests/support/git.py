"""Git identities for synthetic contracts and explicitly ordered test setup artifacts."""

from grassmoth.artifacts.identity import GitOid
from grassmoth_server.actions.contracts.identity import ArtifactId
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore


def git_oid(label: int) -> GitOid:
    return GitOid(f"{label:040x}")


def artifact_revisions(store: ArtifactStore, artifact: ArtifactId) -> list[GitOid]:
    """Order explicitly created setup trees; application reads never use a catalog."""
    prefix = f"refs/artifacts/{artifact}/"
    records = [
        store.read_meta(artifact, name.removeprefix(prefix))
        for name in store.repo.references
        if name.startswith(prefix)
    ]
    return [
        record.revision
        for record in sorted(records, key=lambda info: (info.created_at, info.revision))
    ]


def artifact_revision(workspace: str, artifact: ArtifactId, ordinal: int) -> GitOid:
    return artifact_revisions(ArtifactStore(workspace), artifact)[ordinal - 1]


def commit_id(workspace: str, seq: int) -> GitOid:
    repository = StudyRepository(workspace)
    record = repository.read_attempt(seq)
    assert record is not None
    return record.commit_id
