"""Locate explicitly seeded record trees without adding durable application indexes."""

import pygit2

from grassmoth.artifacts.identity import GitOid
from grassmoth_server.actions.contracts.identity import ArtifactId
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore


def artifact_revisions(store: ArtifactStore, artifact: ArtifactId) -> list[GitOid]:
    records = []
    for oid in store.repo.odb:
        obj = store.repo[oid]
        if isinstance(obj, pygit2.Tree) and "meta.json" in obj:
            record = store.read_record(str(oid))
            if record.artifact_id == artifact:
                records.append(record)
    return [record.revision for record in sorted(records, key=lambda r: (r.created_at, r.revision))]


def artifact_revision(workspace: str, artifact: ArtifactId, ordinal: int) -> GitOid:
    return artifact_revisions(ArtifactStore(workspace), artifact)[ordinal - 1]


def commit_id(workspace: str, seq: int) -> GitOid:
    return next(r.commit_id for r in StudyRepository(workspace).attempts() if r.record.seq == seq)
