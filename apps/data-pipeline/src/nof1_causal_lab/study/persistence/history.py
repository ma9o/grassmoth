"""Git trees select artifacts; commits own action logs and the study journal."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.requests import DataDiffRequest, ModelDiffRequest
    from nof1_causal_lab.study.records import AttemptRecord

import pygit2

from nof1_causal_lab.actions.contracts.requests import (
    FitRequest,
    ScientificActionRequest,
    call_identity,
)
from nof1_causal_lab.artifacts.identity import CallId, GitOid
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.persistence.git_objects import open_repository, read_file, write_tree
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.state import ArtifactRecord, StudyState


class StudyRepository:
    """One bare repository per local study; Git owns revision identity and topology."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        """Open the workspace's study repository, optionally at an explicitly supplied local path."""
        self.workspace_id = workspace_id
        self.repo = open_repository(workspace_id, repository_path)
        self.path = Path(self.repo.path)

    def head(self) -> GitOid:
        """Return the exact commit currently published as the main study head."""
        return GitOid(str(self.repo.references["refs/heads/main"].target))

    def _commit(self, revision: str) -> pygit2.Commit:
        oid = pygit2.Oid(hex=revision)
        if oid not in self.repo:
            raise StudyLookupError(f"Unknown study revision: {revision}")
        obj = self.repo[oid]
        if obj.type != pygit2.GIT_OBJECT_COMMIT:
            raise StudyLookupError("Select a study commit")
        return obj.peel(pygit2.Commit)

    def read_file(self, revision: str, name: str) -> bytes:
        """Read a named file from an exact Git revision without changing the working selection."""
        return read_file(self.repo, revision, name)

    def state(self, revision: str) -> StudyState:
        """Reconstruct artifact selections and the most recent selected data history at a checkpoint."""
        tree = self._commit(revision).tree
        if "artifacts" not in tree:
            return StudyState()
        current = {}
        artifacts = tree["artifacts"].peel(pygit2.Tree)
        for artifact in artifacts:
            info = ArtifactRecord(
                **json.loads(read_file(self.repo, str(artifact.id), "meta.json")),
                revision=GitOid(str(artifact.id)),
            )
            if info.artifact_id != artifact.name:
                raise StudyLookupError("Artifact tree does not match its scientific identity")
            current[info.artifact_id] = info
        data = None
        for record in reversed(self.records(revision)):
            attempt = record.record.attempt
            if not isinstance(attempt.outcome, Applied):
                continue
            if attempt.action == "fit":
                assert isinstance(attempt.request, FitRequest)
                from nof1_causal_lab.actions.fit.selection import fit_source
                from nof1_causal_lab.study.persistence.store import ArtifactStore
                data = fit_source(ArtifactStore(self.workspace_id, repository_path=self.path), attempt.request.input.target).data_ref
                break
            if attempt.action == "prepare_data" and any(
                info.artifact_id == "panel" for info in attempt.outcome.effects.produced
            ):
                data = DataRef[GitOid, int](revision=record.commit_id, replicate_index=0)
                break
        return StudyState(current=current, data=data)

    def resolve(self, *, at: GitOid) -> GitOid:
        """Require an existing study commit whose retained attempt, if any, was applied successfully."""
        revision = at
        commit = self._commit(revision)
        if "logs" in commit.tree and not isinstance(
            self.record(revision).record.attempt.outcome, Applied
        ):
            raise StudyLookupError(f"Commit {revision} is an unsuccessful attempt")
        return revision

    def record(self, revision: str) -> StudyRevision:
        """Parse a retained attempt and pair it with its publication commit and parents."""
        commit = self._commit(revision)
        from nof1_causal_lab.study.records import AttemptRecord

        return StudyRevision(
            record=AttemptRecord.model_validate_json(
                read_file(self.repo, revision, "logs/attempt.json")
            ),
            commit_id=GitOid(str(commit.id)),
            parent_ids=tuple(GitOid(str(parent)) for parent in commit.parent_ids),
        )

    def records(self, revision: str) -> list[StudyRevision]:
        """Read only the selected Git ancestry, in parent-before-child order."""
        return [
            self.record(str(commit.id))
            for commit in self.repo.walk(
                pygit2.Oid(hex=revision),
                pygit2.enums.SortMode.TOPOLOGICAL | pygit2.enums.SortMode.REVERSE,
            )
            if "logs" in commit.tree
        ]

    def attempts(self) -> list[StudyRevision]:
        """Read all indexed attempts in numerical study-sequence order."""
        refs = (ref for ref in self.repo.references if ref.startswith("refs/attempts/"))
        return [
            self.record(str(self.repo.references[ref].target))
            for ref in sorted(refs, key=lambda ref: int(ref.rsplit("/", 1)[1]))
        ]

    def read_attempt(self, seq: int) -> StudyRevision | None:
        """Workflow retry lookup; attempt numbers are not scientific revision identities."""
        ref = f"refs/attempts/{seq}"
        return (
            self.record(str(self.repo.references[ref].target))
            if ref in self.repo.references
            else None
        )

    def latest_seq(self) -> int:
        """Return the highest indexed attempt sequence, or zero before any attempts are recorded."""
        return max(
            (
                int(ref.rsplit("/", 1)[1])
                for ref in self.repo.references
                if ref.startswith("refs/attempts/")
            ),
            default=0,
        )

    def saved_call(
        self, request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    ) -> StudyRevision | None:
        """Every published call is reusable, including rejected and raised outcomes."""
        return self.call(call_identity(request))

    def call(self, identity: CallId) -> StudyRevision | None:
        """Look up the published revision indexed by a call identity, or ``None`` if unsaved."""
        ref = f"refs/calls/{identity.removeprefix('call:')}"
        return (
            self.record(str(self.repo.references[ref].target))
            if ref in self.repo.references
            else None
        )

    def question(self) -> ArtifactRecord:
        """The study's immutable question, independent of its current scientific state."""
        for revision in self.attempts():
            attempt = revision.record.attempt
            if attempt.action == "edit_question" and isinstance(attempt.outcome, Applied):
                return next(
                    item
                    for item in attempt.outcome.effects.produced
                    if item.artifact_id == "question"
                )
        raise StudyLookupError("Set the study question first")

    def append(
        self,
        record: AttemptRecord,
        *,
        parent_id: str | None = None,
        logs: dict[str, bytes] | None = None,
    ) -> StudyRevision:
        """Atomically publish the action's tree and advance successful scientific state."""
        outcome = record.attempt.outcome
        advances = isinstance(outcome, Applied) and record.attempt.action not in {
            "data_diff",
            "model_diff",
        }
        attempt_ref = f"refs/attempts/{record.seq}"
        head_ref = "refs/heads/main"
        log = record.model_dump(mode="json", round_trip=True)
        array_ids: tuple[str, ...] = ()
        with self.repo.transaction() as transaction:
            transaction.lock_ref(attempt_ref)
            call_ref = None
            if record.attempt.request is not None:
                identity = call_identity(record.attempt.request)
                call_ref = f"refs/calls/{identity.removeprefix('call:')}"
                transaction.lock_ref(call_ref)
                saved = self.call(identity)
                if saved is not None and saved.record.seq != record.seq:
                    return saved
            action_ref = (
                f"refs/actions/{record.attempt_id}" if record.attempt_id is not None else None
            )
            if action_ref is not None:
                transaction.lock_ref(action_ref)
                if (
                    action_ref in self.repo.references
                    and self.record(str(self.repo.references[action_ref].target)).record.seq
                    != record.seq
                ):
                    raise FileExistsError(
                        f"Action {record.attempt_id} already has a different attempt"
                    )
            existing = self.read_attempt(record.seq)
            if existing is not None:
                if existing.record.model_dump(mode="json", round_trip=True) != log:
                    raise FileExistsError(f"Attempt {record.seq} already has different content")
                if parent_id is not None and existing.parent_ids != (parent_id,):
                    raise FileExistsError(f"Attempt {record.seq} has a different journal parent")
                return existing
            transaction.lock_ref(head_ref)
            head = self.head()
            parent = parent_id if parent_id is not None else head
            if advances and parent != head:
                raise RuntimeError("The serialized study writer lost its journal parent")
            previous = self._commit(parent).tree
            artifacts = (
                self.repo.TreeBuilder(previous["artifacts"].id)
                if "artifacts" in previous
                else self.repo.TreeBuilder()
            )
            if advances and isinstance(outcome, Applied):
                for item in outcome.effects.retracted:
                    if artifacts.get(item.artifact_id) is not None:  # pyright: ignore[reportUnnecessaryComparison] - pygit2 documents None for a missing entry, but its stub returns Object.
                        artifacts.remove(item.artifact_id)
                for item in outcome.effects.produced:
                    artifacts.insert(
                        item.artifact_id, pygit2.Oid(hex=item.revision), pygit2.GIT_FILEMODE_TREE
                    )
            tree = self.repo.TreeBuilder()
            tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
            log_files = {
                "attempt.json": json.dumps(log, sort_keys=True).encode(),
                **(logs or {}),
            }
            from nof1_causal_lab.study.persistence.log_codec import assemble_call_log

            log_files["messages.json"] = (
                assemble_call_log(
                    record.messages,
                    logs or {},
                    at=datetime.fromisoformat(record.ts),
                    failure=None if isinstance(outcome, Applied) else outcome,
                )
                .model_dump_json(round_trip=True)
                .encode()
            )
            log_tree = self.repo.TreeBuilder(write_tree(self.repo, log_files))
            if isinstance(outcome, Applied):
                from nof1_causal_lab.study.persistence.arrays import owned_arrays
                from nof1_causal_lab.study.persistence.codec import unpack_result

                tree.insert(
                    "result.msgpack", pygit2.Oid(hex=outcome.result), pygit2.GIT_FILEMODE_BLOB
                )
                result = unpack_result(
                    self.repo[pygit2.Oid(hex=outcome.result)].peel(pygit2.Blob).data
                )
                array_ids = tuple(owned_arrays(result))
                for identity in array_ids:
                    array_ref = f"refs/arrays/{identity}"
                    transaction.lock_ref(array_ref)
                    if array_ref not in self.repo.references:
                        transaction.set_target(array_ref, pygit2.Oid(hex=outcome.result))
            tree.insert("logs", log_tree.write(), pygit2.GIT_FILEMODE_TREE)
            timestamp = int(datetime.fromisoformat(record.ts).timestamp())
            signature = pygit2.Signature("nof1-causal-lab", "study@local", timestamp, 0)
            action = record.attempt.action
            oid = self.repo.create_commit(
                None,
                signature,
                signature,
                f"{action} ({outcome.status})",
                tree.write(),
                [pygit2.Oid(hex=parent)],
            )
            transaction.set_target(attempt_ref, oid)
            if call_ref is not None:
                transaction.set_target(call_ref, oid)
            if action_ref is not None:
                transaction.set_target(action_ref, oid)
            if advances:
                transaction.set_target(head_ref, oid, message=f"{action} ({outcome.status})")
        if isinstance(outcome, Applied):
            from nof1_causal_lab.storage import paths as data_module

            # Publication now owns these buffers; execution copies are expendable.
            arrays = Path(data_module.store_dir(self.workspace_id)) / "arrays"
            for identity in array_ids:
                path = arrays / f"{identity}.npy"
                if path.exists():
                    path.unlink()
        return StudyRevision(
            commit_id=GitOid(str(oid)), parent_ids=(GitOid(str(parent)),), record=record
        )
