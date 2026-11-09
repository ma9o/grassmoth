"""Git parents record pinned inputs; attempt sequence selects current scientific values."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import CallId
    from grassmoth_server.actions.contracts.requests import DataDiffRequest, ModelDiffRequest
    from grassmoth_server.study.records import AttemptRecord

import pygit2

from grassmoth.public import DataRef, GitOid
from grassmoth_server.actions.contracts.inputs import FitTarget, NewFitTarget
from grassmoth_server.actions.contracts.logs import CallLogSnapshot
from grassmoth_server.actions.contracts.requests import (
    FitRequest,
    ScientificActionRequest,
    call_identity,
)
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.call import SavedApplied, SavedCall, SavedResult
from grassmoth_server.study.persistence.git_objects import (
    open_repository,
    read_file,
    write_blob,
    write_directory,
)
from grassmoth_server.study.persistence.sources import capture_sources
from grassmoth_server.study.records import (
    ActionMessage,
    Applied,
    Rejected,
    StudyRevision,
    action_attempt,
    argument_parents,
    recorded_question,
)
from grassmoth_server.study.state import ArtifactRecord, StudyState, apply_effects

_EMPTY_LOG_SNAPSHOT = CallLogSnapshot()


class StudyRepository:
    """One bare repository per local study; Git owns revision identity and topology."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        """Open the workspace's study repository, optionally at an explicitly supplied local path."""
        self.workspace_id = workspace_id
        self.repo = open_repository(workspace_id, repository_path)
        self.path = Path(self.repo.path)

    def head(self) -> GitOid:
        """Return the newest applied scientific attempt in submission order."""
        for revision in reversed(self.attempts()):
            attempt = revision.record.attempt
            if isinstance(attempt.outcome, Applied) and attempt.action not in {
                "data_diff",
                "model_diff",
            }:
                return revision.commit_id
        raise StudyLookupError("The study has no applied scientific attempts")

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

    def state(self, revision: str | None = None) -> StudyState:
        """Fold published effects and selected data in submission order, optionally through a revision."""
        state = StudyState()
        for record in self.records(revision):
            attempt = record.record.attempt
            if not isinstance(attempt.outcome, Applied) or attempt.action in {
                "data_diff",
                "model_diff",
            }:
                continue
            state = apply_effects(
                state, attempt.outcome.effects.produced, attempt.outcome.effects.retracted
            )
            if attempt.action == "fit":
                assert isinstance(attempt.request, FitRequest)
                state = state.revised(data=self.fit_source(attempt.request.input.target).data_ref)
            elif attempt.action == "prepare_data" and any(
                info.artifact_id == "panel" for info in attempt.outcome.effects.produced
            ):
                state = state.revised(
                    data=DataRef[GitOid, int](revision=record.commit_id, replicate_index=0)
                )
        return state

    def fit_source(self, target: FitTarget[GitOid]) -> NewFitTarget[GitOid]:
        """Walk successful fit revisions to the original model and data inputs."""
        from typing import cast

        while not isinstance(target, NewFitTarget):
            attempt = self.record(target.fit_ref).record.attempt
            if attempt.action != "fit" or not isinstance(attempt.outcome, Applied):
                raise StudyLookupError("Continuing requires a successful fit revision")
            assert attempt.request is not None
            target = attempt.request.input.target
        return cast("NewFitTarget[GitOid]", target)

    def resolve(self, *, at: GitOid) -> GitOid:
        """Require an existing study commit whose retained attempt, if any, was applied successfully."""
        revision = at
        commit = self._commit(revision)
        if "call.json" in commit.tree and not isinstance(
            self.record(revision).record.attempt.outcome, Applied
        ):
            raise StudyLookupError(f"Commit {revision} is an unsuccessful attempt")
        return revision

    def saved_record(self, revision: str) -> SavedCall:
        """Read storage-only metadata without loading any scientific body."""
        return SavedCall.model_validate_json(self.read_file(revision, "call.json"))

    def record(self, revision: str) -> StudyRevision:
        """Reconstruct public effects from real record trees in their retained order."""
        from grassmoth_server.actions.contracts.effects import ActionEffects
        from grassmoth_server.study.persistence.store import ArtifactStore
        from grassmoth_server.study.records import AttemptRecord

        commit = self._commit(revision)
        saved = self.saved_record(revision)
        outcome = saved.outcome
        store = ArtifactStore(self.workspace_id, repository_path=self.path)
        public_outcome = (
            Applied(
                result=outcome.result.revision,
                effects=ActionEffects(
                    produced=tuple(
                        store.read_meta(kind, str(commit.tree[f"records/{kind}"].id))
                        for kind in outcome.produced
                    ),
                    retracted=outcome.retracted,
                    reports=outcome.reports,
                ),
            )
            if isinstance(outcome, SavedApplied)
            else outcome
        )
        return StudyRevision(
            record=AttemptRecord(
                seq=saved.seq,
                attempt_id=saved.attempt_id,
                ts=saved.ts,
                attempt=action_attempt(saved.request, public_outcome),
            ),
            commit_id=GitOid(str(commit.id)),
            parent_ids=tuple(GitOid(str(parent)) for parent in commit.parent_ids),
        )

    def records(self, revision: str | None = None) -> list[StudyRevision]:
        """Read indexed attempts in submission order, optionally through one attempt's sequence."""
        if revision is None:
            return self.attempts()
        if "call.json" not in self._commit(revision).tree:
            return []
        seq = self.record(revision).record.seq
        return [item for item in self.attempts() if item.record.seq <= seq]

    def attempts(self) -> list[StudyRevision]:
        """Read every retained call in numerical submission order."""
        return sorted(
            (
                self.record(str(self.repo.references[ref].target))
                for ref in self.repo.references
                if ref.startswith("refs/calls/")
            ),
            key=lambda revision: revision.record.seq,
        )

    def latest_seq(self) -> int:
        """Return the greatest saved sequence, including rejected and raised calls."""
        return max((revision.record.seq for revision in self.attempts()), default=0)

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
        question = recorded_question(self.attempts())
        if question is None:
            raise StudyLookupError("Set the study question first")
        return question

    def append(
        self,
        record: AttemptRecord,
        *,
        messages: tuple[ActionMessage, ...] = (),
        snapshot: CallLogSnapshot = _EMPTY_LOG_SNAPSHOT,
    ) -> StudyRevision:
        """Atomically publish a call and its law owners, admitting one immutable question."""
        from grassmoth_server.storage import paths
        from grassmoth_server.study.persistence.log_codec import finalize_call_log

        request = record.attempt.request
        if request is None:
            raise ValueError("Saved calls require their request")
        identity = call_identity(request)
        call_ref = f"refs/calls/{identity.removeprefix('call:')}"
        directory = Path(paths.call_dir(self.workspace_id, identity))
        outcome = record.attempt.outcome
        with self.repo.transaction() as transaction:
            transaction.lock_ref(call_ref)
            existing = self.call(identity)
            if existing is not None:
                return existing
            question_ref = None
            if record.attempt.action == "edit_question" and isinstance(outcome, Applied):
                transaction.lock_ref("refs/question")
                if "refs/question" in self.repo.references:
                    outcome = Rejected(
                        code="INPUT_UNAVAILABLE",
                        subject="inputs",
                        detail="The question is set by the study's first action",
                    )
                    record = record.revised(
                        attempt=record.attempt.revised(outcome=outcome),
                    )
                else:
                    question_ref = "refs/question"
            tree = self.repo.TreeBuilder()
            if isinstance(outcome, Applied):
                saved_result = SavedResult.model_validate_json(
                    (directory / "result.json").read_bytes()
                )
                if saved_result.revision != outcome.result:
                    raise ValueError("The call's encoded result does not match its publication")
                saved_outcome = SavedApplied(
                    result=saved_result,
                    produced=tuple(info.artifact_id for info in outcome.effects.produced),
                    retracted=outcome.effects.retracted,
                    reports=outcome.effects.reports,
                )
                tree.insert(
                    "result.msgpack", pygit2.Oid(hex=outcome.result), pygit2.GIT_FILEMODE_BLOB
                )
                records = self.repo.TreeBuilder()
                for info in outcome.effects.produced:
                    records.insert(
                        info.artifact_id, pygit2.Oid(hex=info.revision), pygit2.GIT_FILEMODE_TREE
                    )
                tree.insert("records", records.write(), pygit2.GIT_FILEMODE_TREE)
                if outcome.effects.reports:
                    reports = self.repo.TreeBuilder()
                    for name, result in outcome.effects.reports.items():
                        reports.insert(name, pygit2.Oid(hex=result), pygit2.GIT_FILEMODE_BLOB)
                    tree.insert("reports", reports.write(), pygit2.GIT_FILEMODE_TREE)
                for array_id in sorted(saved_result.law_arrays):
                    array_ref = f"refs/arrays/{array_id}"
                    transaction.lock_ref(array_ref)
                    if array_ref not in self.repo.references:
                        transaction.set_target(array_ref, pygit2.Oid(hex=outcome.result))
            else:
                saved_outcome = outcome
            saved = SavedCall(
                seq=record.seq,
                attempt_id=record.attempt_id,
                ts=record.ts,
                request=request,
                outcome=saved_outcome,
            )
            tree.insert(
                "call.json",
                write_blob(self.repo, saved.model_dump_json(round_trip=True).encode()),
                pygit2.GIT_FILEMODE_BLOB,
            )
            log = finalize_call_log(record, messages, snapshot)
            tree.insert(
                "log.json",
                write_blob(self.repo, log.model_dump_json(round_trip=True).encode()),
                pygit2.GIT_FILEMODE_BLOB,
            )
            if (directory / "sources").exists():
                tree.insert(
                    "sources", capture_sources(self.repo, directory), pygit2.GIT_FILEMODE_TREE
                )
            if (directory / "profiles").exists():
                tree.insert(
                    "profiles",
                    write_directory(self.repo, directory / "profiles"),
                    pygit2.GIT_FILEMODE_TREE,
                )
            timestamp = int(datetime.fromisoformat(record.ts).timestamp())
            signature = pygit2.Signature("grassmoth", "study@local", timestamp, 0)
            parents = argument_parents(record.attempt, self.attempts())
            oid = self.repo.create_commit(
                None,
                signature,
                signature,
                f"{request.action} ({outcome.status})",
                tree.write(),
                [pygit2.Oid(hex=parent) for parent in parents],
            )
            transaction.set_target(call_ref, oid)
            if question_ref is not None:
                transaction.set_target(question_ref, oid)
        return StudyRevision(commit_id=GitOid(str(oid)), parent_ids=parents, record=record)
