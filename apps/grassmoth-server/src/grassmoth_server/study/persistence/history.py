"""Git parents record pinned inputs; attempt sequence selects current scientific values."""

from __future__ import annotations

import json
from contextlib import suppress
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
from grassmoth_server.actions.contracts.requests import (
    FitRequest,
    ScientificActionRequest,
    call_identity,
)
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.git_objects import (
    open_repository,
    read_blob,
    read_file,
    write_tree,
)
from grassmoth_server.study.records import (
    ActionMessage,
    Applied,
    Rejected,
    StudyRevision,
    argument_parents,
    recorded_question,
)
from grassmoth_server.study.state import ArtifactRecord, StudyState, apply_effects


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
        if "logs" in commit.tree and not isinstance(
            self.record(revision).record.attempt.outcome, Applied
        ):
            raise StudyLookupError(f"Commit {revision} is an unsuccessful attempt")
        return revision

    def record(self, revision: str) -> StudyRevision:
        """Parse a retained attempt and pair it with its publication commit and parents."""
        commit = self._commit(revision)
        from grassmoth_server.study.records import AttemptRecord

        return StudyRevision(
            record=AttemptRecord.model_validate_json(
                read_file(self.repo, revision, "logs/attempt.json")
            ),
            commit_id=GitOid(str(commit.id)),
            parent_ids=tuple(GitOid(str(parent)) for parent in commit.parent_ids),
        )

    def records(self, revision: str | None = None) -> list[StudyRevision]:
        """Read indexed attempts in submission order, optionally through one attempt's sequence."""
        if revision is None:
            return self.attempts()
        if "logs" not in self._commit(revision).tree:
            return []
        seq = self.record(revision).record.seq
        return [item for item in self.attempts() if item.record.seq <= seq]

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
        question = recorded_question(self.attempts())
        if question is None:
            raise StudyLookupError("Set the study question first")
        return question

    def append(
        self,
        record: AttemptRecord,
        *,
        logs: dict[str, bytes] | None = None,
    ) -> StudyRevision:
        """Publish an independent attempt, deduplicating calls and admitting one question."""
        outcome = record.attempt.outcome
        attempt_ref = f"refs/attempts/{record.seq}"
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
                        messages=(
                            *record.messages[:-1],
                            ActionMessage(
                                timestamp=datetime.fromisoformat(record.ts),
                                severity="error",
                                code=outcome.code,
                                subject=outcome.subject,
                                detail=outcome.detail,
                            ),
                        ),
                    )
                    log = record.model_dump(mode="json", round_trip=True)
                else:
                    question_ref = "refs/question"
            tree = self.repo.TreeBuilder()
            log_files = {
                "attempt.json": json.dumps(log, sort_keys=True).encode(),
                **(logs or {}),
            }
            from grassmoth_server.study.persistence.log_codec import assemble_call_log

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
                from grassmoth_server.study.persistence.arrays import owned_arrays
                from grassmoth_server.study.persistence.codec import unpack_result

                tree.insert(
                    "result.msgpack", pygit2.Oid(hex=outcome.result), pygit2.GIT_FILEMODE_BLOB
                )
                result = unpack_result(read_blob(self.repo, pygit2.Oid(hex=outcome.result)))
                array_ids = tuple(owned_arrays(result))
                for identity in array_ids:
                    array_ref = f"refs/arrays/{identity}"
                    if array_ref not in self.repo.references:
                        # Every owner embeds the same content-addressed array.
                        with suppress(pygit2.AlreadyExistsError):
                            self.repo.references.create(array_ref, pygit2.Oid(hex=outcome.result))
            tree.insert("logs", log_tree.write(), pygit2.GIT_FILEMODE_TREE)
            timestamp = int(datetime.fromisoformat(record.ts).timestamp())
            signature = pygit2.Signature("grassmoth", "study@local", timestamp, 0)
            action = record.attempt.action
            parents = argument_parents(record.attempt, self.attempts())
            oid = self.repo.create_commit(
                None,
                signature,
                signature,
                f"{action} ({outcome.status})",
                tree.write(),
                [pygit2.Oid(hex=parent) for parent in parents],
            )
            transaction.set_target(attempt_ref, oid)
            if call_ref is not None:
                transaction.set_target(call_ref, oid)
            if action_ref is not None:
                transaction.set_target(action_ref, oid)
            if question_ref is not None:
                transaction.set_target(question_ref, oid)
        if isinstance(outcome, Applied):
            from grassmoth_server.storage import paths as data_module

            # Publication now owns these buffers; execution copies are expendable.
            arrays = Path(data_module.store_dir(self.workspace_id)) / "arrays"
            for identity in array_ids:
                path = arrays / f"{identity}.npy"
                path.unlink(missing_ok=True)
        return StudyRevision(commit_id=GitOid(str(oid)), parent_ids=parents, record=record)
