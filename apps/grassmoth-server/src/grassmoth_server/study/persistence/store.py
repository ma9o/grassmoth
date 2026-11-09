"""Complete result blobs and their embedded scientific record trees."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pygit2

from grassmoth.public import ConstructId, GitOid, GitRef
from grassmoth_server.actions.errors import execution_failure_handler
from grassmoth_server.storage import paths
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.call import SavedResult
from grassmoth_server.study.persistence.git_objects import (
    open_repository,
    read_blob,
    read_file,
    write_blob,
    write_file_blob,
)
from grassmoth_server.study.persistence.sources import capture_sources
from grassmoth_server.study.state import (
    ArtifactFiles,
    ArtifactRecord,
    ArtifactResult,
    ArtifactSource,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from pydantic import BaseModel

    from grassmoth.public import DynamicalModelSpec, JsonObject, QuestionSpec
    from grassmoth_server.actions.contracts.identity import ArtifactId
    from grassmoth_server.compute.analysis_cache import RetainedFit


def utc_now_iso() -> str:
    """Read the current UTC clock as a timezone-qualified ISO timestamp."""
    return datetime.now(tz=UTC).isoformat()


def read_model(store: ArtifactStore, revision: GitOid) -> DynamicalModelSpec:
    """Decode a pinned model, keeping its numerical arrays lazy."""
    from functools import cache

    from grassmoth.public import DynamicalModelSpec

    return DynamicalModelSpec.model_validate(
        store.read_document("model", revision),
    ).materialized(context={"distribution_array_loader": cache(store.read_array)})


def model_values(store: ArtifactStore, model: DynamicalModelSpec) -> DynamicalModelSpec:
    """Read every referenced law buffer before handing a document to a library action."""
    from functools import cache

    from grassmoth.public import DynamicalModelSpec

    return DynamicalModelSpec.model_validate(
        model.model_dump(
            mode="python",
            context={"binary_arrays": True, "distribution_array_loader": cache(store.read_array)},
            warnings="error",
        )
    )


def read_question(store: ArtifactStore, revision: GitOid) -> QuestionSpec[ConstructId]:
    """Decode the study question pinned at one revision."""
    from grassmoth.public import QuestionSpec

    return QuestionSpec[ConstructId].model_validate(store.read_document("question", revision))


class ArtifactStore:
    """Exact values in Git; unfinished encodings belong to their call's temporary directory."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        """Open the workspace's local study repository."""
        self.workspace_id = workspace_id
        self.repo = open_repository(workspace_id, repository_path)

    def read_array(self, identity: str) -> np.ndarray:
        """Load a law buffer using its published content identity."""
        from grassmoth_server.storage.arrays import decode_array

        return decode_array(identity, self.read_array_bytes(identity))

    @execution_failure_handler
    def read_array_bytes(self, identity: str) -> bytes:
        """Read a law buffer from its immutable result owner."""
        from grassmoth_server.study.persistence.arrays import owned_arrays
        from grassmoth_server.study.persistence.codec import unpack_result

        target = self.repo.references[f"refs/arrays/{identity}"].resolve().target
        assert isinstance(target, pygit2.Oid)
        payload = unpack_result(read_blob(self.repo, target))
        return owned_arrays(payload["dynamical_model_spec"])[identity].npy

    def write_result(self, result: BaseModel, *, call_id: str) -> GitOid:
        """Stream the served body and retain its law index for later atomic publication."""
        from grassmoth_server.study.persistence.arrays import owned_arrays
        from grassmoth_server.study.persistence.codec import result_payload, write_payload

        values = result_payload(result, array_loader=self.read_array, thin_fit_draws=True)
        directory = Path(paths.call_dir(self.workspace_id, call_id))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.msgpack"
        with path.open("wb") as stream:
            write_payload(values, stream.write)
        return self.import_result(
            call_id=call_id,
            path=path,
            law_arrays=tuple(owned_arrays(values.get("dynamical_model_spec"))),
            full_fit=None,
        )

    def import_result(
        self,
        *,
        call_id: str,
        path: Path,
        law_arrays: tuple[str, ...],
        full_fit: RetainedFit | None,
    ) -> GitOid:
        """Save encoded result bytes and their publication metadata without decoding them."""
        revision = GitOid(str(write_file_blob(self.repo, path)))
        saved = SavedResult(revision=revision, law_arrays=law_arrays, full_fit=full_fit)
        metadata = Path(paths.call_dir(self.workspace_id, call_id)) / "result.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(saved.model_dump_json())
        return revision

    def read_result_bytes(self, revision: GitOid) -> bytes:
        """Read the exact served body locally, without remote storage or projection."""
        return read_blob(self.repo, pygit2.Oid(hex=revision))

    def read_result[ResultT: BaseModel](self, revision: GitOid, target: type[ResultT]) -> ResultT:
        """Restore the saved body using its action-owned schema."""
        from functools import cache

        from grassmoth_server.study.persistence.codec import unpack_result

        return target.model_validate(
            unpack_result(self.read_result_bytes(revision)),
            context={"distribution_array_loader": cache(self.read_array)},
        )

    def result_record(
        self,
        artifact_id: ArtifactId,
        result: GitOid,
        *,
        derived_from: Mapping[ArtifactId, GitOid],
        produced_by: str | None,
    ) -> ArtifactRecord:
        """Create the final record directly against its complete result."""
        return self._record(
            artifact_id,
            derived_from=derived_from,
            produced_by=produced_by,
            source=ArtifactResult(result=result),
            entries={"result.msgpack": (pygit2.Oid(hex=result), pygit2.GIT_FILEMODE_BLOB)},
        )

    def raw_record(self, call_id: str) -> ArtifactRecord:
        """Capture source bytes as the durable raw-data record; parsed rows stay temporary."""
        directory = Path(paths.call_dir(self.workspace_id, call_id))
        return self._record(
            "raw_data",
            derived_from={},
            produced_by="prepare_data",
            source=ArtifactFiles(),
            entries={"sources": (capture_sources(self.repo, directory), pygit2.GIT_FILEMODE_TREE)},
        )

    def _record(
        self,
        artifact_id: ArtifactId,
        *,
        derived_from: Mapping[ArtifactId, GitOid],
        produced_by: str | None,
        source: ArtifactSource,
        entries: Mapping[str, tuple[pygit2.Oid, int]],
    ) -> ArtifactRecord:
        metadata = {
            "artifact_id": artifact_id,
            "derived_from": dict(derived_from),
            "produced_by": produced_by,
            "created_at": utc_now_iso(),
            "source": source.model_dump(mode="json"),
        }
        tree = self.repo.TreeBuilder()
        tree.insert(
            "meta.json",
            write_blob(self.repo, json.dumps(metadata, sort_keys=True).encode()),
            pygit2.GIT_FILEMODE_BLOB,
        )
        for name, (oid, mode) in entries.items():
            tree.insert(name, oid, mode)
        return ArtifactRecord.model_validate({**metadata, "revision": str(tree.write())})

    def read_record(self, revision: str) -> ArtifactRecord:
        """Read an artifact's identity and provenance without presupposing its kind."""
        metadata = json.loads(read_file(self.repo, revision, "meta.json"))
        return ArtifactRecord(**metadata, revision=GitOid(revision))

    def read_meta(self, artifact_id: ArtifactId, revision: str) -> ArtifactRecord:
        """Parse artifact ownership at the storage boundary."""
        info = self.read_record(revision)
        if info.artifact_id != artifact_id:
            raise StudyLookupError(
                f"Git object {revision} is {info.artifact_id}, not {artifact_id}"
            )
        return info

    def dynamical_model_spec_ref(self, revision: GitOid) -> GitRef:
        """Locate the complete result owning a model's scientific value."""
        self.read_meta("model", revision)
        return GitRef(workspace_id=self.workspace_id, revision=revision, path="result.msgpack")

    def read_document(self, artifact_id: ArtifactId, revision: str) -> JsonObject:
        """Select a scientific field from the embedded complete result."""
        from grassmoth_server.study.persistence.codec import unpack_result

        self.read_meta(artifact_id, revision)
        output = unpack_result(read_file(self.repo, revision, "result.msgpack"))
        key = {"model": "dynamical_model_spec", "question": "question", "panel": "metadata"}[
            artifact_id
        ]
        return cast("JsonObject", output[key])
