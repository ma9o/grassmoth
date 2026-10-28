"""Immutable artifact payloads and access to commit-local execution records.

Git owns complete action results in ``study/history.git``. Published artifacts
select those bodies; ``store/`` holds uploaded tables and temporary execution
buffers. Git OIDs identify exact scientific inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pygit2

from grassmoth.public import GitOid, GitRef
from grassmoth_server.actions.errors import execution_failure_handler
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.git_objects import (
    object_tree,
    open_repository,
    read_blob,
    read_file,
    write_blob,
    write_tree,
)
from grassmoth_server.study.state import ArtifactRecord, ArtifactResult

if TYPE_CHECKING:
    import polars as pl
    from jax.typing import ArrayLike
    from pydantic import BaseModel

    from grassmoth.public import DynamicalModelSpec, JsonObject, QuestionSpec
    from grassmoth_server.actions.contracts.identity import ArtifactId


def utc_now_iso() -> str:
    """Read the current UTC clock as a timezone-qualified ISO timestamp."""
    return datetime.now(tz=UTC).isoformat()


def read_model(store: ArtifactStore, revision: GitOid) -> DynamicalModelSpec:
    """Decode a pinned model, keeping its numerical arrays lazy."""
    from functools import cache

    from grassmoth.public import DynamicalModelSpec

    return DynamicalModelSpec.model_validate(
        store.read_json_file("model", revision, "model.json"),
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


def read_question(store: ArtifactStore, revision: GitOid) -> QuestionSpec:
    """Decode the study question pinned at one revision."""
    from grassmoth.public import QuestionSpec

    return store.read_value("question", revision, "question.json", QuestionSpec)


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Immutable Git trees for payloads and input references; large values live in a blob store."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        """Bind the workspace's blob store and open its local artifact repository."""
        self.workspace_id = workspace_id
        self._root = data_module.store_dir(workspace_id)
        self.repo = open_repository(workspace_id, repository_path)

    def write_array(self, values: ArrayLike) -> str:
        """Persist numerical values in the workspace array store and return their content identity."""
        from grassmoth.public import NumericalArray
        from grassmoth_server.storage.arrays import write_owned_array

        array = NumericalArray.from_numpy(np.asarray(values))
        if f"refs/arrays/{array.identity}" in self.repo.references:
            return array.identity
        return write_owned_array(storage.join(self._root, "arrays"), array)

    def read_array(self, identity: str) -> np.ndarray:
        """Load and verify a numerical array using its retained content identity."""
        from grassmoth_server.storage.arrays import decode_array

        return decode_array(identity, self.read_array_bytes(identity))

    @execution_failure_handler
    def read_array_bytes(self, identity: str) -> bytes:
        """Read staging once; publication removing it redirects the read to its retained owner."""
        from grassmoth_server.storage.arrays import read_array_bytes
        from grassmoth_server.study.persistence.codec import unpack_result

        try:
            return read_array_bytes(storage.join(self._root, "arrays"), identity)
        except FileNotFoundError:
            # Publication installs the immutable owner before deleting staging.
            from grassmoth_server.study.persistence.arrays import owned_arrays

            ref = f"refs/arrays/{identity}"
            target = self.repo.references[ref].resolve().target
            assert isinstance(target, pygit2.Oid)
            payload = unpack_result(read_blob(self.repo, target))
            return owned_arrays(payload)[identity].npy

    def write_result(self, result: BaseModel) -> GitOid:
        """Store the exact public result, including display values evaluated during execution."""
        from grassmoth_server.study.persistence.codec import pack_result

        return GitOid(str(write_blob(self.repo, pack_result(result, array_loader=self.read_array))))

    def read_result[ResultT: BaseModel](self, revision: GitOid, target: type[ResultT]) -> ResultT:
        """Restore a saved MessagePack body using its action-owned schema."""
        from functools import cache

        from grassmoth_server.study.persistence.codec import unpack_result

        payload = read_blob(self.repo, pygit2.Oid(hex=revision))
        return target.model_validate(
            unpack_result(payload), context={"distribution_array_loader": cache(self.read_array)}
        )

    def write_artifact(
        self,
        artifact_id: ArtifactId,
        *,
        derived_from: dict[ArtifactId, GitOid],
        produced_by: str | None,
        json_files: JsonObject | None = None,
        parquet_files: dict[str, pl.DataFrame | pa.Table] | None = None,
    ) -> ArtifactRecord:
        """Write one content-addressed tree; Git assigns its immutable identity."""
        metadata = {
            "artifact_id": artifact_id,
            "derived_from": derived_from,
            "produced_by": produced_by,
            "created_at": utc_now_iso(),
        }

        files = {
            name: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            for name, value in (json_files or {}).items()
        }
        external = {}
        for name, frame in (parquet_files or {}).items():
            buffer = io.BytesIO()
            if isinstance(frame, pa.Table):
                pq.write_table(frame, buffer, compression="zstd")
            else:
                frame.write_parquet(buffer)
            payload = buffer.getvalue()
            identity = hashlib.sha256(payload).hexdigest()
            path = Path(self._root) / "blobs" / identity
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(payload)
            external[name] = identity
        files["meta.json"] = json.dumps(metadata, sort_keys=True).encode()
        if external:
            files["external.json"] = json.dumps(external, sort_keys=True).encode()
        revision = write_tree(self.repo, files)
        self.repo.references.create(
            f"refs/artifacts/{artifact_id}/{revision}", revision, force=True
        )
        return ArtifactRecord.model_validate({**metadata, "revision": str(revision)})

    def result_artifact(self, info: ArtifactRecord, result: GitOid) -> ArtifactRecord:
        """Point a scientific artifact at its owning result without copying its payload."""
        if info.artifact_id == "raw_data":
            return info
        metadata = info.revised(source=ArtifactResult(result=result)).model_dump(
            mode="json", exclude={"revision"}
        )
        tree = self.repo.TreeBuilder()
        tree.insert(
            "meta.json",
            write_blob(self.repo, json.dumps(metadata, sort_keys=True).encode()),
            pygit2.GIT_FILEMODE_BLOB,
        )
        tree.insert("result.msgpack", pygit2.Oid(hex=result), pygit2.GIT_FILEMODE_BLOB)
        revision = GitOid(str(tree.write()))
        self.repo.references.create(
            f"refs/artifacts/{info.artifact_id}/{revision}", pygit2.Oid(hex=revision), force=True
        )
        return ArtifactRecord(**metadata, revision=revision)

    def read_record(self, revision: str) -> ArtifactRecord:
        """Read an artifact's identity and provenance without presupposing its kind."""
        metadata = json.loads(read_file(self.repo, revision, "meta.json"))
        return ArtifactRecord(**metadata, revision=GitOid(revision))

    def read_meta(self, artifact_id: ArtifactId, revision: str) -> ArtifactRecord:
        """Read artifact metadata and reject revisions belonging to a different artifact kind."""
        info = self.read_record(revision)
        if info.artifact_id != artifact_id:
            raise StudyLookupError(
                f"Git object {revision} is {info.artifact_id}, not {artifact_id}"
            )
        return info

    def dynamical_model_spec_ref(self, revision: GitOid) -> GitRef:
        """Locate the file that owns a staged or published model's scientific value."""
        info = self.read_meta("model", revision)
        return GitRef(
            workspace_id=self.workspace_id,
            revision=revision,
            path="result.msgpack" if isinstance(info.source, ArtifactResult) else "model.json",
        )

    def read_value[ValueT: BaseModel](
        self, artifact_id: ArtifactId, revision: str, name: str, target: type[ValueT]
    ) -> ValueT:
        """Check artifact ownership and parse a named JSON payload with its production schema."""
        return target.model_validate(self.read_json_file(artifact_id, revision, name))

    def read_json_file(self, artifact_id: ArtifactId, revision: str, name: str) -> JsonObject:
        """Check artifact ownership and read a named JSON object from the selected revision."""
        info = self.read_meta(artifact_id, revision)
        if isinstance(info.source, ArtifactResult):
            from grassmoth_server.study.persistence.codec import unpack_result

            output = unpack_result(read_blob(self.repo, pygit2.Oid(hex=info.source.result)))
            key = {
                "model": "dynamical_model_spec",
                "question": "question",
                "panel": "metadata",
            }[artifact_id]
            return cast("JsonObject", output[key])
        value: JsonObject = json.loads(read_file(self.repo, revision, name))
        return value

    def file_path(self, artifact_id: ArtifactId, revision: str, name: str) -> str:
        """Materialize an artifact file locally, verifying external blob content identities."""
        self.read_meta(artifact_id, revision)
        tree = object_tree(self.repo, revision)
        external = (
            json.loads(read_file(self.repo, revision, "external.json"))
            if "external.json" in tree
            else {}
        )
        if name in external:
            path = Path(self._root) / "blobs" / external[name]
            if hashlib.sha256(path.read_bytes()).hexdigest() != external[name]:
                raise ValueError("Stored table failed its content identity check")
            return str(path)
        payload = read_file(self.repo, revision, name)
        path = Path(data_module.cache_dir(self.workspace_id)) / "git" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return str(path)

    def read_parquet_file(self, artifact_id: ArtifactId, revision: str, name: str) -> pl.DataFrame:
        """Read an artifact's retained Parquet file into a Polars dataframe."""
        import polars as pl

        info = self.read_meta(artifact_id, revision)
        if isinstance(info.source, ArtifactResult):
            from grassmoth.public import PrepareDataOutput

            assert artifact_id == "panel"
            return self.read_result(info.source.result, PrepareDataOutput).observation_frame()
        return pl.read_parquet(self.file_path(artifact_id, revision, name))

    def read_parquet_table(self, artifact_id: ArtifactId, revision: str, name: str) -> pa.Table:
        """Read an artifact's retained Parquet file as an Arrow table with its schema metadata."""
        return pq.read_table(self.file_path(artifact_id, revision, name))


# ---------------------------------------------------------------------------
# Attempt logs and derived state
# ---------------------------------------------------------------------------


_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
