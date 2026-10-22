"""Content-named model revision storage shared by scientific execution jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth_server.study.artifact_files import json_filename

if TYPE_CHECKING:
    from grassmoth.public import DynamicalModelSpec, GitOid
    from grassmoth_server.actions.contracts.identity import ArtifactId
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.state import ArtifactRecord


def write_model_revision(
    store: ArtifactStore,
    payload: DynamicalModelSpec,
    *,
    derived_from: dict[ArtifactId, GitOid],
    produced_by: str | None,
) -> ArtifactRecord:
    """Write the whole definition with its explicitly selected input revisions."""
    from grassmoth_server.study.persistence.arrays import owned_arrays

    for array in owned_arrays(
        payload.model_dump(mode="python", context={"binary_arrays": True})
    ).values():
        store.write_array(array.values)
    return store.write_artifact(
        "model",
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={json_filename("model", "model"): payload.model_dump(mode="json")},
    )
