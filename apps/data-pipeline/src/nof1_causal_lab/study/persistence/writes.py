"""Content-named model revision storage shared by scientific execution jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.study.artifact_files import json_filename

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from nof1_causal_lab.study.state import ArtifactRecord


def write_model_revision(
    store: ArtifactStore,
    payload: DynamicalModelSpec,
    *,
    derived_from: dict[ArtifactId, GitOid],
    produced_by: str | None,
) -> ArtifactRecord:
    """Write the whole definition with its explicitly selected input revisions."""
    return store.write_artifact(
        "model",
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={json_filename("model", "model"): payload.model_dump(mode="json")},
    )
