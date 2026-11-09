"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.public import DynamicalModelSpec
from grassmoth.public import model_diff as compare_models

if TYPE_CHECKING:
    from grassmoth.public import GitOid, ModelDiffOutput


def _model_revision(workspace_id: str, revision: GitOid) -> DynamicalModelSpec:
    """Read the selected model's saved scientific value."""
    from grassmoth_server.study.persistence.store import ArtifactStore, model_values, read_model

    store = ArtifactStore(workspace_id)
    return model_values(store, read_model(store, revision))


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Compare saved spec documents without compiling or evaluating scientific findings."""
    before = _model_revision(workspace_id, before_id)
    after = _model_revision(workspace_id, after_id)
    return compare_models(before, after)
