"""Read pinned case inputs without opening or migrating a live study for writing."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pygit2

from evaluation.declarations import CaseFile, StudyModel
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.study.persistence.codec import unpack_result
from nof1_causal_lab.study.persistence.git_objects import read_file

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.declarations import ModelSource


def read_source(source: ModelSource, root: Path) -> bytes:
    """Verify the exact captured bytes; missing inputs and changed content are errors."""
    if isinstance(source, CaseFile):
        payload = (root / source.path).read_bytes()
    else:
        repository = pygit2.Repository(str(root / source.repository))
        payload = read_file(repository, source.revision, "result.msgpack")
    if hashlib.sha256(payload).hexdigest() != source.sha256:
        raise ValueError(f"Case source hash mismatch: {source}")
    return payload


def model_document(source: ModelSource, payload: bytes) -> DynamicalModelSpec:
    """Decode the pinned document using the production model and payload contracts."""
    if isinstance(source, StudyModel):
        return DynamicalModelSpec.model_validate(unpack_result(payload)["dynamical_model_spec"])
    return DynamicalModelSpec.model_validate_json(payload)
