"""Canonical file layout within an immutable artifact's Git tree.

The artifact graph names semantic dependencies such as ``panel`` or
``model``. This module is the single map from those artifact ids to the
payload names within each artifact tree. UI projections, fixture seeders,
stage runners, and tool contexts should refer to this map instead of spelling
filenames independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


class ArtifactFileSpec(Value):
    """An artifact file specification declares its JSON payloads and tables."""

    json_files: Mapping[str, str] = Field(default_factory=dict)
    parquet_files: Mapping[str, str] = Field(default_factory=dict)


ARTIFACT_FILE_SPECS: Mapping[ArtifactId, ArtifactFileSpec] = MappingProxyType(
    {
        "question": ArtifactFileSpec(json_files={"question": "question.json"}),
        "raw_data": ArtifactFileSpec(parquet_files={"raw": "raw.parquet"}),
        "model": ArtifactFileSpec(json_files={"model": "model.json"}),
        "panel": ArtifactFileSpec(
            json_files={"metadata": "metadata.json"}, parquet_files={"panel": "panel.parquet"}
        ),
    }
)


def artifact_file_spec(artifact_id: ArtifactId) -> ArtifactFileSpec:
    """Look up the canonical JSON and Parquet filenames owned by an artifact kind."""
    return ARTIFACT_FILE_SPECS[artifact_id]


def json_filename(artifact_id: ArtifactId, key: str) -> str:
    """Resolve a logical JSON payload name to its canonical filename within the artifact."""
    return artifact_file_spec(artifact_id).json_files[key]


def parquet_filename(artifact_id: ArtifactId, key: str) -> str:
    """Resolve a logical table name to its canonical Parquet filename within the artifact."""
    return artifact_file_spec(artifact_id).parquet_files[key]
