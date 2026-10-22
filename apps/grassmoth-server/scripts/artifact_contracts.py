"""Canonical JSON payload contracts for every file-backed machine artifact.

Prepared panels include data-owned metadata beside their Parquet observations.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from typing import TypeAliasType

from pydantic import BaseModel

from grassmoth.public import DynamicalModelSpec, PreparedDataMetadata, QuestionSpec
from grassmoth_server.actions.contracts.identity import ArtifactId

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel] | TypeAliasType] = {
    "question": QuestionSpec,
    "panel": PreparedDataMetadata,
    "model": DynamicalModelSpec,
}
