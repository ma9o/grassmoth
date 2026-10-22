"""Canonical JSON payload contracts for every file-backed machine artifact.

Prepared panels include data-owned metadata beside their Parquet observations.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from typing import TypeAliasType

from pydantic import BaseModel

from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
from nof1_causal_service.actions.contracts.identity import ArtifactId

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel] | TypeAliasType] = {
    "question": QuestionSpec,
    "panel": PreparedDataMetadata,
    "model": DynamicalModelSpec,
}
