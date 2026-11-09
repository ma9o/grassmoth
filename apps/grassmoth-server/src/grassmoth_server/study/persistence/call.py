"""Storage-only call records; HTTP effects are reconstructed from embedded trees."""

from collections.abc import Mapping
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from grassmoth.public import GitOid, Value
from grassmoth_server.actions.contracts.effects import ActionReportName
from grassmoth_server.actions.contracts.identity import ArtifactId
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from grassmoth_server.compute.analysis_cache import RetainedFit
from grassmoth_server.study.records import Raised, Rejected
from grassmoth_server.study.state import RetractedArtifact


class SavedResult(Value):
    """An encoded body, its law buffers, and optional retained full Modal fit."""

    revision: GitOid
    law_arrays: tuple[str, ...] = ()
    full_fit: RetainedFit | None = None


class SavedApplied(Value):
    """Record names preserve effect order without duplicating embedded metadata."""

    status: Literal["applied"] = "applied"
    result: SavedResult
    produced: tuple[ArtifactId, ...]
    retracted: tuple[RetractedArtifact, ...]
    reports: Mapping[ActionReportName, GitOid]


class SavedCall(Value):
    """A required request and one final outcome stored inside its publication commit."""

    seq: int
    attempt_id: UUID | None
    ts: str
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    outcome: Annotated[SavedApplied | Rejected | Raised, Field(discriminator="status")]
