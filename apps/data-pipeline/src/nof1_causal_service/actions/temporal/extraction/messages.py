"""Extraction workflow plans, chunks, and materialization results."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
from nof1_causal_lab.actions.prepare_data.worker_output import ExtractionRow, WorkerOutput
from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.artifacts.observations.preparation import (
    CompletedExtractionWorker,
    FailedExtractionChunk,
    FilePreparationSpec,
    FileSourceRef,
)
from nof1_causal_lab.artifacts.observations.records import ObservationRecord
from nof1_causal_lab.llm.specs import EmbeddedLLMSpec
from nof1_causal_service.actions.contracts.inputs import PrepareDataInput
from nof1_causal_service.actions.contracts.progress import ProgressEvent


class MeasurementsWorkflowInput(BaseModel):
    """Preparation workflow inputs pinned to a raw table revision and study attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    attempt_id: UUID
    raw_data_revision: GitOid
    preparation: PrepareDataInput[GitOid, FileSourceRef]


class ProgressEventInput(BaseModel):
    """Progress event paired with the workspace in which it must be persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    event: ProgressEvent


class ReadSourceDataInput(BaseModel):
    """Captured source files to read into a workspace's raw table artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    source: FileSourceRef


class MeasurementChunkRef(BaseModel):
    """One extraction chunk's specification and optional reusable worker output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    n_windows: int
    spec_ref: str
    cached_result_ref: str | None = None


class MeasurementsPlan(BaseModel):
    """Pinned measurement plan with extraction chunks and their execution limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunks: list[MeasurementChunkRef] = Field(default_factory=list)
    max_concurrent_workers: int
    max_rpm: int
    max_tool_turns: int
    llm: EmbeddedLLMSpec


class ExtractionChunkWorkflowInput(BaseModel):
    """One planned extraction chunk and the LLM settings used to execute it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    chunk: MeasurementChunkRef
    llm: EmbeddedLLMSpec
    max_tool_turns: int


class ExtractionChunkFinalizeInput(BaseModel):
    """Chunk output and execution provenance to validate and retain as measurements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    attempt: int
    n_windows: int
    result_ref: str
    conversation_ref: str
    n_llm_calls: int
    spec_ref: str
    reused: bool | None = False


class CompletedExtractionChunk(CompletedExtractionWorker):
    """The worker's execution envelope adds only its transient result path."""

    n_llm_calls: int | None = 0
    reused: bool | None = False
    result_ref: str


type ExtractionChunkResult = Annotated[
    CompletedExtractionChunk | FailedExtractionChunk, Field(discriminator="status")
]


class MeasurementsFinalizeInput(BaseModel):
    """Pinned plan and completed or failed chunks to assemble into a prepared panel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class MeasurementChunkContext(BaseModel):
    """Stored input of one foreign extraction subroutine, composing its measurement owner."""

    question: str
    window_text: str
    window_starts: list[str]
    measurement_structure: MeasurementContext
    cache_ref: str | None = None


class MeasurementsFile(BaseModel):
    """Materialization inputs retained by the existing plan's scratch transport."""

    measurement_structure: MeasurementContext
    preparation: FilePreparationSpec
    computed_dicts: list[ExtractionRow]
    recorded_dicts: list[ObservationRecord]
    empty_output: WorkerOutput
    chunks: list[MeasurementChunkRef]
