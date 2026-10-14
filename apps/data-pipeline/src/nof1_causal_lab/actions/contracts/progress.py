"""Immutable progress contracts for a running action."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

type StepStatus = Literal["running", "completed", "failed"]


class StepError(Value):
    """The error type and message of a failed step."""

    type: str
    message: str


class StepEvent(Value):
    """Measurement extraction changed status."""

    attempt_id: UUID
    cursor: str = ""
    event: Literal["nof1-causal-lab.step"] = "nof1-causal-lab.step"
    status: StepStatus
    error: StepError | None = None


class ExtractionPlanEvent(Value):
    """The extraction fan-out plan."""

    attempt_id: UUID
    cursor: str = ""
    event: Literal["nof1-causal-lab.extraction.plan"] = "nof1-causal-lab.extraction.plan"
    total_workers: int = Field(ge=0)
    max_concurrent_workers: int | None = Field(default=None, gt=0)


class ExtractionWorkerEvent(Value):
    """One extraction worker's state; a worker reports its LLM calls when it finishes."""

    attempt_id: UUID
    cursor: str = ""
    event: Literal["nof1-causal-lab.extraction.worker"] = "nof1-causal-lab.extraction.worker"
    worker_id: int = Field(ge=0)
    state: Literal["pending", "running", "completed", "failed"]
    n_windows: int = Field(ge=0)
    n_extractions: int | None = Field(default=None, ge=0)
    n_llm_calls: int | None = Field(default=None, ge=0)
    error: str | None = None


class ExtractionSnapshotEvent(Value):
    """Aggregate extraction worker counts."""

    attempt_id: UUID
    cursor: str = ""
    event: Literal["nof1-causal-lab.extraction.snapshot"] = "nof1-causal-lab.extraction.snapshot"
    total_workers: int = Field(ge=0)
    pending_workers: int = Field(ge=0)
    running_workers: int = Field(ge=0)
    completed_workers: int = Field(ge=0)
    failed_workers: int = Field(ge=0)


type ProgressEvent = Annotated[
    StepEvent | ExtractionPlanEvent | ExtractionWorkerEvent | ExtractionSnapshotEvent,
    Field(discriminator="event"),
]
