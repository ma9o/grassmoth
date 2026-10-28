"""Host-side inference phase events, shared by production runners and benchmarks."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from grassmoth.artifacts.reports.inference import InferencePhase, PhaseOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.json_types import JsonScalar


_PHASE_CAPTURE: ContextVar[tuple[float, list[InferencePhase]] | None] = ContextVar(
    "inference_phases", default=None
)


@contextmanager
def capture_phases(clock: Callable[[], float]) -> Iterator[list[InferencePhase]]:
    """Collect the existing phase intervals within one shell execution."""
    phases: list[InferencePhase] = []
    token = _PHASE_CAPTURE.set((clock(), phases))
    try:
        yield phases
    finally:
        _PHASE_CAPTURE.reset(token)


def log_inference_event(
    label: str,
    stage: str,
    event: str,
    *,
    execution_context: ModelExecutionContext,
    **details: JsonScalar,
) -> None:
    """Emit one timestamped event through the caller's configured logging handlers."""
    payload = {
        **details,
        "sampler": label,
        "stage": stage,
        "event": event,
        "time_utc": execution_context.wall_clock().isoformat(),
    }
    execution_context.logger.log(
        logging.ERROR if event == "failed" else logging.INFO,
        json.dumps(payload, allow_nan=False),
        extra={"inference_progress": payload},
    )


@contextmanager
def inference_phase(
    label: str,
    stage: str,
    *,
    execution_context: ModelExecutionContext,
    **details: JsonScalar,
) -> Iterator[None]:
    """Mark phase entry and completion; preserve exceptions and identify their phase."""
    started = execution_context.clock()
    log_inference_event(label, stage, "started", execution_context=execution_context, **details)
    outcome: PhaseOutcome = "failed"
    try:
        yield
        outcome = "complete"
    finally:
        ended = execution_context.clock()
        capture = _PHASE_CAPTURE.get()
        if capture is not None:
            origin, phases = capture
            phases.append(
                InferencePhase(
                    sampler=label,
                    stage=stage,
                    start_seconds=started - origin,
                    end_seconds=ended - origin,
                    outcome=outcome,
                )
            )
        log_inference_event(
            label,
            stage,
            outcome,
            elapsed_seconds=ended - started,
            execution_context=execution_context,
            **details,
        )
