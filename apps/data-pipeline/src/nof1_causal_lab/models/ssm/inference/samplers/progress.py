"""Host-side inference phase events, shared by production runners and benchmarks."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.reports.inference import InferencePhase

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from nof1_causal_lab.json_types import JsonScalar

logger = logging.getLogger(__name__)


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


def log_inference_event(label: str, stage: str, event: str, **details: JsonScalar) -> None:
    """Emit one timestamped event through the caller's configured logging handlers."""
    payload = {
        **details,
        "sampler": label,
        "stage": stage,
        "event": event,
        "time_utc": datetime.now(UTC).isoformat(),
    }
    logger.log(
        logging.ERROR if event == "failed" else logging.INFO,
        json.dumps(payload, allow_nan=False),
        extra={"inference_progress": payload},
    )


@contextmanager
def inference_phase(
    label: str, stage: str, *, clock: Callable[[], float], **details: JsonScalar
) -> Iterator[None]:
    """Mark phase entry and completion; preserve exceptions and identify their phase."""
    started = clock()
    log_inference_event(label, stage, "started", **details)
    outcome: Literal["complete", "failed"] = "failed"
    try:
        yield
        outcome = "complete"
    finally:
        ended = clock()
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
            **details,
        )
