"""Host-side inference phase events, shared by production runners and benchmarks."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from nof1_causal_lab.json_types import JsonScalar

logger = logging.getLogger(__name__)


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
    completed = False
    try:
        yield
        completed = True
    finally:
        log_inference_event(
            label,
            stage,
            "complete" if completed else "failed",  # ty: ignore[redundant-condition-strict] - yield can raise before assignment.
            elapsed_seconds=clock() - started,
            **details,
        )
