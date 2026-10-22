"""Phase events describe completed work and preserve infrastructure failures."""

import json
import logging
from datetime import UTC, datetime

import pytest

from grassmoth.models.ssm.inference.samplers.progress import inference_phase

pytestmark = pytest.mark.contract


def test_compilation_completion_is_emitted_only_after_the_work_finishes(caplog):
    caplog.set_level(logging.INFO)
    ticks = iter((12.0, 17.5))
    with inference_phase(
        "pgibbs",
        "compiling_sweeps",
        clock=lambda: next(ticks),
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    ):
        assert [json.loads(record.message)["event"] for record in caplog.records] == ["started"]
    events = [json.loads(record.message) for record in caplog.records]
    assert [event["event"] for event in events] == ["started", "complete"]
    assert events[-1]["elapsed_seconds"] == 5.5
    assert events[-1]["sampler"] == "pgibbs"
    assert events[-1]["stage"] == "compiling_sweeps"
    assert events[-1]["time_utc"].endswith("+00:00")


def test_compilation_failure_is_reported_without_a_false_completion(caplog):
    caplog.set_level(logging.INFO)
    failure = MemoryError("compiled buffers exceed budget")
    with (
        pytest.raises(MemoryError) as caught,
        inference_phase(
            "mpgibbs",
            "compiling_sweeps",
            clock=lambda: 1.0,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
        ),
    ):
        raise failure
    assert caught.value is failure
    assert [json.loads(record.message)["event"] for record in caplog.records] == [
        "started",
        "failed",
    ]
    assert caplog.records[-1].levelno == logging.ERROR
