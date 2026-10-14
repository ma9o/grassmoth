"""Tests for subprocess stream framing shared by all harnesses."""

from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from nof1_causal_lab.llm.harness.streaming import (
    drain_newline_delimited_stream,
    finish_harness_process,
    handle_stream_event,
)

pytestmark = pytest.mark.contract


class _ChunkedReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = [*chunks, b""]

    async def read(self, n: int = -1) -> bytes:
        del n
        return self._chunks.pop(0)


def test_drain_stream_frames_split_lines_blanks_and_unterminated_tail() -> None:
    received: list[bytes] = []
    stream = _ChunkedReader([b'{"first":', b"1}\n\n", b'{"second":2}\ntail'])

    asyncio.run(drain_newline_delimited_stream(stream, received.append))

    assert received == [b'{"first":1}', b"", b'{"second":2}', b"tail"]


def test_drain_stream_accepts_frames_larger_than_asyncio_readline_limit() -> None:
    received: list[bytes] = []
    long_frame = b"x" * 70_000

    asyncio.run(
        drain_newline_delimited_stream(
            _ChunkedReader([long_frame[:40_000], long_frame[40_000:] + b"\n"]),
            received.append,
        )
    )

    assert received == [long_frame]


def test_drain_stream_propagates_callback_errors() -> None:
    def fail(_raw: bytes) -> None:
        raise RuntimeError("invalid frame")

    with pytest.raises(RuntimeError, match="invalid frame"):
        asyncio.run(drain_newline_delimited_stream(_ChunkedReader([b"bad\n"]), fail))


def test_drain_stream_accepts_missing_stdout() -> None:
    asyncio.run(drain_newline_delimited_stream(None, lambda _raw: None))


@pytest.mark.parametrize("raw", [b"malformed", b"[]"])
def test_stream_event_rejects_invalid_objects_before_applying(raw: bytes) -> None:
    received = []
    with pytest.raises(RuntimeError, match="provider emitted non-JSON"):
        handle_stream_event(
            raw,
            received,
            error_label="provider",
            log_label="test",
            logger=logging.getLogger(__name__),
            format_event=lambda _event: None,
            apply_event=lambda state, event: state.append(event),
        )
    assert received == []


def test_process_drains_both_pipes_without_stderr_backpressure() -> None:
    stdout: list[bytes] = []
    stderr: list[str] = []

    async def _run() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('e' * 100_000); sys.stderr.flush(); print('ok')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await finish_harness_process(
            proc,
            timeout_seconds=5,
            backend="test",
            handle_stdout=stdout.append,
            handle_stderr=stderr.append,
        )
        assert proc.returncode == 0

    asyncio.run(_run())
    assert stdout == [b"ok"]
    assert "".join(stderr) == "e" * 100_000


def test_process_timeout_kills_and_reaps_child() -> None:
    async def _run() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        with pytest.raises(TimeoutError):
            await finish_harness_process(
                proc,
                timeout_seconds=0.05,
                backend="test",
                handle_stdout=lambda _raw: None,
                handle_stderr=lambda _text: None,
            )
        assert proc.returncode is not None
        assert proc.returncode < 0

    asyncio.run(_run())
