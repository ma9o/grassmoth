"""Shared stream framing, event dispatch and subprocess completion for harnesses."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.llm.harness.events.common import parse_stream_event

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger

    from nof1_causal_lab.json_types import JsonObject


class AsyncByteReader(Protocol):
    """Minimal subprocess stdout surface used by the framing loop."""

    async def read(self, n: int = -1) -> bytes:
        """Read up to ``n`` bytes asynchronously, or read to EOF when ``n`` is negative."""
        ...


async def drain_newline_delimited_stream(
    stream: AsyncByteReader | None,
    handle_line: Callable[[bytes], None],
) -> None:
    """Read arbitrarily long newline-delimited frames and flush the final frame."""
    if stream is None:
        return

    buffer = bytearray()
    while chunk := await stream.read(65536):
        buffer.extend(chunk)
        while (newline := buffer.find(b"\n")) >= 0:
            handle_line(bytes(buffer[:newline]))
            del buffer[: newline + 1]
    if buffer:
        handle_line(bytes(buffer))


def handle_stream_event[State](
    raw: bytes,
    state: State,
    *,
    error_label: str,
    log_label: str | None,
    logger: Logger,
    format_event: Callable[[JsonObject], str | None],
    apply_event: Callable[[State, JsonObject], None],
) -> None:
    """Decode once at the event parser, log and apply a provider-specific event."""
    line = raw.decode("utf-8", errors="replace").strip()
    if not line:
        return
    try:
        event = parse_stream_event(line)
    except ValidationError as exc:
        raise RuntimeError(f"{error_label} emitted non-JSON on stdout: {line[:200]!r}") from exc
    log_line = format_event(event)
    if log_line is not None:
        logger.info("[%s] %s", log_label, log_line)
    apply_event(state, event)


@execution_failure_handler
async def finish_harness_process(
    proc: asyncio.subprocess.Process,
    *,
    timeout_seconds: float,
    backend: str,
    handle_stdout: Callable[[bytes], None],
    handle_stderr: Callable[[str], None],
) -> None:
    """Drain child pipes, bound both waits and report a failed subprocess exit."""
    stderr_bytes = bytearray()

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while chunk := await proc.stderr.read(65536):
            stderr_bytes.extend(chunk)
            for line in chunk.split(b"\n"):
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    handle_stderr(text)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                drain_newline_delimited_stream(proc.stdout, handle_stdout), _drain_stderr()
            ),
            timeout=timeout_seconds,
        )
        await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise
    if proc.returncode != 0:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"{backend} exited with status {proc.returncode}: {stderr_text}")
