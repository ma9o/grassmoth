"""Typed action exceptions.

Discriminator: an exception means the action FAILED TO EXECUTE ITS
CONTRACT — state must not change, and the failure is a property of the
attempt (recorded in the attempt log), never of the world. Negative
findings (no estimable treatments, zero observations extracted) are NOT
exceptions: those actions succeed and simply withhold their enabling
artifact while producing a report.

Temporal maps these to non-retryable ApplicationErrors; transient infra
failures (network, OOM, Modal preemption) stay as ordinary exceptions and
are retried by policy without the navigator ever seeing them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


def execution_failure_handler[HandlerT: Callable[..., object]](handler: HandlerT) -> HandlerT:
    """Declare a shell operation whose failure is returned or retried as a failure.

    The checker permits broad exception handling only in these shell operations;
    scientific decisions and HTTP input translation still use specific outcomes.
    """
    return handler


if TYPE_CHECKING:
    from grassmoth.public import JsonObject


class ActionExecutionError(Exception):
    """An action failed to produce its output artifacts.

    ``diagnostics`` carries whatever partial information the attempt yielded
    (e.g. sampler diagnostics from a diverged fit) — informative for the
    navigator, but never persisted as a poisoned pseudo-artifact.
    """

    def __init__(self, message: str, *, diagnostics: JsonObject | None = None) -> None:
        """Retain a failure message and structured diagnostics for the execution boundary."""
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelFitError(ActionExecutionError):
    """Fitting failed to produce a usable posterior."""
