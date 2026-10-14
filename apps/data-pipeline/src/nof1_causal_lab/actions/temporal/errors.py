"""Shared conversion of action failures into Temporal activity failures."""

from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.errors import ActionExecutionError


def as_non_retryable_application_error(exc: Exception) -> ApplicationError:
    """Preserve action diagnostics while preventing deterministic retries."""
    if isinstance(exc, ActionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)
