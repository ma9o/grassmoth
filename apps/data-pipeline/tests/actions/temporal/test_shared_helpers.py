import pytest

from nof1_causal_service.actions.errors import ActionExecutionError
from nof1_causal_service.actions.temporal.errors import as_non_retryable_application_error

pytestmark = pytest.mark.contract


def test_application_error_preserves_transition_diagnostics():
    error = ActionExecutionError(
        "model failed",
        diagnostics={"reason": "diverged"},
    )

    converted = as_non_retryable_application_error(error)

    assert converted.message == "model failed"
    assert converted.type == "ActionExecutionError"
    assert converted.non_retryable is True
    assert converted.details == ({"reason": "diverged"},)


def test_application_error_omits_diagnostics_for_untyped_failures():
    converted = as_non_retryable_application_error(ValueError("invalid input"))

    assert converted.message == "invalid input"
    assert converted.type == "ValueError"
    assert converted.non_retryable is True
    assert converted.details == ()
