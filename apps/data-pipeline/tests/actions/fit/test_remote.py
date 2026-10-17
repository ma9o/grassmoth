"""Modal dispatch uses the selected hardware and the unchanged resolved sampler."""

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import modal
import pytest

from nof1_causal_lab.actions.fit.remote import FitComputeInput, FitComputeResult, _dispatch_fit
from nof1_causal_lab.artifacts.sampling import InferenceHardware, NutsDASpec, SamplerSpec

pytestmark = pytest.mark.contract


def test_pytest_fits_locally_and_rejects_live_modal_apps():
    from nof1_causal_lab.config import get_config

    assert get_config().inference.sampler.hardware is InferenceHardware.LOCAL
    with pytest.raises(pytest.fail.Exception, match="Live Modal execution is disabled"):
        modal.App("pytest-local-only").run()


@pytest.mark.parametrize("hardware", [InferenceHardware.L40S, InferenceHardware.H100])
@pytest.mark.parametrize("time_limit", [14400, 90])
def test_modal_function_requests_selected_gpu_without_changing_joint_sampler(
    monkeypatch, hardware, time_limit
):
    # The SDK's process-global terminal renderer races when independent fit
    # activities create apps concurrently. Service dispatch must stay headless.
    monkeypatch.setattr(
        modal, "enable_output", Mock(side_effect=AssertionError("Interactive Modal output"))
    )
    payload = FitComputeInput(
        dynamical_model_spec_bytes=b"pinned-model",
        outcome=None,
        panel_parquet=b"pinned-history",
        variables=(),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        sampler=SamplerSpec(algorithm=NutsDASpec(), num_chains=4, time_limit_seconds=time_limit),
        compute_loo_diagnostics=False,
    )
    expected = FitComputeResult(b"fitted-model", b"evidence", "{}")
    remote = Mock()
    remote.remote.return_value = expected
    app = Mock()
    app.function.return_value = lambda _function: remote
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    assert _dispatch_fit(payload, hardware=hardware) is expected
    assert app.function.call_args.kwargs["gpu"] == hardware.value
    assert app.function.call_args.kwargs["timeout"] == time_limit
    assert app.function.call_args.kwargs["startup_timeout"] == time_limit
    assert app.function.call_args.kwargs["scaledown_window"] == 2
    assert "max_containers" not in app.function.call_args.kwargs
    assert app.function.call_args.kwargs["cpu"] == (8, 8)
    assert app.function.call_args.kwargs["memory"] == (32768, 32768)
    remote.remote.assert_called_once_with(payload)
    remote.spawn.assert_not_called()


def test_modal_timeout_propagates_without_another_call(monkeypatch):
    payload = FitComputeInput(
        b"model", None, b"history", (), datetime.now(UTC), SamplerSpec(), False
    )
    function = Mock(remote=Mock(side_effect=modal.exception.FunctionTimeoutError("expired")))
    app = Mock()
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    with pytest.raises(modal.exception.FunctionTimeoutError, match="expired"):
        _dispatch_fit(payload, hardware=InferenceHardware.L40S)
    function.remote.assert_called_once_with(payload)
    function.spawn.assert_not_called()


def test_runtime_records_the_enforced_modal_limit():
    from nof1_causal_lab.actions.fit.runtime import fit_runtime

    sampler = SamplerSpec(hardware="L40S", time_limit_seconds=90)
    assert fit_runtime(sampler).time_limit_seconds == 90
    assert fit_runtime(sampler.revised(hardware="local")).time_limit_seconds is None
