"""Modal dispatch uses the selected hardware and the unchanged resolved sampler."""

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import modal
import pytest

from nof1_causal_lab.actions.fit.remote import (
    FIT_CACHE_VOLUME_ENV,
    FitComputeInput,
    FitComputeResult,
    _dispatch_fit,
    decode_fit_result,
    encode_fit_result,
)
from nof1_causal_lab.artifacts.sampling import InferenceHardware, NutsDASpec, SamplerSpec

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def isolated_cache_volume(monkeypatch):
    monkeypatch.setenv(FIT_CACHE_VOLUME_ENV, "test-fit-cache")
    volume = Mock()
    monkeypatch.setattr(modal.Volume, "from_name", volume)
    return volume


def test_pytest_fits_locally_and_rejects_live_modal_apps():
    from nof1_causal_lab.config import get_config

    assert get_config().inference.sampler.hardware is InferenceHardware.LOCAL
    with pytest.raises(pytest.fail.Exception, match="Live Modal execution is disabled"):
        modal.App("pytest-local-only").run()


@pytest.mark.parametrize("hardware", [InferenceHardware.L40S, InferenceHardware.H100])
@pytest.mark.parametrize("time_limit", [14400, 90])
def test_modal_function_requests_selected_gpu_without_changing_joint_sampler(
    monkeypatch, hardware, time_limit, isolated_cache_volume
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
    remote.remote.return_value = encode_fit_result(expected)
    app = Mock()
    app.function.return_value = lambda _function: remote
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    assert _dispatch_fit(payload, hardware=hardware) == expected
    assert app.function.call_args.kwargs["gpu"] == hardware.value
    assert app.function.call_args.kwargs["timeout"] == time_limit
    assert app.function.call_args.kwargs["startup_timeout"] == time_limit
    assert app.function.call_args.kwargs["scaledown_window"] == 2
    assert "max_containers" not in app.function.call_args.kwargs
    assert app.function.call_args.kwargs["cpu"] == (8, 8)
    assert app.function.call_args.kwargs["memory"] == (32768, 32768)
    assert app.function.call_args.kwargs["env"][FIT_CACHE_VOLUME_ENV] == "test-fit-cache"
    assert app.function.call_args.kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert app.function.call_args.kwargs["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.90"
    assert app.function.call_args.kwargs["env"]["JAX_COMPILATION_CACHE_DIR"].startswith("/tmp/")
    assert (
        app.function.call_args.kwargs["env"]["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0.0"
    )
    assert app.function.call_args.kwargs["volumes"] == {
        "/cache": isolated_cache_volume.return_value
    }
    isolated_cache_volume.assert_called_once_with("test-fit-cache", create_if_missing=True)
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


@pytest.mark.parametrize("model", [None, bytes(range(256)) * 1000])
def test_fit_return_compression_preserves_all_bytes(model):
    import pickle

    expected = FitComputeResult(model, bytes(range(256)) * 2000, '{"message":"é"}')
    frame = encode_fit_result(expected)
    restored = decode_fit_result(frame)
    assert restored == expected
    assert pickle.dumps(restored, protocol=5) == pickle.dumps(expected, protocol=5)
    assert len(frame) < len(expected.evidence) // 10


def test_fit_return_rejects_corruption_and_sizes_before_decode(monkeypatch):
    import pyarrow as pa

    from nof1_causal_lab.actions.fit import remote

    frame = encode_fit_result(FitComputeResult(b"model", b"evidence", "{}"))
    corrupted = frame[:8] + bytes([frame[8] ^ 1]) + frame[9:]
    with pytest.raises(ValueError, match="checksum"):
        decode_fit_result(corrupted)
    decompress = Mock(side_effect=AssertionError("Must reject before allocation"))
    monkeypatch.setattr(pa, "decompress", decompress)
    with pytest.raises(ValueError, match="decoded size bound"):
        decode_fit_result((remote.FIT_RESULT_MAX_BYTES + 1).to_bytes(8, "big") + frame[8:])
    with pytest.raises(ValueError, match="transport size"):
        decode_fit_result(b"truncated")
    monkeypatch.setattr(remote, "FIT_RESULT_MAX_BYTES", len(frame) - 1)
    with pytest.raises(ValueError, match="transport size"):
        decode_fit_result(frame)
    with pytest.raises(ValueError, match="transport size bound"):
        encode_fit_result(FitComputeResult(b"too large" * len(frame), b"", "{}"))
    decompress.assert_not_called()
