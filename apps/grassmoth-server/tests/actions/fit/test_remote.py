"""Modal dispatch uses the selected hardware and the unchanged resolved sampler."""

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import modal
import pytest

from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.sampling import InferenceHardware, NutsDASpec, SamplerSpec
from grassmoth_server.actions.fit.remote import (
    FIT_CACHE_VOLUME_ENV,
    FitComputeInput,
    FitComputeResult,
    _claim_fit_attempt,
    _dispatch_fit,
    _result_chunks,
    decode_fit_result,
    encode_fit_result,
)
from grassmoth_test_support.helpers import fixture_entity_id

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def isolated_cache_volume(monkeypatch):
    monkeypatch.setenv(FIT_CACHE_VOLUME_ENV, "test-fit-cache")
    volume = Mock()
    monkeypatch.setattr(modal.Volume, "from_name", volume)
    return volume


@pytest.fixture(autouse=True)
def isolated_attempt(monkeypatch):
    attempt = Mock()
    monkeypatch.setattr(modal.Dict, "ephemeral", lambda: nullcontext(attempt))
    return attempt


def test_pytest_fits_locally_and_rejects_live_modal_apps():
    from grassmoth_server.config import get_config

    assert get_config().inference.sampler.hardware is InferenceHardware.LOCAL
    with pytest.raises(pytest.fail.Exception, match="Live Modal execution is disabled"):
        modal.App("pytest-local-only").run()


@pytest.mark.parametrize("hardware", [InferenceHardware.L40S, InferenceHardware.H100])
@pytest.mark.parametrize("time_limit", [14400, 90])
def test_modal_function_requests_selected_gpu_without_changing_joint_sampler(
    monkeypatch, hardware, time_limit, isolated_cache_volume, isolated_attempt
):
    # The SDK's process-global terminal renderer races when independent fit
    # activities create apps concurrently. Service dispatch must stay headless.
    monkeypatch.setattr(
        modal, "enable_output", Mock(side_effect=AssertionError("Interactive Modal output"))
    )
    payload = FitComputeInput(
        dynamical_model_spec_bytes=b"pinned-model",
        question=QuestionSpec(text="Test", outcome=fixture_entity_id("construct", "outcome")),
        panel_parquet=b"pinned-history",
        variables=(),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        sampler=SamplerSpec(algorithm=NutsDASpec(), num_chains=4, time_limit_seconds=time_limit),
        compute_loo_diagnostics=False,
    )
    expected = FitComputeResult(b"complete-fit")
    remote = Mock()
    remote.remote_gen.return_value = _result_chunks(encode_fit_result(expected))
    app = Mock()
    app.function.return_value = lambda _function: remote
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    assert _dispatch_fit(payload, hardware=hardware) == expected
    assert app.function.call_args.kwargs["gpu"] == hardware.value
    assert app.function.call_args.kwargs["timeout"] == time_limit
    assert app.function.call_args.kwargs["startup_timeout"] == time_limit
    assert app.function.call_args.kwargs["scaledown_window"] == 2
    assert "retries" not in app.function.call_args.kwargs
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
        "/cache": isolated_cache_volume.return_value,
    }
    assert isolated_cache_volume.call_count == 1
    isolated_cache_volume.assert_any_call("test-fit-cache", create_if_missing=True)
    remote.remote_gen.assert_called_once_with(payload, isolated_attempt)
    remote.remote.assert_not_called()
    remote.spawn.assert_not_called()


def test_modal_timeout_propagates_without_another_call(monkeypatch, isolated_attempt):
    payload = FitComputeInput(
        b"model",
        QuestionSpec(text="Test", outcome=fixture_entity_id("construct", "outcome")),
        b"history",
        (),
        datetime.now(UTC),
        SamplerSpec(),
        False,
    )
    function = Mock(remote_gen=Mock(side_effect=modal.exception.FunctionTimeoutError("expired")))
    app = Mock()
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    with pytest.raises(modal.exception.FunctionTimeoutError, match="expired"):
        _dispatch_fit(payload, hardware=InferenceHardware.L40S)
    function.remote_gen.assert_called_once_with(payload, isolated_attempt)
    function.spawn.assert_not_called()


def test_result_chunks_bound_sdk_uploads_and_preserve_the_transport():
    from grassmoth_server.actions.fit.remote import _RESULT_CHUNK_BYTES

    frame = bytes(range(256)) * (_RESULT_CHUNK_BYTES // 256 + 1)
    chunks = tuple(_result_chunks(frame))
    assert tuple(map(len, chunks)) == (_RESULT_CHUNK_BYTES, 256)
    assert b"".join(chunks) == frame


def test_crashed_runner_cannot_start_another_fit():
    attempt = Mock(put=Mock(side_effect=[True, False]))
    _claim_fit_attempt(attempt)
    with pytest.raises(RuntimeError, match="interrupted; refusing to repeat"):
        _claim_fit_attempt(attempt)
    assert attempt.put.call_count == 2
    attempt.put.assert_called_with("started", True, skip_if_exists=True)


def test_interrupted_result_stream_does_not_decode_partial_output(monkeypatch):
    from grassmoth_server.actions.fit import remote

    def interrupted(*_args):
        yield b"partial-frame"
        raise RuntimeError("runner interrupted")

    function = Mock(remote_gen=interrupted)
    app = Mock()
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    decode = Mock(side_effect=AssertionError("Partial output must not be decoded"))
    monkeypatch.setattr(remote, "decode_fit_result", decode)
    payload = FitComputeInput(
        b"model",
        QuestionSpec(text="Test", outcome=fixture_entity_id("construct", "outcome")),
        b"history",
        (),
        datetime.now(UTC),
        SamplerSpec(),
        False,
    )
    with pytest.raises(RuntimeError, match="runner interrupted"):
        _dispatch_fit(payload, hardware=InferenceHardware.L40S)
    decode.assert_not_called()


def test_runtime_records_the_enforced_modal_limit():
    from grassmoth_server.actions.fit.runtime import fit_runtime

    sampler = SamplerSpec(hardware="L40S", time_limit_seconds=90)
    assert fit_runtime(sampler).time_limit_seconds == 90
    assert fit_runtime(sampler.revised(hardware="local")).time_limit_seconds is None


@pytest.mark.parametrize("model", [None, bytes(range(256)) * 1000])
def test_fit_return_compression_preserves_all_bytes(model):
    import pickle

    expected = FitComputeResult((model or b"") + bytes(range(256)) * 2000)
    frame = encode_fit_result(expected)
    restored = decode_fit_result(frame)
    assert restored == expected
    assert pickle.dumps(restored, protocol=5) == pickle.dumps(expected, protocol=5)
    assert len(frame) < len(expected.output) // 10


def test_fit_return_rejects_corruption_and_truncation_before_unpickling(monkeypatch):
    from grassmoth_server.actions.fit import remote

    frame = encode_fit_result(FitComputeResult(b"complete-fit"))
    unpickle = Mock(side_effect=AssertionError("Must verify before unpickling"))
    monkeypatch.setattr(remote.pickle, "loads", unpickle)
    corrupted = frame[:8] + bytes([frame[8] ^ 1]) + frame[9:]
    with pytest.raises(ValueError, match="checksum"):
        decode_fit_result(corrupted)
    with pytest.raises(ValueError, match="transport size"):
        decode_fit_result(b"truncated")
    unpickle.assert_not_called()


def test_fit_return_rejects_inflated_declared_length_without_allocating_it(monkeypatch):
    from grassmoth_server.actions.fit import remote

    frame = encode_fit_result(FitComputeResult(b"complete-fit"))
    unpickle = Mock(side_effect=AssertionError("Must verify before unpickling"))
    monkeypatch.setattr(remote.pickle, "loads", unpickle)
    with pytest.raises(ValueError, match="length"):
        decode_fit_result((2**63).to_bytes(8, "big") + frame[8:])
    unpickle.assert_not_called()
