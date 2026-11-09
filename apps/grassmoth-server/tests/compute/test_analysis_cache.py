"""Cache integrity, fourteen-day collection, and CPU script execution."""

import os
from contextlib import nullcontext
from unittest.mock import Mock

import modal
import pytest

from grassmoth_server.actions.contracts.identity import CallId
from grassmoth_server.compute.analysis import AnalysisOutput, _analyze_cpu, analyze_on_modal
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_RETENTION_SECONDS,
    AnalysisCall,
    cache_fit_result,
    expire_analysis_cache,
    fit_profile_directory,
    retained_fit_result,
)
from grassmoth_server.study.persistence.codec import unpack_result
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract


def test_cached_calls_share_exact_content_and_reject_corruption(tmp_path):
    first = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    second = revise_fixture(first, call_id=CallId("call:" + "b" * 64))
    left = cache_fit_result(tmp_path, first, {"body": b"complete-result"})
    right = cache_fit_result(tmp_path, second, {"body": b"complete-result"})
    assert left.sha256 == right.sha256
    assert retained_fit_result(tmp_path, left) == retained_fit_result(tmp_path, right)
    assert len(tuple((tmp_path / "results").iterdir())) == 1
    retained_fit_result(tmp_path, left).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content does not match"):
        retained_fit_result(tmp_path, left)


def test_collection_removes_only_expired_temporary_content(tmp_path):
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    old = cache_fit_result(tmp_path, call, {"body": b"old"})
    paths = (tmp_path / "results" / (old.sha256 + ".msgpack"),)
    for path in paths:
        os.utime(path, (1, 1))
    old_profile = fit_profile_directory(tmp_path, call)
    old_profile.mkdir(parents=True)
    (old_profile / "trace.json").write_text("old trace")
    os.utime(old_profile, (1, 1))
    current = revise_fixture(call, call_id=CallId("call:" + "b" * 64))
    current_entry = cache_fit_result(tmp_path, current, {"body": b"current"})
    current_profile = fit_profile_directory(tmp_path, current)
    current_profile.mkdir(parents=True)
    (current_profile / "trace.json").write_text("current trace")
    removed, _ = expire_analysis_cache(tmp_path, now_seconds=ANALYSIS_RETENTION_SECONDS + 2)
    assert removed == 2
    assert not old_profile.exists()
    assert (current_profile / "trace.json").read_text() == "current trace"
    assert unpack_result(retained_fit_result(tmp_path, current_entry).read_bytes()) == {
        "body": b"current"
    }


def test_pointer_backed_results_survive_temporary_cache_expiry(tmp_path):
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    entry = cache_fit_result(tmp_path / "retained", call, {"body": b"published-evidence"})
    for path in tmp_path.rglob("*"):
        os.utime(path, (1, 1))
    assert expire_analysis_cache(tmp_path, now_seconds=ANALYSIS_RETENTION_SECONDS + 2) == (0, 0)
    assert unpack_result(retained_fit_result(tmp_path / "retained", entry).read_bytes()) == {
        "body": b"published-evidence"
    }


def test_cpu_script_receives_verified_path_and_preserves_exit_status(monkeypatch, tmp_path):
    from grassmoth_server.compute import analysis

    monkeypatch.setattr(analysis, "ANALYSIS_CACHE_ROOT", tmp_path)
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    entry = cache_fit_result(tmp_path / "retained", call, {"body": b"result"})
    script = (
        b"import msgpack, pathlib, sys\n"
        b"body = msgpack.unpackb(pathlib.Path(sys.argv[1]).read_bytes(), raw=False)['body']\n"
        b"print(body.decode(), sys.argv[2])\nsys.exit(7)\n"
    )
    (output,) = _analyze_cpu(entry, script, ("argument",))
    assert output.exit_code == 7
    assert output.log == "result argument\n"


def test_analysis_dispatch_sends_only_the_script_to_cpu(monkeypatch, tmp_path):
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    entry = cache_fit_result(tmp_path / "retained", call, {"body": b"result"})
    expected = AnalysisOutput(exit_code=0, log="done")
    function = Mock(remote_gen=Mock(return_value=iter((expected,))))
    app = Mock()
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    monkeypatch.setattr(modal.Volume, "from_name", Mock())
    assert analyze_on_modal(entry, b"print('done')", ()) == expected
    assert "gpu" not in app.function.call_args.kwargs
    assert "retries" not in app.function.call_args.kwargs
    assert app.function.call_args.kwargs["cpu"] == (2, 2)
    assert app.function.call_args.kwargs["memory"] == (32768, 32768)
    assert app.function.call_args.kwargs["env"]["JAX_PLATFORMS"] == "cpu"
    function.remote_gen.assert_called_once_with(entry, b"print('done')", ())


def test_streamed_cache_preserves_full_pointer_and_bounded_wire_chunks(tmp_path, monkeypatch):
    import io

    from grassmoth_server.compute.analysis_cache import retained_fit_result
    from grassmoth_server.study.persistence import codec

    monkeypatch.setattr(codec, "_BINARY_CHUNK_BYTES", 32)
    payload = {"array": b"a" * 257, "same": b"a" * 257}
    chunks = []
    sink = io.BytesIO()

    def write(chunk):
        chunks.append(len(chunk))
        return sink.write(chunk)

    codec.write_payload(payload, write)
    assert max(chunks) <= 34
    assert sink.getvalue() == codec.pack_payload(payload)
    assert codec.unpack_result(sink.getvalue()) == payload
    call = AnalysisCall(workspace_id="STREAM", call_id=CallId("call:" + "d" * 64))
    full = cache_fit_result(tmp_path, call, payload)
    small = cache_fit_result(tmp_path, call, {"array": b"a"})
    assert codec.unpack_result(retained_fit_result(tmp_path, full).read_bytes()) == payload
    assert full.size > small.size
    assert codec.unpack_result(retained_fit_result(tmp_path, small).read_bytes()) == {"array": b"a"}
