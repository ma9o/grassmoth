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
    cached_fit_result,
    expire_analysis_cache,
)

pytestmark = pytest.mark.contract


def test_cached_calls_share_exact_content_and_reject_corruption(tmp_path):
    first = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    second = first.revised(call_id=CallId("call:" + "b" * 64))
    left = cache_fit_result(tmp_path, first, b"complete-result")
    right = cache_fit_result(tmp_path, second, b"complete-result")
    assert left.sha256 == right.sha256
    assert cached_fit_result(tmp_path, first) == cached_fit_result(tmp_path, second)
    assert len(tuple((tmp_path / "results").iterdir())) == 1
    cached_fit_result(tmp_path, first).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content does not match"):
        cached_fit_result(tmp_path, first)


def test_collection_removes_only_expired_aliases_and_content(tmp_path):
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    old = cache_fit_result(tmp_path, call, b"old")
    paths = (
        tmp_path / "calls" / (call.identity + ".json"),
        tmp_path / "results" / (old.sha256 + ".msgpack"),
    )
    for path in paths:
        os.utime(path, (1, 1))
    current = call.revised(call_id=CallId("call:" + "b" * 64))
    cache_fit_result(tmp_path, current, b"current")
    removed, _ = expire_analysis_cache(tmp_path, now_seconds=ANALYSIS_RETENTION_SECONDS + 2)
    assert removed == 2
    assert cached_fit_result(tmp_path, current).read_bytes() == b"current"


def test_cpu_script_receives_verified_path_and_preserves_exit_status(monkeypatch, tmp_path):
    from grassmoth_server.compute import analysis

    monkeypatch.setattr(analysis, "ANALYSIS_CACHE_ROOT", tmp_path)
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    cache_fit_result(tmp_path, call, b"result")
    script = b"import pathlib, sys\nprint(pathlib.Path(sys.argv[1]).read_bytes().decode(), sys.argv[2])\nsys.exit(7)\n"
    (output,) = _analyze_cpu(call, script, ("argument",))
    assert output.exit_code == 7
    assert output.log == "result argument\n"


def test_analysis_dispatch_sends_only_the_script_to_cpu(monkeypatch):
    call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))
    expected = AnalysisOutput(exit_code=0, log="done")
    function = Mock(remote_gen=Mock(return_value=iter((expected,))))
    app = Mock()
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    monkeypatch.setattr(modal.Volume, "from_name", Mock())
    assert analyze_on_modal(call, b"print('done')", ()) == expected
    assert "gpu" not in app.function.call_args.kwargs
    assert "retries" not in app.function.call_args.kwargs
    assert app.function.call_args.kwargs["cpu"] == (2, 2)
    assert app.function.call_args.kwargs["memory"] == (32768, 32768)
    assert app.function.call_args.kwargs["env"]["JAX_PLATFORMS"] == "cpu"
    function.remote_gen.assert_called_once_with(call, b"print('done')", ())
