"""Compiled profiling artifacts preserve JAX's native text and numeric outputs."""

import json

import jax
import jax.numpy as jnp
import pytest

from grassmoth_server.actions.fit.execution import dump_compiled_analysis

pytestmark = pytest.mark.contract


def test_compiled_analysis_writes_readable_hlo_and_numeric_costs(tmp_path):
    @jax.jit
    def affine(values):
        return values * 2.0 + 1.0

    dump_compiled_analysis(
        affine.lower(jnp.arange(8, dtype=jnp.float32)).compile(),
        profile_dir=tmp_path,
        label="affine",
    )

    assert "HloModule" in (tmp_path / "affine.hlo.txt").read_text()
    costs = json.loads((tmp_path / "affine.cost.json").read_text())
    assert costs["flops"] >= 8
    assert costs["bytes accessed"] > 0
    operations = json.loads((tmp_path / "affine.access_patterns.json").read_text())
    assert any(count > 0 for count in operations.values())


@pytest.mark.parametrize("failed", [False, True])
def test_fit_shell_closes_trace_and_owns_artifact_writes(monkeypatch, tmp_path, failed):
    from datetime import UTC, datetime
    from importlib import import_module
    from unittest.mock import Mock

    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.programs import PreparedProgram

    shell = import_module("grassmoth_server.actions.fit.action")
    calls = []
    from jax.stages import Compiled

    compiled = Mock(spec=Compiled)
    result = object()

    class ExecutionFailed(Exception):
        pass

    def run(**kwargs):
        calls.append("fit")
        assert callable(kwargs["clock"])
        assert callable(kwargs["wall_clock"])
        assert kwargs["logger"].name == shell.__name__
        if failed:
            raise ExecutionFailed()
        kwargs["compiler"](lambda: None)
        return result

    def dump(executable, profile_dir, label):
        assert executable is compiled
        assert profile_dir == tmp_path
        assert label == "run_batched_step"
        calls.append("dump")

    monkeypatch.setenv("GRASSMOTH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(shell, "fit", run)
    monkeypatch.setattr(
        shell, "compile_program", lambda *_args, **_kwargs: PreparedProgram(compiled, lambda: None)
    )
    monkeypatch.setattr(
        jax.profiler, "start_trace", lambda *_args, **_kwargs: calls.append("start")
    )
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: calls.append("stop"))
    monkeypatch.setattr(shell, "dump_compiled_analysis", dump)

    def execute():
        return shell.run_fit(
            question=Mock(),
            selection=Mock(),
            data_for_model=Mock(),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
            compute_loo_diagnostics=False,
        )

    if failed:
        with pytest.raises(ExecutionFailed):
            execute()
        assert calls == ["start", "fit", "stop"]
    else:
        assert execute() is result
        assert calls == ["start", "fit", "dump", "stop"]
