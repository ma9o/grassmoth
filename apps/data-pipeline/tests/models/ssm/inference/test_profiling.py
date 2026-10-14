"""Compiled profiling artifacts preserve JAX's native text and numeric outputs."""

import json

import jax
import jax.numpy as jnp
import pytest

from nof1_causal_lab.actions.fit.execution import dump_compiled_analysis

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
    from importlib import import_module
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel, PreparedFit

    shell = import_module("nof1_causal_lab.actions.fit.execution")
    inference = import_module("nof1_causal_lab.models.ssm.inference.fit")
    calls = []
    compiled = object()
    result = SimpleNamespace(diagnostics=SimpleNamespace(compiled_step=compiled))

    class ExecutionFailed(Exception):
        pass

    def run(*_args, **_kwargs):
        calls.append("fit")
        if failed:
            raise ExecutionFailed()
        return result

    def dump(executable, *, profile_dir, label):
        assert executable is compiled
        assert profile_dir == tmp_path
        assert label == "run_batched_step"
        calls.append("dump")

    monkeypatch.setenv("NOF1_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(inference, "fit", run)
    monkeypatch.setattr(
        jax.profiler, "start_trace", lambda *_args, **_kwargs: calls.append("start")
    )
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: calls.append("stop"))
    monkeypatch.setattr(shell, "dump_compiled_analysis", dump)
    inputs = Mock(spec=CompiledFitInputs, prior_runtime_bundle=object())
    panel = Mock(spec=BoundPanel)
    prepared = Mock(spec=PreparedFit, inputs=inputs, panel=panel)
    if failed:
        with pytest.raises(ExecutionFailed):
            shell.fit_prepared_model(prepared, sampler=SamplerSpec())
        assert calls == ["start", "fit", "stop"]
    else:
        assert shell.fit_prepared_model(prepared, sampler=SamplerSpec()) is result
        assert calls == ["start", "fit", "stop", "dump"]
