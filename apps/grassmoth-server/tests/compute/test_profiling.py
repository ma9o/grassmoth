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
def test_fit_shell_attributes_cache_events_and_defers_dumps(monkeypatch, tmp_path, failed):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import UTC, datetime
    from importlib import import_module
    from threading import Barrier
    from unittest.mock import Mock

    from jax.stages import Compiled

    from grassmoth.artifacts.reports.profiling import FitProfileReport, ProgramCompileReport
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.budget import NO_MEMORY_DEMAND
    from grassmoth.compute.programs import PreparedProgram

    shell = import_module("grassmoth_server.actions.fit.action")
    calls = []
    barrier = Barrier(2, timeout=5)
    executables = {
        name: Mock(spec=Compiled) for name in ("production_sweep", "observation_projection")
    }
    report = FitProfileReport(
        phase="compile_only",
        repetitions=0,
        chains=1,
        time_steps=1,
        compilation=tuple(ProgramCompileReport(name=name, seconds=1) for name in executables),
    )
    result = Mock(profile=report, revised=Mock(side_effect=lambda **fields: Mock(**fields)))
    register = Mock(wraps=jax.monitoring.register_event_listener)
    unregister = Mock(wraps=jax.monitoring.unregister_event_listener)
    monkeypatch.setattr(jax.monitoring, "register_event_listener", register)
    monkeypatch.setattr(jax.monitoring, "unregister_event_listener", unregister)

    class ExecutionFailed(Exception):
        pass

    def compile_one(name):
        jax.monitoring.record_event("/jax/compilation_cache/compile_requests_use_cache")
        barrier.wait()
        if name == "observation_projection":
            jax.monitoring.record_event("/jax/compilation_cache/cache_hits")
        barrier.wait()
        return executables[name]

    def compile_program(function, *_args, **_kwargs):
        return PreparedProgram(compile_one(function.__name__), lambda: None)

    def prepare_program(function, *_args, **_kwargs):
        return lambda: compile_one(function.__name__), (), (), NO_MEMORY_DEMAND

    def run(**kwargs):
        calls.append("fit")
        assert callable(kwargs["clock"])
        assert callable(kwargs["wall_clock"])
        assert kwargs["logger"].name == shell.__name__

        def production_sweep():
            pass

        def observation_projection():
            pass

        compile_projection, *_ = kwargs["prepare_program"](observation_projection)
        with ThreadPoolExecutor(max_workers=1) as pool:
            projection = pool.submit(compile_projection)
            kwargs["compiler"](production_sweep)
            projection.result()
        assert calls == ["fit"]
        if failed:
            raise ExecutionFailed()
        calls.append("fit_done")
        return result

    def dump(executable, profile_dir, label):
        assert executable is executables[label]
        assert profile_dir == tmp_path / "profile"
        assert "fit_done" in calls
        calls.append(label)

    def trace_forbidden(*_args, **_kwargs):
        raise AssertionError("profile must never start or stop a JAX trace")

    monkeypatch.setattr(shell, "fit", run)
    monkeypatch.setattr(shell, "compile_program", compile_program)
    monkeypatch.setattr(shell, "prepare_program", prepare_program)
    monkeypatch.setattr(jax.profiler, "start_trace", trace_forbidden)
    monkeypatch.setattr(jax.profiler, "stop_trace", trace_forbidden)
    monkeypatch.setattr(shell, "dump_compiled_analysis", dump)

    def execute():
        return shell.run_fit(
            question=Mock(),
            selection=Mock(),
            data_for_model=Mock(),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
            compute_loo_diagnostics=False,
            profile=True,
            profile_dir=tmp_path / "profile",
        )

    if failed:
        with pytest.raises(ExecutionFailed):
            execute()
        assert calls == ["fit"]
    else:
        observed = execute()
        assert {entry.name: entry.persistent_cache for entry in observed.profile.compilation} == {
            "production_sweep": "miss",
            "observation_projection": "hit",
        }
        assert calls == ["fit", "fit_done", "production_sweep", "observation_projection"]
    register.assert_called_once()
    unregister.assert_called_once_with(register.call_args.args[0])
