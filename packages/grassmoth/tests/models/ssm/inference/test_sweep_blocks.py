import json
import logging
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.sampling import GuidedPathSpec, SamplerSpec
from grassmoth.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    MemoryDemand,
    SharedMemoryBudget,
)
from grassmoth.models.ssm.inference.samplers import sweep_plan, sweeps
from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    ADAPTATION_WINDOW,
    adapt_scales,
    initialize_adaptation,
    latent_adaptation,
)
from grassmoth.models.ssm.inference.samplers.sweep_plan import SweepProgram
from grassmoth.models.ssm.inference.samplers.sweeps import run_sweeps
from grassmoth_test_support.compute import EXECUTION_CONTEXT

"""Compiled blocks preserve transitions, phase outputs and explicit memory limits."""


pytestmark = pytest.mark.inference(concern="sampling")


class _Carry(NamedTuple):
    key: jax.Array
    state: jax.Array


def _step(carry, posterior):
    key, state = carry
    key, draw = jax.random.split(key)
    state = state + jax.random.normal(draw, state.shape) * jnp.where(posterior, 1, 2)
    record = {"telemetry": state.sum(axis=-1), "paths": state}
    return _Carry(key, state), record


@pytest.mark.parametrize(("warmup", "samples"), [(0, 29), (28, 27), (7, 0)])
def test_block_boundaries_and_padding_preserve_key_stream_and_phase_outputs(
    warmup, samples, caplog
):
    caplog.set_level(logging.INFO, logger="grassmoth.models.ssm.inference.samplers.progress")
    initial = _Carry(jax.random.key(791), jnp.zeros((3, 4)))
    plan = sweeps.compile_sweeps(
        SweepProgram(_step),
        initial,
        execution_context=EXECUTION_CONTEXT,
        sampler=SamplerSpec(num_warmup=warmup, num_samples_per_chain=samples),
    )
    run = run_sweeps(
        plan,
        initial,
        execution_context=replace(
            EXECUTION_CONTEXT,
            logger=logging.getLogger("grassmoth.models.ssm.inference.samplers.progress"),
        ),
    )
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name.endswith("samplers.progress")
    ]
    boundaries = [
        (event["stage"], event["event"]) for event in events if event["event"] != "progress"
    ]
    assert boundaries == [
        *(([("warmup", "started"), ("warmup", "complete")]) if warmup else [("warmup", "skipped")]),
        *(
            ([("sampling", "started"), ("sampling", "complete")])
            if samples
            else [("sampling", "skipped")]
        ),
    ]
    for stage, total in (("warmup", warmup), ("sampling", samples)):
        progress = [
            event for event in events if event["stage"] == stage and event["event"] == "progress"
        ]
        assert [event["completed"] for event in progress][-1:] == ([total] if total else [])
    assert isinstance(run.state, _Carry)
    reference = initial
    for phase, count, actual in (
        ("warmup", warmup, run.history.warmup),
        ("posterior", samples, run.history.posterior),
    ):
        reference, records = jax.jit(
            lambda state, phase=phase, count=count: jax.lax.scan(
                lambda carry, _: _step(carry, jnp.asarray(phase == "posterior")),
                state,
                None,
                length=count,
            )
        )(reference)
        if phase == "warmup":
            records = {name: value for name, value in records.items() if name != "paths"}
        assert set(actual) == set(records)
        for name, expected in records.items():
            assert isinstance(actual[name], jax.Array)
            assert actual[name].dtype == expected.dtype
            # Keys match bitwise; blocked and plain scans may round float32 differently.
            np.testing.assert_allclose(
                actual[name], np.asarray(expected).swapaxes(0, 1), rtol=1e-5, atol=1e-5
            )
    np.testing.assert_array_equal(
        jax.random.key_data(run.state[0]), jax.random.key_data(reference[0])
    )
    np.testing.assert_allclose(run.state[1], reference[1], atol=1e-5)
    assert "paths" not in run.history.warmup
    assert run.retained_draws == samples


@pytest.mark.parametrize("accelerator", [False, True])
def test_planner_uses_compiled_memory_without_executing_or_serializing_chains(
    monkeypatch, accelerator
):
    compiled_phases = []
    executed = []

    def step(state, _posterior):
        return state + 1, {"value": state}

    def memory(executable):
        compiled_phases.append(executable)
        return MemoryDemand(host_bytes=0, device_bytes=200_000)

    original_call = jax.stages.Compiled.__call__

    def execute(self, *args, **kwargs):
        executed.append(True)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
    monkeypatch.setattr(sweeps, "compiled_memory", memory)
    budget = ComputeBudget(
        # History (384) plus one transferred block (1200) fit on the host.
        memory=AcceleratorMemoryBudget(host_bytes=2000, device_bytes=300_000)
        if accelerator
        else SharedMemoryBudget(available_bytes=300_000)
    )
    initial = jnp.zeros((4, 3))
    plan = sweeps.compile_sweeps(
        SweepProgram(step),
        initial,
        execution_context=replace(EXECUTION_CONTEXT, budget=budget),
        sampler=SamplerSpec(num_warmup=3, num_samples_per_chain=5),
    )
    assert len(compiled_phases) == 1
    assert isinstance(plan.warmup, sweep_plan.CompiledSweepPhase)
    assert isinstance(plan.posterior, sweep_plan.CompiledSweepPhase)
    assert plan.warmup.executable is plan.posterior.executable
    assert not executed
    run = run_sweeps(plan, initial, execution_context=EXECUTION_CONTEXT)
    np.testing.assert_array_equal(run.state, np.full((4, 3), 8))
    assert run.history.posterior["value"].shape == (4, 5, 3)
    assert len(executed) == 3  # First-step timing, remaining warmup, posterior.
    with pytest.raises(MemoryError, match="One sweep lane"):
        sweeps.compile_sweeps(
            SweepProgram(step),
            initial,
            execution_context=replace(
                EXECUTION_CONTEXT,
                budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=100)),
            ),
            sampler=SamplerSpec(num_warmup=3, num_samples_per_chain=5),
        )


def test_history_budget_places_combined_diagnostics_on_host(monkeypatch):
    monkeypatch.setattr(
        sweeps, "compiled_memory", lambda _: MemoryDemand(host_bytes=0, device_bytes=64)
    )
    initial = _Carry(jax.random.key(791), jnp.zeros((2, 1024)))
    program = SweepProgram(_step)
    sampler = SamplerSpec(num_warmup=1, num_samples_per_chain=100)

    def context(device_bytes):
        return replace(
            EXECUTION_CONTEXT,
            budget=ComputeBudget(
                memory=AcceleratorMemoryBudget(host_bytes=1_100_000, device_bytes=device_bytes)
            ),
        )

    plan = sweeps.compile_sweeps(
        program, initial, execution_context=context(900_000), sampler=sampler
    )
    paths = 100 * 2 * 1024 * 4
    diagnostics = 101 * 2 * 4
    block = 25 * (2 * 1024 * 4 + 2 * 4)
    assert plan.memory.device_bytes == 64 + paths + diagnostics
    assert plan.memory.host_bytes == paths + 2 * diagnostics + block
    with pytest.raises(MemoryError, match="One sweep lane"):
        sweeps.compile_sweeps(program, initial, execution_context=context(800_000), sampler=sampler)


def test_adaptation_rolling_totals_match_window_and_freeze_after_warmup():
    spec = latent_adaptation(GuidedPathSpec())
    state = initialize_adaptation(jnp.ones((2, 3)))
    rng = np.random.default_rng(91)
    proposed = rng.integers(0, 2, (ADAPTATION_WINDOW + 8, 2, 3)).astype(bool)
    proposed[:, :, -1] = False
    accepted = proposed & rng.integers(0, 2, proposed.shape).astype(bool)
    frozen = state
    for index in range(ADAPTATION_WINDOW + 8):
        state = adapt_scales(
            state,
            spec,
            jnp.asarray(accepted[index], dtype=state.delta.dtype),
            jnp.asarray(proposed[index]),
            ADAPTATION_WINDOW + 4,
            jnp.full_like(state.delta, 10.0),
        )
        end = min(index + 1, ADAPTATION_WINDOW + 4)
        start = max(0, end - ADAPTATION_WINDOW)
        np.testing.assert_array_equal(state.accepted_total, accepted[start:end].sum(axis=0))
        np.testing.assert_array_equal(state.proposed_total, proposed[start:end].sum(axis=0))
        np.testing.assert_array_equal(state.delta[:, -1], np.ones(2))
        if index == ADAPTATION_WINDOW + 3:
            frozen = state
        if index >= ADAPTATION_WINDOW + 4:
            for before, after in zip(frozen, state, strict=True):
                np.testing.assert_array_equal(before, after)


def test_optional_particle_histories_preserve_draws_and_random_stream():
    from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
        excluded_particle_diagnostics,
    )

    def step(carry, posterior):
        updated, record = _step(carry, posterior)
        return updated, {
            **record,
            "latent_move_rms_per_t": updated.state,
            "selected_particle_per_t": updated.state,
        }

    initial = _Carry(jax.random.key(791), jnp.zeros((3, 4)))
    runs = []
    for enabled in (False, True):
        excluded = excluded_particle_diagnostics(
            ("regional_movement", "particle_identity") if enabled else ()
        )
        plan = sweeps.compile_sweeps(
            SweepProgram(step, excluded_history=excluded),
            initial,
            execution_context=EXECUTION_CONTEXT,
            sampler=SamplerSpec(num_warmup=2, num_samples_per_chain=3),
        )
        run = run_sweeps(plan, initial, execution_context=EXECUTION_CONTEXT)
        for history in (run.history.warmup, run.history.posterior):
            assert ("latent_move_rms_per_t" in history) is enabled
            assert ("selected_particle_per_t" in history) is enabled
        assert run.retained_draws == 3
        runs.append(run)

    np.testing.assert_array_equal(
        runs[0].history.posterior["paths"], runs[1].history.posterior["paths"]
    )
    np.testing.assert_array_equal(runs[0].state.state, runs[1].state.state)
    np.testing.assert_array_equal(
        jax.random.key_data(runs[0].state.key), jax.random.key_data(runs[1].state.key)
    )
