import json
import logging
from datetime import UTC, datetime
from time import monotonic
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.reports.checks import Evaluated, NumericCriterionEvidence
from grassmoth.artifacts.reports.posterior_diagnostics import ConvergenceReport
from grassmoth.artifacts.sampling import ConvergenceStoppingSpec, DsmcSpec
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
from grassmoth.models.ssm.inference.samplers.sweeps import run_sweeps
from grassmoth_test_support.compute import COMPUTE_BUDGET, prepare_program

"""Compiled blocks preserve transitions, phase outputs and explicit memory limits."""


pytestmark = pytest.mark.inference(concern="sampling")


class _Carry(NamedTuple):
    key: jax.Array
    state: jax.Array


def _factory(_batch):
    def step(carry, posterior):
        key, state = carry
        key, draw = jax.random.split(key)
        state = state + jax.random.normal(draw, state.shape) * jnp.where(posterior, 1, 2)
        record = {"telemetry": state.sum(axis=-1), "paths": state}
        return _Carry(key, state), record

    return step


@pytest.mark.parametrize(("warmup", "samples"), [(0, 29), (28, 27), (7, 0)])
def test_block_boundaries_and_padding_preserve_key_stream_and_phase_outputs(
    warmup, samples, caplog
):
    caplog.set_level(logging.INFO, logger="grassmoth.models.ssm.inference.samplers.progress")
    initial = _Carry(jax.random.key(791), jnp.zeros((3, 4)))
    plan = sweeps.compile_sweeps(
        _factory,
        initial,
        num_warmup=warmup,
        num_samples=samples,
        max_batch_size=8,
        budget=COMPUTE_BUDGET,
        prepare_program=prepare_program,
    )
    assert plan.batch_size == 8
    run = run_sweeps(
        plan,
        initial,
        clock=monotonic,
        label="test",
        logger=logging.getLogger("grassmoth.models.ssm.inference.samplers.progress"),
        wall_clock=lambda: datetime.now(UTC),
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
                lambda carry, _: _factory(8)(carry, jnp.asarray(phase == "posterior")),
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
    assert run.stopping_checkpoint is None


@pytest.mark.parametrize(
    ("interval", "minimum", "passing_at", "ceiling", "expected", "checkpoints"),
    [
        (7, 10, 7, 32, 14, [7, 14]),
        (7, 7, 21, 32, 21, [7, 14, 21]),
        (7, 7, 35, 32, 32, [7, 14, 21, 28]),
        (7, 40, 7, 32, 32, [7, 14, 21, 28]),
        (7, 7, 28, 28, 28, [7, 14, 21, 28]),
        (7, 7, 7, 0, 0, []),
    ],
)
def test_declared_checkpoints_preserve_prefix_and_never_exceed_ceiling(
    interval, minimum, passing_at, ceiling, expected, checkpoints
):
    initial = _Carry(jax.random.key(791), jnp.zeros((3, 4)))
    plan = sweeps.compile_sweeps(
        _factory,
        initial,
        num_warmup=11,
        num_samples=ceiling,
        max_batch_size=1,
        budget=COMPUTE_BUDGET,
        prepare_program=prepare_program,
    )
    observed = []

    def evaluate(history):
        count = history["paths"].shape[1]
        observed.append(count)
        return ConvergenceReport(
            findings=(
                Evaluated(
                    code="test_gate",
                    subject="recorded_parameter_chains",
                    outcome="passed" if count >= passing_at else "failed",
                    evidence=NumericCriterionEvidence(criterion="test", value=1.0),
                ),
            )
        )

    def execute(current_plan, convergence_check=None):
        return run_sweeps(
            current_plan,
            initial,
            convergence=convergence_check,
            clock=monotonic,
            label="test",
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
        )

    full = execute(plan)
    stopped = execute(
        plan,
        sweeps.RetainedConvergenceCheck(  # noqa: CORE001 -- Synthetic scheduling test composes a validated policy and callback, not compiled model evidence.
            settings=ConvergenceStoppingSpec(checkpoint_draws=interval, min_draws=minimum),
            evaluate=evaluate,
        ),
    )
    assert observed == checkpoints
    assert stopped.retained_draws == expected
    for name, value in stopped.history.warmup.items():
        np.testing.assert_array_equal(value, full.history.warmup[name])
    for name, value in stopped.history.posterior.items():
        np.testing.assert_array_equal(value, full.history.posterior[name][:, :expected])
    eligible_pass = (
        bool(checkpoints) and expected >= max(minimum, passing_at) and expected % interval == 0
    )
    if eligible_pass:
        assert stopped.stopping_checkpoint is not None
        assert stopped.stopping_checkpoint.draws == expected
        assert stopped.stopping_checkpoint.report.status == "passed"
    else:
        assert stopped.stopping_checkpoint is None
    # Same last transition and random stream as a fit whose requested count is this prefix.
    prefix_plan = sweeps.compile_sweeps(
        _factory,
        initial,
        num_warmup=11,
        num_samples=expected,
        max_batch_size=1,
        budget=COMPUTE_BUDGET,
        prepare_program=prepare_program,
    )
    prefix = execute(prefix_plan)
    np.testing.assert_array_equal(stopped.state.state, prefix.state.state)
    np.testing.assert_array_equal(
        jax.random.key_data(stopped.state.key), jax.random.key_data(prefix.state.key)
    )


@pytest.mark.parametrize("accelerator", [False, True])
def test_planner_uses_compiled_memory_without_executing_or_serializing_chains(
    monkeypatch, accelerator
):
    compiled_batches = []
    compiled_phases = []
    executed = []

    def factory(batch):
        compiled_batches.append(batch)

        def step(state, _posterior):
            return state + 1, {"value": state}

        return step

    def memory(executable):
        compiled_phases.append(executable)
        return MemoryDemand(host_bytes=0, device_bytes=compiled_batches[-1] * 100_000)

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
        factory,
        initial,
        num_warmup=3,
        num_samples=5,
        max_batch_size=8,
        budget=budget,
        prepare_program=prepare_program,
    )
    assert compiled_batches == [8, 4, 2]
    assert len(compiled_phases) == 3
    assert plan.batch_size == 2
    assert isinstance(plan.warmup, sweep_plan.CompiledSweepPhase)
    assert isinstance(plan.posterior, sweep_plan.CompiledSweepPhase)
    assert plan.warmup.executable is plan.posterior.executable
    assert not executed
    run = run_sweeps(
        plan,
        initial,
        clock=monotonic,
        label="test",
        logger=logging.getLogger("grassmoth.models.ssm.inference.samplers.progress"),
        wall_clock=lambda: datetime.now(UTC),
    )
    np.testing.assert_array_equal(run.state, np.full((4, 3), 8))
    assert run.history.posterior["value"].shape == (4, 5, 3)
    assert len(executed) == 3  # First-step timing, remaining warmup, posterior.
    with pytest.raises(MemoryError, match="One sweep lane"):
        sweeps.compile_sweeps(
            factory,
            initial,
            num_warmup=3,
            num_samples=5,
            max_batch_size=2,
            budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=100)),
            prepare_program=prepare_program,
        )


def test_adaptation_rolling_totals_match_window_and_freeze_after_warmup():
    spec = latent_adaptation(DsmcSpec())
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
