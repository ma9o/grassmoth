"""Compiled blocks preserve transitions, phase outputs and explicit memory limits."""

import json
import logging
from functools import partial
from time import monotonic
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.sampling import DsmcSpec
from nof1_causal_lab.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    MemoryDemand,
    SharedMemoryBudget,
)
from nof1_causal_lab.models.ssm.inference.samplers import sweep_plan, sweeps
from nof1_causal_lab.models.ssm.inference.samplers.particle.adaptation import (
    ADAPTATION_WINDOW,
    adapt_scales,
    initialize_adaptation,
)
from nof1_causal_lab.models.ssm.inference.samplers.sweeps import run_sweeps
from tests.support.compute import COMPUTE_BUDGET

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
    caplog.set_level(logging.INFO, logger="nof1_causal_lab.models.ssm.inference.samplers.progress")
    initial = _Carry(jax.random.key(791), jnp.zeros((3, 4)))
    plan = sweeps.compile_sweeps(
        _factory,
        initial,
        num_warmup=warmup,
        num_samples=samples,
        max_batch_size=8,
        budget=COMPUTE_BUDGET,
    )
    assert plan.batch_size == 8
    run = run_sweeps(plan, initial, clock=monotonic, label="test")
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
            np.testing.assert_allclose(
                actual[name], np.asarray(expected).swapaxes(0, 1), rtol=1e-6, atol=1e-6
            )
    np.testing.assert_array_equal(
        jax.random.key_data(run.state[0]), jax.random.key_data(reference[0])
    )
    np.testing.assert_allclose(run.state[1], reference[1], atol=1e-6)
    assert "paths" not in run.history.warmup


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
        # History (384) plus one transferred block (240) fit; two full histories do not.
        memory=AcceleratorMemoryBudget(host_bytes=700, device_bytes=300_000)
        if accelerator
        else SharedMemoryBudget(available_bytes=300_000)
    )
    initial = jnp.zeros((4, 3))
    plan = sweeps.compile_sweeps(
        factory, initial, num_warmup=3, num_samples=5, max_batch_size=8, budget=budget
    )
    assert compiled_batches == [8, 4, 2]
    assert len(compiled_phases) == 3
    assert plan.batch_size == 2
    assert isinstance(plan.warmup, sweep_plan.CompiledSweepPhase)
    assert isinstance(plan.posterior, sweep_plan.CompiledSweepPhase)
    assert plan.warmup.executable is plan.posterior.executable
    assert not executed
    run = run_sweeps(plan, initial, clock=monotonic, label="test")
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
        )


def test_adaptation_rolling_totals_match_window_and_freeze_after_warmup():
    spec = DsmcSpec()
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
            jnp.asarray(accepted[index]),
            jnp.asarray(proposed[index]),
            ADAPTATION_WINDOW + 4,
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


def test_runtime_inputs_rebind_nested_observations_priors_geometry_and_tuning():
    """A compiled sweep accepts another same-layout fit's numerical inputs."""

    def factory(values):
        observations, location, scale, geometry, tuning = values
        prior = dist.Normal(location, scale)

        @partial(jax.jit, inline=jax.Inline.JAX_EARLY)
        def transition(state):
            position = geometry @ state + tuning * observations
            return position, {"density": prior.log_prob(position)}

        def step(state, _posterior):
            return transition(state)

        return lambda _batch: step

    initial = jnp.ones((2, 1))
    first = (jnp.zeros((2, 1)), jnp.asarray(0.0), jnp.asarray(1.0), jnp.eye(2), jnp.asarray(0.1))
    second = (
        jnp.ones((2, 1)),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
        jnp.eye(2) * 2,
        jnp.asarray(0.2),
    )
    plans = tuple(
        sweeps.compile_sweeps(
            factory(values),
            initial,
            num_warmup=1,
            num_samples=1,
            max_batch_size=1,
            budget=COMPUTE_BUDGET,
        )
        for values in (first, second)
    )
    first_phase, second_phase = (plan.posterior for plan in plans)
    assert isinstance(first_phase, sweep_plan.CompiledSweepPhase)
    assert isinstance(second_phase, sweep_plan.CompiledSweepPhase)
    state, actual = first_phase.executable(
        second_phase.arguments, initial, jnp.asarray(1), jnp.asarray(True)
    )
    expected_state, expected = factory(second)(1)(initial, jnp.asarray(True))
    np.testing.assert_array_equal(state, expected_state)
    np.testing.assert_array_equal(actual["density"][0], expected["density"])
