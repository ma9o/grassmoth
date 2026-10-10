"""Shared budget and batch-planning contracts without scientific execution."""

from unittest.mock import Mock

import pytest

from nof1_causal_lab.compute.batching import compile_batched
from nof1_causal_lab.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    MemoryDemand,
    SharedMemoryBudget,
)

pytestmark = pytest.mark.contract


def test_shared_memory_adds_host_and_device_workspaces_and_partitions_concurrency():
    demand = MemoryDemand(host_bytes=30, device_bytes=40)
    shared = ComputeBudget(cpu_workers=4, memory=SharedMemoryBudget(available_bytes=200))
    separate = ComputeBudget(
        cpu_workers=4, memory=AcceleratorMemoryBudget(host_bytes=200, device_bytes=200)
    )
    assert shared.capacity(demand, limit=4) == 2
    assert separate.capacity(demand, limit=4) == 4
    assert shared.per_worker(2, host_workspace_bytes=40).capacity(demand, limit=1) == 0
    assert separate.per_worker(2, host_workspace_bytes=40).capacity(demand, limit=1) == 1


def test_batch_width_adapts_to_memory_and_workload_without_executing():
    executable = Mock(side_effect=AssertionError("Planning must not execute numerical kernels"))

    def compile_batch(size):
        return executable, MemoryDemand(host_bytes=0, device_bytes=20 * size)

    sizes = tuple(
        compile_batched(
            compile_batch,
            max_batch_size=12,
            budget=ComputeBudget(cpu_workers=1, memory=SharedMemoryBudget(available_bytes=memory)),
        ).batch_size
        for memory in (100, 200, 1000)
    )
    assert sizes == (4, 8, 12)
    executable.assert_not_called()


def test_compiled_memory_is_verified_when_workspaces_scale_non_linearly():
    widths = []
    executable = Mock(side_effect=AssertionError("Planning must not execute numerical kernels"))

    def compile_batch(size):
        widths.append(size)
        return executable, MemoryDemand(host_bytes=0, device_bytes=10 * size**2)

    selected = compile_batched(
        compile_batch,
        max_batch_size=16,
        budget=ComputeBudget(cpu_workers=1, memory=SharedMemoryBudget(available_bytes=100)),
    )
    assert selected.batch_size == 2
    assert widths == [1, 8, 4, 2]
    executable.assert_not_called()


def test_impossible_single_lane_fails_before_execution():
    executable = Mock(side_effect=AssertionError("Planning must not execute numerical kernels"))
    with pytest.raises(MemoryError, match="One batch lane"):
        compile_batched(
            lambda _size: (executable, MemoryDemand(host_bytes=30, device_bytes=40)),
            max_batch_size=4,
            budget=ComputeBudget(cpu_workers=1, memory=SharedMemoryBudget(available_bytes=50)),
        )
    executable.assert_not_called()


def test_predictive_compiled_batch_preserves_aligned_draws_with_a_stub_solver(monkeypatch):
    import jax.numpy as jnp
    import numpy as np

    from nof1_causal_lab.compute.batching import compiled_memory
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    # The solver is replaced with arithmetic markers: this exercises the actual
    # partition/map/compile boundary without generating any scientific trajectory.
    def markers(model, times, key, span, rate):
        del span
        return jnp.broadcast_to(model + key[0] + rate, (times.size, 1))

    monkeypatch.setattr(registry_runtime, "_simulate_model_predictive_latent_draw", markers)
    args = (
        jnp.arange(5.0),
        jnp.arange(3.0),
        jnp.arange(10.0).reshape(5, 2),
        2.0,
        jnp.arange(5.0) * 10,
    )
    for width in (2, 4):
        executable = registry_runtime._simulate_model_predictive_draws.lower(
            *args, batch_size=width
        ).compile()
        assert compiled_memory(executable.compiled).device_bytes > 0
        actual = executable(*args, batch_size=width)
        np.testing.assert_array_equal(
            actual[:, :, 0], np.arange(5.0)[:, None] * 13 + np.zeros((5, 3))
        )
