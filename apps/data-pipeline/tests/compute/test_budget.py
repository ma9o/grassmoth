"""Shared budget and batch-planning contracts without scientific execution."""

from unittest.mock import Mock

import pytest

from nof1_causal_lab.compute.batching import compile_batched, compile_from_estimate
from nof1_causal_lab.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    MemoryDemand,
    SharedMemoryBudget,
)

pytestmark = pytest.mark.contract


def test_shared_memory_adds_host_and_device_workspaces():
    demand = MemoryDemand(host_bytes=30, device_bytes=40)
    shared = ComputeBudget(memory=SharedMemoryBudget(available_bytes=200))
    separate = ComputeBudget(memory=AcceleratorMemoryBudget(host_bytes=200, device_bytes=200))
    assert shared.capacity(demand, limit=4) == 2
    assert separate.capacity(demand, limit=4) == 4
    overhead = MemoryDemand(host_bytes=100, device_bytes=10)
    assert shared.capacity(demand, limit=4, overhead=overhead) == 1
    assert separate.capacity(demand, limit=4, overhead=overhead) == 3
    assert (
        shared.capacity(
            MemoryDemand(host_bytes=0, device_bytes=0),
            limit=4,
            overhead=MemoryDemand(host_bytes=201, device_bytes=0),
        )
        == 0
    )


def test_batch_width_adapts_to_memory_and_workload_without_executing():
    executable = Mock(side_effect=AssertionError("Planning must not execute numerical kernels"))

    def compile_batch(size):
        return executable, MemoryDemand(host_bytes=0, device_bytes=20 * size)

    sizes = tuple(
        compile_batched(
            compile_batch,
            max_batch_size=12,
            budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=memory)),
        ).batch_size
        for memory in (100, 200, 1000)
    )
    assert sizes == (4, 8, 12)
    executable.assert_not_called()


def test_width_depends_on_total_capacity_and_busy_memory_rejects_before_execution():
    widths = []
    executable = Mock(side_effect=AssertionError("Planning must not execute numerical kernels"))

    def compile_batch(size):
        widths.append(size)
        return executable, MemoryDemand(host_bytes=0, device_bytes=10 * size)

    for occupied in (0, 30):
        selected = compile_from_estimate(
            compile_batch,
            single_memory=MemoryDemand(host_bytes=0, device_bytes=20),
            max_batch_size=16,
            budget=ComputeBudget(
                memory=SharedMemoryBudget(available_bytes=100),
                occupied=MemoryDemand(host_bytes=occupied, device_bytes=0),
            ),
        )
        assert selected.batch_size == 4
    with pytest.raises(MemoryError, match="currently available"):
        compile_from_estimate(
            compile_batch,
            single_memory=MemoryDemand(host_bytes=0, device_bytes=20),
            max_batch_size=16,
            budget=ComputeBudget(
                memory=SharedMemoryBudget(available_bytes=100),
                occupied=MemoryDemand(host_bytes=70, device_bytes=0),
            ),
        )
    assert widths == [4, 4, 4]
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
        budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=100)),
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
            budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=50)),
        )
    executable.assert_not_called()


def test_arithmetic_sizing_compiles_only_selected_width_and_checks_actual_memory():
    widths = []

    def compile_batch(size):
        widths.append(size)
        return size, MemoryDemand(host_bytes=0, device_bytes=20 * size)

    batch = compile_from_estimate(
        compile_batch,
        single_memory=MemoryDemand(host_bytes=0, device_bytes=10),
        max_batch_size=32,
        budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=100)),
    )
    assert widths == [8, 4]
    assert batch.batch_size == 4
    # An unfused estimate can overstate memory: only compiled demand rejects size 1.
    widths.clear()
    batch = compile_from_estimate(
        compile_batch,
        single_memory=MemoryDemand(host_bytes=0, device_bytes=1000),
        max_batch_size=32,
        budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=100)),
    )
    assert widths == [1]
    assert batch.batch_size == 1


def test_traced_memory_scales_parallel_arrays_and_reuses_sequential_workspace():
    import jax
    import jax.numpy as jnp

    from nof1_causal_lab.compute.traced_memory import traced_memory

    def dense_markers(values):
        return jnp.sin(values[:, None] * values[None, :]).sum(axis=1)

    single = jax.make_jaxpr(dense_markers)(jax.ShapeDtypeStruct((16,), jnp.float32))
    parallel = jax.make_jaxpr(jax.vmap(dense_markers))(jax.ShapeDtypeStruct((4, 16), jnp.float32))
    assert traced_memory(parallel).device_bytes == 4 * traced_memory(single).device_bytes

    def loop(size):
        return jax.make_jaxpr(
            lambda initial: jax.lax.scan(
                lambda carry, _: (dense_markers(carry), carry), initial, None, length=size
            )
        )(jax.ShapeDtypeStruct((16,), jnp.float32))

    small, large = traced_memory(loop(2)), traced_memory(loop(20))
    assert large.device_bytes - small.device_bytes == 18 * 16 * 4
    assert small.device_bytes > 16 * 16 * 4  # Nested pairwise workspace is included.


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
