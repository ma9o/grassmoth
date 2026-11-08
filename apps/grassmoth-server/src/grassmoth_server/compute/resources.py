"""Application-boundary discovery of process/container and device resources."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import psutil

from grassmoth.public import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    InferenceHardware,
    MemoryDemand,
    SharedMemoryBudget,
)

# This planning reserve covers sampling-estimate error and allocation overhead.
# Initialization and compiler/autotuning peaks do not enter launch sizing.
# The margin is independent of JAX's allocator fraction.
# GPU bytes_limit is the usable allocator pool, not Modal's host-memory request.
MEMORY_BUDGET_FRACTION = 0.8
HOST_MEMORY_ALLOCATION_ENV = "GRASSMOTH_HOST_MEMORY_ALLOCATION_BYTES"
FIT_HOST_MEMORY_MIB = 32768
FIT_DEVICE_MEMORY_FRACTION = "0.90"


def fit_compute_budget(hardware: InferenceHardware) -> ComputeBudget:
    """Plan before dispatch from fixed deployment allocations, without acquiring a GPU.

    Modal https://modal.com/docs/guide/gpu documents A100/H100 upgrades; use
    the smaller promised device. NVIDIA's product specifications give advertised
    GB, converted conservatively as decimal 10**9 bytes, not GiB. Capacities were
    verified 2026-11-08 against NVIDIA's virtualization GPU table, NIM support
    matrix and Modal's GPU guide. The remote allocator reserves 90%, then the
    existing runtime planning policy leaves 20% of that pool as headroom.
    https://www.nvidia.com/en-us/data-center/graphics-cards-for-virtualization/
    https://docs.nvidia.com/nim/large-language-models/2.0.13/reference/support-matrix.html
    https://docs.nvidia.com/brev/reference/gpu-types
    """
    if hardware is InferenceHardware.LOCAL:
        return read_compute_budget()
    gigabytes = {
        InferenceHardware.T4: 16,
        InferenceHardware.L4: 24,
        InferenceHardware.A10: 24,
        InferenceHardware.L40S: 48,
        InferenceHardware.A100: 40,
        InferenceHardware.A100_40GB: 40,
        InferenceHardware.A100_80GB: 80,
        InferenceHardware.RTX_PRO_6000: 96,
        InferenceHardware.H100: 80,
        InferenceHardware.H100_ONLY: 80,
        InferenceHardware.H200: 141,
        InferenceHardware.B200: 180,
    }[hardware]
    return ComputeBudget(
        memory=AcceleratorMemoryBudget(
            host_bytes=int(FIT_HOST_MEMORY_MIB * 1024**2 * MEMORY_BUDGET_FRACTION),
            device_bytes=int(
                gigabytes * 10**9 * float(FIT_DEVICE_MEMORY_FRACTION) * MEMORY_BUDGET_FRACTION
            ),
        )
    )


def _host_memory() -> tuple[int, int]:
    """Respect explicit deployment allocations and Linux cgroup v2 ceilings."""
    physical = psutil.virtual_memory()
    total, available = int(physical.total), int(physical.available)
    if sys.platform == "linux":
        cgroup = Path("/sys/fs/cgroup")
        if (cgroup / "cgroup.controllers").exists():
            memory_limit = (cgroup / "memory.max").read_text().strip()
            if memory_limit != "max":
                used = int((cgroup / "memory.current").read_text())
                total = min(total, int(memory_limit))
                available = min(available, max(0, int(memory_limit) - used))
    if HOST_MEMORY_ALLOCATION_ENV in os.environ:
        requested = int(os.environ[HOST_MEMORY_ALLOCATION_ENV])
        total = min(total, requested)
        available = min(available, max(0, requested - int(psutil.Process().memory_info().rss)))
    return total, available


def read_compute_budget() -> ComputeBudget:
    """Plan from total capacity; retain a separate availability check with 20% headroom."""
    host_total, host_available = _host_memory()
    host_bytes = int(host_total * MEMORY_BUDGET_FRACTION)
    host_occupied = host_bytes - int(host_available * MEMORY_BUDGET_FRACTION)
    device = jax.local_devices()[0]
    memory: SharedMemoryBudget | AcceleratorMemoryBudget
    if device.platform == "cpu":
        memory = SharedMemoryBudget(available_bytes=host_bytes)
        device_occupied = 0
    elif device.platform == "gpu":
        stats = device.memory_stats()
        if stats is None:
            raise RuntimeError("The GPU backend did not report its allocator memory budget")
        device_available = max(0, stats["bytes_limit"] - stats["bytes_in_use"])
        device_bytes = int(stats["bytes_limit"] * MEMORY_BUDGET_FRACTION)
        device_occupied = device_bytes - int(device_available * MEMORY_BUDGET_FRACTION)
        memory = AcceleratorMemoryBudget(
            host_bytes=host_bytes,
            device_bytes=device_bytes,
        )
    else:
        raise RuntimeError(f"Compute resource discovery does not support {device.platform!r}")
    return ComputeBudget(
        memory=memory,
        occupied=MemoryDemand(host_bytes=host_occupied, device_bytes=device_occupied),
    )
