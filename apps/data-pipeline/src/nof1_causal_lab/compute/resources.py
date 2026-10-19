"""Application-boundary discovery of process/container and device resources."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import psutil

from nof1_causal_lab.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    MemoryDemand,
    SharedMemoryBudget,
)

# This is a planning reserve, independent of JAX's allocator fraction. GPU
# bytes_limit is the allocator's usable pool, not Modal's host-memory request.
MEMORY_BUDGET_FRACTION = 0.8
HOST_MEMORY_ALLOCATION_ENV = "NOF1_HOST_MEMORY_ALLOCATION_BYTES"


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
