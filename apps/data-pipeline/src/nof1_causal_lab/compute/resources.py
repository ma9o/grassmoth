"""Application-boundary discovery of process/container and device resources."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import jax
import psutil

from nof1_causal_lab.compute.budget import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    SharedMemoryBudget,
)

# This is a planning reserve, independent of JAX's allocator fraction. GPU
# bytes_limit is the allocator's usable pool, not Modal's host-memory request.
MEMORY_BUDGET_FRACTION = 0.8
CPU_ALLOCATION_ENV = "NOF1_CPU_ALLOCATION"
HOST_MEMORY_ALLOCATION_ENV = "NOF1_HOST_MEMORY_ALLOCATION_BYTES"


def _host_resources() -> tuple[int, int]:
    """Respect explicit deployment allocations and Linux cgroup v2 ceilings."""
    cores = os.cpu_count()
    if cores is None:
        raise RuntimeError("The operating system did not report a CPU allocation")
    available = int(psutil.virtual_memory().available)
    if sys.platform == "linux":
        cores = min(cores, len(os.sched_getaffinity(0)))
        cgroup = Path("/sys/fs/cgroup")
        if (cgroup / "cgroup.controllers").exists():
            quota, period = (cgroup / "cpu.max").read_text().split()
            if quota != "max":
                cores = min(cores, max(1, math.floor(int(quota) / int(period))))
            memory_limit = (cgroup / "memory.max").read_text().strip()
            if memory_limit != "max":
                used = int((cgroup / "memory.current").read_text())
                available = min(available, max(0, int(memory_limit) - used))
    if CPU_ALLOCATION_ENV in os.environ:
        cores = min(cores, int(os.environ[CPU_ALLOCATION_ENV]))
    if HOST_MEMORY_ALLOCATION_ENV in os.environ:
        requested = int(os.environ[HOST_MEMORY_ALLOCATION_ENV])
        available = min(available, max(0, requested - int(psutil.Process().memory_info().rss)))
    return cores, available


def read_compute_budget() -> ComputeBudget:
    """Reserve 20% of available host and JAX allocator memory before sizing work."""
    workers, host_available = _host_resources()
    host_bytes = int(host_available * MEMORY_BUDGET_FRACTION)
    device = jax.local_devices()[0]
    memory: SharedMemoryBudget | AcceleratorMemoryBudget
    if device.platform == "cpu":
        memory = SharedMemoryBudget(available_bytes=host_bytes)
    elif device.platform == "gpu":
        stats = device.memory_stats()
        if stats is None:
            raise RuntimeError("The GPU backend did not report its allocator memory budget")
        device_available = max(0, stats["bytes_limit"] - stats["bytes_in_use"])
        memory = AcceleratorMemoryBudget(
            host_bytes=host_bytes,
            device_bytes=int(device_available * MEMORY_BUDGET_FRACTION),
        )
    else:
        raise RuntimeError(f"Compute resource discovery does not support {device.platform!r}")
    return ComputeBudget(cpu_workers=workers, memory=memory)
