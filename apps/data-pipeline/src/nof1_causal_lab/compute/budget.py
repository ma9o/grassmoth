"""Shared immutable memory and concurrency budgets for numerical execution."""

from __future__ import annotations

from typing import Self

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class MemoryDemand(Value):
    """Host and device bytes required by one concurrent operation."""

    host_bytes: int = Field(ge=0)
    device_bytes: int = Field(ge=0)

    def with_host_workspace(self, size: int) -> Self:
        """Include Python/NumPy workspaces outside the compiled executable."""
        return type(self)(host_bytes=self.host_bytes + size, device_bytes=self.device_bytes)


class SharedMemoryBudget(Value):
    """CPU execution and Python workspaces share one memory allocation."""

    available_bytes: int = Field(ge=0)


class AcceleratorMemoryBudget(Value):
    """GPU execution and host workspaces have separate allocations."""

    host_bytes: int = Field(ge=0)
    device_bytes: int = Field(ge=0)


class ComputeBudget(Value):
    """Resolved resources, with headroom already reserved at the application boundary."""

    cpu_workers: int = Field(ge=1)
    memory: SharedMemoryBudget | AcceleratorMemoryBudget

    def capacity(self, demand: MemoryDemand, *, limit: int) -> int:
        """Count operations that fit, including shared host/device memory on CPUs."""
        if isinstance(self.memory, SharedMemoryBudget):
            pools = ((self.memory.available_bytes, demand.host_bytes + demand.device_bytes),)
        else:
            pools = (
                (self.memory.host_bytes, demand.host_bytes),
                (self.memory.device_bytes, demand.device_bytes),
            )
        return min((limit, *(available // needed for available, needed in pools if needed)))

    def per_worker(self, workers: int, *, host_workspace_bytes: int = 0) -> Self:
        """Partition one budget among concurrent starts and reserve their host workspaces."""
        memory: SharedMemoryBudget | AcceleratorMemoryBudget
        if isinstance(self.memory, SharedMemoryBudget):
            memory = SharedMemoryBudget(
                available_bytes=max(
                    0, self.memory.available_bytes // workers - host_workspace_bytes
                )
            )
        else:
            memory = AcceleratorMemoryBudget(
                host_bytes=max(0, self.memory.host_bytes // workers - host_workspace_bytes),
                device_bytes=self.memory.device_bytes // workers,
            )
        return type(self)(cpu_workers=1, memory=memory)
