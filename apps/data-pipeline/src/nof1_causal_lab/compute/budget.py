"""Shared immutable memory and concurrency budgets for numerical execution."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class MemoryDemand(Value):
    """Host and device bytes required by one concurrent operation."""

    host_bytes: int = Field(ge=0)
    device_bytes: int = Field(ge=0)


class SharedMemoryBudget(Value):
    """CPU execution and Python workspaces share one memory allocation."""

    available_bytes: int = Field(ge=0)


class AcceleratorMemoryBudget(Value):
    """GPU execution and host workspaces have separate allocations."""

    host_bytes: int = Field(ge=0)
    device_bytes: int = Field(ge=0)


class ComputeBudget(Value):
    """Resolved resources, with headroom already reserved at the application boundary."""

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
