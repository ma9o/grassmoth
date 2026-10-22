"""Shared immutable memory and concurrency budgets for numerical execution."""

from __future__ import annotations

from pydantic import Field

from grassmoth.artifacts.base import Value


class MemoryDemand(Value):
    """Host and device bytes required by one concurrent operation."""

    host_bytes: int = Field(ge=0)
    device_bytes: int = Field(ge=0)


NO_MEMORY_DEMAND = MemoryDemand(host_bytes=0, device_bytes=0)


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
    occupied: MemoryDemand = NO_MEMORY_DEMAND

    def capacity(
        self,
        demand: MemoryDemand,
        *,
        limit: int,
        overhead: MemoryDemand = NO_MEMORY_DEMAND,
    ) -> int:
        """Count operations that fit the unoccupied part of the allocation."""
        return self.planning_capacity(
            demand,
            limit=limit,
            overhead=MemoryDemand(
                host_bytes=overhead.host_bytes + self.occupied.host_bytes,
                device_bytes=overhead.device_bytes + self.occupied.device_bytes,
            ),
        )

    def planning_capacity(
        self,
        demand: MemoryDemand,
        *,
        limit: int,
        overhead: MemoryDemand = NO_MEMORY_DEMAND,
    ) -> int:
        """Count operations after reserving fixed host/device buffers once."""
        if isinstance(self.memory, SharedMemoryBudget):
            pools = (
                (
                    self.memory.available_bytes,
                    demand.host_bytes + demand.device_bytes,
                    overhead.host_bytes + overhead.device_bytes,
                ),
            )
        else:
            pools = (
                (self.memory.host_bytes, demand.host_bytes, overhead.host_bytes),
                (self.memory.device_bytes, demand.device_bytes, overhead.device_bytes),
            )
        if any(available < fixed for available, _, fixed in pools):
            return 0
        return min(
            (
                limit,
                *((available - fixed) // needed for available, needed, fixed in pools if needed),
            )
        )
