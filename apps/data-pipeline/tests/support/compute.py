"""Explicit resource budgets keep numerical tests independent of the host machine."""

from nof1_causal_lab.compute.budget import ComputeBudget, SharedMemoryBudget

COMPUTE_BUDGET = ComputeBudget(
    memory=SharedMemoryBudget(available_bytes=256 * 1024**2),
)
