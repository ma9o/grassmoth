"""Shared CPU/GPU resource planning for numerical workloads.

``budget`` owns immutable memory and concurrency budgets. ``resources`` reads
host/device allocations at application boundaries. ``batching`` selects batch
widths from compiler estimates without executing the workload. Callers supply
their compilation function and algorithm-specific workspace estimates.
"""
