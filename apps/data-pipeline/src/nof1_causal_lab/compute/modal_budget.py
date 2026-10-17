"""Deployment-owned Modal admission limits and durable allocation records."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.sampling import InferenceHardware, SamplerSpec


class ModalFitBudget(Value):
    """An operator's bounded allocation policy, separate from scientific fit settings."""

    directory: Path
    hold_file: Path
    hardware: InferenceHardware
    timeout_seconds: int = Field(ge=1)
    startup_seconds: int = Field(default=120, ge=1)
    termination_seconds: int = Field(default=60, ge=1)
    usd_per_second: float = Field(gt=0, allow_inf_nan=False)
    reservation_usd: float = Field(gt=0, allow_inf_nan=False)
    max_reserved_usd: float = Field(gt=0, allow_inf_nan=False)
    max_concurrent: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def covers_lifetime(self) -> Self:
        """Every reservation covers the requested startup, execution and shutdown ceilings."""
        if self.hardware == InferenceHardware.LOCAL:
            raise ValueError("A Modal budget must select a remote GPU")
        if self.reservation_usd < self.usd_per_second * self.wall_seconds:
            raise ValueError("Reservation does not cover the bounded allocation lifetime")
        return self

    @property
    def wall_seconds(self) -> int:
        """Outer watchdog ceiling, including provisioning and termination headroom."""
        return self.startup_seconds + self.timeout_seconds + self.termination_seconds


class ModalAllocation(Value):
    """Admission and provider identifiers retained even when a fit never returns evidence."""

    identity: str
    created_at: float
    policy: ModalFitBudget
    sampler: SamplerSpec
    app_id: str | None = None
    function_call_id: str | None = None
    finished_at: float | None = None

    @property
    def app_name(self) -> str:
        """Unique name lets the watchdog find an app while image construction is still pending."""
        return "nof1-bench-" + self.identity

    @property
    def path(self) -> Path:
        """One atomically replaced record per allocation, below the operator's registry."""
        return self.policy.directory / (self.identity + ".json")


def configured_budget() -> ModalFitBudget | None:
    """Read optional deployment admission policy; it never changes sampler mathematics."""
    path = os.environ.get("NOF1_MODAL_FIT_BUDGET")
    return ModalFitBudget.model_validate_json(Path(path).read_text()) if path is not None else None


def save_allocation(allocation: ModalAllocation) -> None:
    """Publish a complete record atomically for the out-of-process watchdog."""
    temporary = allocation.path.with_suffix(".tmp")
    temporary.write_text(allocation.model_dump_json(indent=2))
    temporary.replace(allocation.path)


def reserve_allocation(policy: ModalFitBudget, sampler: SamplerSpec) -> ModalAllocation:
    """Debit the whole reservation before allocation; failures and retries keep their debit."""
    if sampler.hardware != policy.hardware:
        raise ValueError("Requested GPU differs from the budgeted hardware")
    policy.directory.mkdir(parents=True, exist_ok=True)
    with (policy.directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if policy.hold_file.exists():
            raise RuntimeError("Paid dispatch held by the campaign reviewer")
        heartbeat = policy.directory / "watchdog.heartbeat"
        if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15:
            raise RuntimeError("Paid dispatch requires a live allocation watchdog")
        allocations = tuple(
            ModalAllocation.model_validate_json(path.read_text())
            for path in policy.directory.glob("*.json")
        )
        reserved = sum(item.policy.reservation_usd for item in allocations)
        if reserved + policy.reservation_usd > policy.max_reserved_usd + 1e-9:
            raise RuntimeError("Modal allocation would exceed the reserved stage budget")
        if sum(item.finished_at is None for item in allocations) >= policy.max_concurrent:
            raise RuntimeError("Modal allocation would exceed campaign concurrency")
        allocation = ModalAllocation(
            identity=uuid4().hex,
            created_at=time.time(),
            policy=policy,
            sampler=sampler,
        )
        save_allocation(allocation)
        return allocation
