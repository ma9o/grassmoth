"""Admission debits survive failures; stale supervision and overspend stop before allocation."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.compute.modal_budget import (
    ModalFitBudget,
    reserve_allocation,
    save_allocation,
)

pytestmark = pytest.mark.contract


def test_budget_admission_is_atomic_and_does_not_refund_finished_calls(tmp_path):
    policy = ModalFitBudget(
        directory=tmp_path / "allocations",
        hold_file=tmp_path / "HOLD",
        hardware="L40S",
        timeout_seconds=10,
        startup_seconds=10,
        termination_seconds=10,
        usd_per_second=0.001,
        reservation_usd=1,
        max_reserved_usd=2,
        max_concurrent=2,
    )
    sampler = SamplerSpec()
    with pytest.raises(RuntimeError, match="live allocation watchdog"):
        reserve_allocation(policy, sampler)
    (policy.directory / "watchdog.heartbeat").touch()
    with ThreadPoolExecutor(max_workers=2) as pool:
        allocations = tuple(pool.map(lambda _: reserve_allocation(policy, sampler), range(2)))
    assert len({item.identity for item in allocations}) == 2
    from evaluation.watchdog import expired

    allocation = allocations[0]
    assert not expired(allocation, allocation.created_at + 19)
    assert expired(allocation, allocation.created_at + 20)
    assert allocation.policy.wall_seconds == 30  # Ten seconds remain for termination.
    for item in allocations:
        save_allocation(item.revised(finished_at=time.time()))
    with pytest.raises(RuntimeError, match="reserved stage budget"):
        reserve_allocation(policy, sampler)
    policy.hold_file.touch()
    with pytest.raises(RuntimeError, match="reviewer"):
        reserve_allocation(policy, sampler)
    with pytest.raises(ValueError, match="hardware"):
        reserve_allocation(policy, sampler.revised(hardware="H100"))


def test_isolated_data_root_does_not_change_the_repository_default(tmp_path, monkeypatch):
    from nof1_causal_lab.storage.backend import get_base_uri

    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setenv("NOF1_DATA_ROOT", str(tmp_path))
    assert get_base_uri() == str(tmp_path)
