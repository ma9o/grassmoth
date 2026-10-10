"""Resource discovery respects container requests and distinguishes RAM from VRAM."""

from unittest.mock import Mock

import pytest

from nof1_causal_lab.compute import resources
from nof1_causal_lab.compute.budget import AcceleratorMemoryBudget, SharedMemoryBudget

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_resource_budget_reserves_headroom_in_each_memory_pool(monkeypatch, platform):
    monkeypatch.setattr(resources, "_host_resources", lambda: (8, 1000))
    device = Mock(platform=platform)
    device.memory_stats.return_value = {"bytes_limit": 2000, "bytes_in_use": 500}
    monkeypatch.setattr(resources.jax, "local_devices", lambda: [device])
    budget = resources.read_compute_budget()
    assert budget.cpu_workers == 8
    if platform == "cpu":
        assert budget.memory == SharedMemoryBudget(available_bytes=800)
        device.memory_stats.assert_not_called()
    else:
        assert budget.memory == AcceleratorMemoryBudget(host_bytes=800, device_bytes=1200)


def test_modal_requests_and_cgroup_limits_bound_host_resources(monkeypatch, tmp_path):
    (tmp_path / "cgroup.controllers").write_text("cpu memory")
    (tmp_path / "cpu.max").write_text("600000 100000")
    (tmp_path / "memory.max").write_text("10000")
    (tmp_path / "memory.current").write_text("3000")
    monkeypatch.setattr(resources, "Path", Mock(return_value=tmp_path))
    monkeypatch.setattr(resources, "sys", Mock(platform="linux"))
    monkeypatch.setattr(
        resources,
        "os",
        Mock(
            cpu_count=lambda: 64,
            sched_getaffinity=lambda _pid: set(range(32)),
            environ={
                resources.CPU_ALLOCATION_ENV: "8",
                resources.HOST_MEMORY_ALLOCATION_ENV: "6000",
            },
        ),
    )
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: Mock(available=100000))
    process = Mock()
    process.memory_info.return_value = Mock(rss=1000)
    monkeypatch.setattr(resources.psutil, "Process", lambda: process)
    assert resources._host_resources() == (6, 5000)
