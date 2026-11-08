"""Pure launch admission chooses counts without allocating compute or executing inference."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from grassmoth.public import (
    AcceleratorMemoryBudget,
    ComputeBudget,
    InferenceHardware,
    MemoryDemand,
    ParticleMgradMemoryReport,
    ParticleMgradMemoryShape,
    SamplerLaunchSpec,
    SamplerSpec,
    SharedMemoryBudget,
    estimate_particle_mgrad_memory,
)
from grassmoth_server.actions.contracts.identity import CallId
from grassmoth_server.actions.contracts.settings import FitSettingsSpec
from grassmoth_server.actions.fit import capacity
from grassmoth_server.actions.fit.execution import resolve_sampler_spec
from grassmoth_server.actions.fit.selection import FitContinuation
from grassmoth_server.compute.analysis_cache import AnalysisCacheEntry, AnalysisCall
from grassmoth_server.compute.resources import fit_compute_budget
from grassmoth_server.study.records import Rejected

pytestmark = pytest.mark.contract


@pytest.fixture
def shape(monkeypatch):
    value = ParticleMgradMemoryShape(
        time_points=30,
        state_dimensions=3,
        initial_dimensions=2,
        transition_dimensions=2,
        parameter_dimensions=2,
        observation_dimensions=1,
        free_path_coordinates=60,
        initial_covariance="diagonal",
        transition_covariance="diagonal",
        sweep_block_size=25,
    )
    monkeypatch.setattr(capacity, "particle_mgrad_memory_shape", Mock(return_value=value))
    return value


def _budget(host=10**15, device=10**15):
    return ComputeBudget(memory=AcceleratorMemoryBudget(host_bytes=host, device_bytes=device))


def _admit(launch, **options):
    if "previous" in options:
        options["previous"] = Mock(settings=options["previous"], retained_host_bytes=0)
    return capacity.admit_sampler(launch, Mock(), Mock(), **options)


@pytest.mark.parametrize(("axis", "expected"), [("particles", 512), ("parameters", 64)])
def test_auto_particle_count_respects_supported_bank_even_with_abundant_memory(
    shape, axis, expected
):
    payload = (
        {"algorithm": {"method": "mpgibbs", "n_parameter_particles": "auto"}}
        if axis == "parameters"
        else {
            "algorithm": {
                "method": "mpgibbs",
                "path_update": {"method": {"kind": "particle_mgrad", "num_particles": "auto"}},
            }
        }
    )
    launch = SamplerLaunchSpec.model_validate(payload)
    resolved = _admit(launch, budget=_budget())
    assert isinstance(resolved, capacity.FitAdmission)
    assert resolved.sampler == launch.resolve(expected)
    assert (f"N={expected}" if axis == "particles" else f"M={expected}") in resolved.detail


def test_auto_chains_selects_integer_capacity_and_preserves_fixed_counts(shape):
    launch = SamplerLaunchSpec(num_chains="auto", num_samples_per_chain=100)
    report = estimate_particle_mgrad_memory(shape, launch.resolve(7))
    assert isinstance(report, ParticleMgradMemoryReport)
    resolved = _admit(
        launch, budget=_budget(report.sampling.host_bytes, report.sampling.device_bytes)
    )
    assert isinstance(resolved, capacity.FitAdmission)
    assert resolved.sampler == launch.resolve(7)
    assert "C=7" in resolved.detail
    assert resolved.detail.count("100.00%") == 2


def test_initialization_peak_does_not_control_static_or_automatic_admission(shape, monkeypatch):
    launch = SamplerLaunchSpec(num_chains="auto", num_samples_per_chain=100)
    expected = launch.resolve(7)
    report = estimate_particle_mgrad_memory(shape, expected)
    assert isinstance(report, ParticleMgradMemoryReport)
    budget = _budget(report.sampling.host_bytes, report.sampling.device_bytes)
    initialization = MemoryDemand(host_bytes=10**15, device_bytes=10**15)

    def with_large_initialization(model, sampler):
        result = estimate_particle_mgrad_memory(model, sampler)
        assert isinstance(result, ParticleMgradMemoryReport)
        return result.revised(initialization=initialization, peak=initialization)

    monkeypatch.setattr(capacity, "estimate_particle_mgrad_memory", with_large_initialization)
    for configured in (SamplerLaunchSpec.from_resolved(expected), launch):
        admitted = _admit(configured, budget=budget)
        assert isinstance(admitted, capacity.FitAdmission)
        assert admitted.sampler == expected
        assert str(initialization.host_bytes) not in admitted.detail
    rejected = _admit(SamplerLaunchSpec.from_resolved(launch.resolve(8)), budget=budget)
    assert isinstance(rejected, Rejected)
    assert rejected.code == "FIT_CAPACITY_EXCEEDED"
    rejected_report = estimate_particle_mgrad_memory(shape, launch.resolve(8))
    assert isinstance(rejected_report, ParticleMgradMemoryReport)
    assert "required /" in rejected.detail
    assert "available" in rejected.detail
    assert "short by" in rejected.detail
    assert str(initialization.host_bytes) not in rejected.detail


def test_static_and_minimum_auto_reject_without_execution(shape):
    for launch in (SamplerLaunchSpec(), SamplerLaunchSpec(num_chains="auto")):
        rejected = _admit(launch, budget=_budget(0, 0))
        assert isinstance(rejected, Rejected)
        assert rejected.code == "FIT_CAPACITY_EXCEEDED"
    launch = SamplerLaunchSpec(num_chains=3)
    admitted = _admit(launch, budget=_budget())
    assert isinstance(admitted, capacity.FitAdmission)
    assert admitted.sampler == launch.resolve(2)


def test_continuation_keeps_chain_random_stream_and_kernel_rules(shape):
    previous = SamplerSpec(num_chains=3, seed=7, num_warmup=0)
    inherited = SamplerLaunchSpec.from_resolved(previous)
    assert isinstance(inherited, SamplerLaunchSpec)
    for changes in ({"num_chains": "auto"}, {"num_chains": 4}, {"seed": 8}):
        launch = inherited.revised(**changes)
        result = _admit(launch, budget=_budget(), previous=previous)
        assert isinstance(result, Rejected)
        assert result.code == "INCOMPATIBLE_CONTINUATION"
    launch = inherited.revised(algorithm=inherited.algorithm.revised(n_parameter_particles="auto"))
    assert isinstance(_admit(launch, budget=_budget(), previous=previous), capacity.FitAdmission)
    changed = launch.revised(
        algorithm=launch.algorithm.revised(parameter_proposal="pseudo_langevin")
    )
    rejected = _admit(changed, budget=_budget(), previous=previous)
    assert isinstance(rejected, Rejected)
    assert rejected.code == "CONTINUATION_REQUIRES_WARMUP"


def test_partial_request_rejects_two_explicit_auto_counts():
    with pytest.raises(ValidationError, match="At most one"):
        FitSettingsSpec.model_validate(
            {
                "num_chains": "auto",
                "algorithm": {"method": "mpgibbs", "n_parameter_particles": "auto"},
            }
        )


@pytest.mark.parametrize("metrics", [(), ("particle_weights", "particle_identity")])
def test_previous_cached_bytes_reserve_host_memory_independent_of_new_diagnostics(shape, metrics):
    previous = SamplerSpec()
    launch = SamplerLaunchSpec.from_resolved(previous)
    assert isinstance(launch, SamplerLaunchSpec)
    launch = launch.revised(algorithm=launch.algorithm.revised(diagnostic_metrics=metrics))
    report = estimate_particle_mgrad_memory(shape, launch.resolve(2))
    assert isinstance(report, ParticleMgradMemoryReport)
    cached_bytes = 10**6
    continuation = FitContinuation(
        settings=previous,
        full_fit=AnalysisCacheEntry(
            call=AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64)),
            sha256="b" * 64,
            size=cached_bytes,
        ),
    )
    assert continuation.retained_host_bytes == 3 * cached_bytes
    budget = _budget(
        report.sampling.host_bytes + 3 * cached_bytes - 1, report.sampling.device_bytes
    )
    assert isinstance(_admit(launch, budget=budget), capacity.FitAdmission)
    rejected = capacity.admit_sampler(launch, Mock(), Mock(), budget=budget, previous=continuation)
    assert isinstance(rejected, Rejected)
    assert rejected.code == "FIT_CAPACITY_EXCEEDED"
    assert "Occupied host=2.86 MiB" in rejected.detail
    assert "short by 1 bytes" in rejected.detail


def test_omissions_inherit_auto_but_a_second_merged_auto_is_rejected(monkeypatch):
    launch = SamplerLaunchSpec.model_validate(
        {"algorithm": {"method": "mpgibbs", "n_parameter_particles": "auto"}}
    )
    monkeypatch.setattr(
        "grassmoth_server.config.get_config", lambda: Mock(inference=Mock(sampler=launch))
    )
    assert resolve_sampler_spec(FitSettingsSpec()) == launch
    with pytest.raises(ValidationError, match="At most one"):
        resolve_sampler_spec(FitSettingsSpec(num_chains="auto"))
    previous = SamplerSpec(num_chains=7)
    assert resolve_sampler_spec(FitSettingsSpec(), previous) == SamplerLaunchSpec.from_resolved(
        previous
    )


def test_remote_hardware_budget_uses_fixed_host_and_conservative_device_pool(monkeypatch):
    monkeypatch.setattr(
        "grassmoth_server.compute.resources.read_compute_budget",
        Mock(side_effect=AssertionError("local discovery")),
    )
    for hardware in InferenceHardware:
        if hardware is not InferenceHardware.LOCAL:
            assert isinstance(fit_compute_budget(hardware).memory, AcceleratorMemoryBudget)
    budget = fit_compute_budget(InferenceHardware.L40S)
    assert isinstance(budget.memory, AcceleratorMemoryBudget)
    assert budget.memory.host_bytes == int(32768 * 1024**2 * 0.8)
    assert budget.memory.device_bytes == 48_000_000_000 * 9 // 10 * 8 // 10
    assert fit_compute_budget(InferenceHardware.A100) == fit_compute_budget(
        InferenceHardware.A100_40GB
    )


@pytest.mark.parametrize("shared", [False, True])
def test_capacity_log_and_errors_use_remaining_budget_after_occupied_memory(shape, shared):
    launch = SamplerLaunchSpec(hardware="local" if shared else "L40S", num_chains="auto")
    report = estimate_particle_mgrad_memory(shape, launch.resolve(5))
    assert isinstance(report, ParticleMgradMemoryReport)
    occupied = MemoryDemand(host_bytes=1234, device_bytes=5678)
    budget = ComputeBudget(
        memory=SharedMemoryBudget(
            available_bytes=report.sampling.host_bytes
            + report.sampling.device_bytes
            + occupied.host_bytes
            + occupied.device_bytes
        )
        if shared
        else AcceleratorMemoryBudget(
            host_bytes=report.sampling.host_bytes + occupied.host_bytes,
            device_bytes=report.sampling.device_bytes + occupied.device_bytes,
        ),
        occupied=occupied,
    )
    admitted = _admit(launch, budget=budget)
    assert isinstance(admitted, capacity.FitAdmission)
    assert admitted.sampler.num_chains == 5
    assert "N=64, M=8, C=5" in admitted.detail
    assert admitted.detail.count("100.00%") == (1 if shared else 2)
    assert ("shared host/device=" in admitted.detail) is shared
    rejected = _admit(launch.revised(num_chains=6), budget=budget)
    assert isinstance(rejected, Rejected)
    assert "120.00%" in rejected.detail
    assert "required /" in rejected.detail
    assert "available" in rejected.detail
    assert "short by" in rejected.detail
