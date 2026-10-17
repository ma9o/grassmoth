"""Modal dispatch uses the selected hardware and the unchanged resolved sampler."""

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import modal
import pytest

from nof1_causal_lab.actions.fit.remote import FitComputeInput, FitComputeResult, _dispatch_fit
from nof1_causal_lab.artifacts.sampling import InferenceHardware, NutsDASpec, SamplerSpec

pytestmark = pytest.mark.contract


def test_pytest_fits_locally_and_rejects_live_modal_apps():
    from nof1_causal_lab.config import get_config

    assert get_config().inference.sampler.hardware is InferenceHardware.LOCAL
    with pytest.raises(pytest.fail.Exception, match="Live Modal execution is disabled"):
        modal.App("pytest-local-only").run()


@pytest.mark.parametrize("hardware", [InferenceHardware.L40S, InferenceHardware.H100])
def test_modal_function_requests_selected_gpu_without_changing_joint_sampler(monkeypatch, hardware):
    payload = FitComputeInput(
        dynamical_model_spec_bytes=b"pinned-model",
        outcome=None,
        panel_parquet=b"pinned-history",
        variables=(),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        sampler=SamplerSpec(algorithm=NutsDASpec(), num_chains=4),
        compute_loo_diagnostics=False,
    )
    expected = FitComputeResult(b"fitted-model", b"evidence", "{}")
    remote = Mock()
    remote.remote.return_value = expected
    app = Mock()
    app.function.return_value = lambda _function: remote
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    assert _dispatch_fit(payload, hardware=hardware) is expected
    assert app.function.call_args.kwargs["gpu"] == hardware.value
    assert app.function.call_args.kwargs["timeout"] == 4 * 60 * 60
    remote.remote.assert_called_once_with(payload)


def test_bounded_dispatch_records_provider_ids_and_terminates_timed_out_call(tmp_path, monkeypatch):
    from nof1_causal_lab.compute.modal_budget import ModalAllocation, ModalFitBudget

    policy = ModalFitBudget(
        directory=tmp_path / "registry",
        hold_file=tmp_path / "HOLD",
        hardware="L40S",
        timeout_seconds=10,
        startup_seconds=10,
        termination_seconds=10,
        usd_per_second=0.001,
        reservation_usd=1,
        max_reserved_usd=1,
    )
    policy.directory.mkdir()
    (policy.directory / "watchdog.heartbeat").touch()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json())
    monkeypatch.setenv("NOF1_MODAL_FIT_BUDGET", str(policy_path))
    payload = FitComputeInput(
        b"model", None, b"history", (), datetime.now(UTC), SamplerSpec(), False
    )
    call = Mock(object_id="fc-budget-test", get=Mock(side_effect=TimeoutError("expired")))
    function = Mock()
    function.spawn.return_value = call
    app = Mock(app_id="ap-budget-test")
    app.function.return_value = lambda _function: function
    app.run.return_value = nullcontext()
    monkeypatch.setattr(modal, "App", lambda _name: app)
    with pytest.raises(TimeoutError, match="expired"):
        _dispatch_fit(payload, hardware=InferenceHardware.L40S)
    call.cancel.assert_called_once_with(terminate_containers=True)
    assert app.function.call_args.kwargs["cpu"] == (8, 8)
    assert app.function.call_args.kwargs["memory"] == (32768, 32768)
    record = ModalAllocation.model_validate_json(next(policy.directory.glob("*.json")).read_text())
    assert record.app_id == "ap-budget-test"
    assert record.function_call_id == "fc-budget-test"
    assert record.finished_at is not None
