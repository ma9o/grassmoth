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

    assert get_config().inference.hardware is InferenceHardware.LOCAL
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
