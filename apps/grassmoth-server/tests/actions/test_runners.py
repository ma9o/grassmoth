"""Runner requests keep their concrete type through Temporal."""

from datetime import date

import pytest

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth_server.actions.contracts.inputs import (
    ContinueFitTarget,
    FitInput,
    NewFitTarget,
    SimulateInput,
)
from grassmoth_server.actions.contracts.requests import FitRequest, SimulateRequest
from grassmoth_server.actions.temporal.client import pydantic_data_converter
from grassmoth_server.actions.temporal.study.messages import ActionInput
from grassmoth_server.study.state import StudyState
from grassmoth_server_test_support.helpers import run_async
from grassmoth_test_support.git import git_oid

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "action",
    [
        FitRequest[GitOid](
            input=FitInput[GitOid](
                target=NewFitTarget(
                    dynamical_model_spec_ref=git_oid(1),
                    data_ref=DataRef(revision=git_oid(2), replicate_index=0),
                ),
            )
        ),
        SimulateRequest[GitOid](
            input=SimulateInput[GitOid](
                simulation=SimulationSpec(
                    start=date(2026, 1, 1),
                    horizon="1w",
                    interventions=({"target": "construct:x", "after": "2d", "value": 1},),
                ),
                dynamical_model_spec_ref=git_oid(1),
            )
        ),
    ],
)
def test_action_inputs_keep_their_request_through_temporal(action):
    original = ActionInput(workspace_id="test", request=action, state=StudyState())

    async def round_trip():
        encoded = await pydantic_data_converter.encode([original])
        return await pydantic_data_converter.decode(encoded, [ActionInput])

    assert run_async(round_trip()) == [original]


@pytest.mark.parametrize("hardware", ["local", "L40S"])
def test_unavailable_continuation_kernel_stops_before_data_loading_or_dispatch(
    monkeypatch, hardware
):
    from unittest.mock import Mock

    from grassmoth.public import SamplerSpec
    from grassmoth_server.actions import call_logs, runners
    from grassmoth_server.actions.contracts.requests import call_identity
    from grassmoth_server.actions.fit import capacity, remote
    from grassmoth_server.study.queries import data
    from grassmoth_server.study.records import Rejected

    sampler = SamplerSpec.model_validate(
        {
            "hardware": hardware,
            "algorithm": {"method": "nuts_da"},
        }
    )
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=ContinueFitTarget(fit_ref=git_oid(3)),
        )
    )
    monkeypatch.setattr(runners, "fit_source", Mock())
    monkeypatch.setattr(runners, "prior_fit", Mock(return_value=Mock(settings=sampler)))
    admission, dispatch, emit = Mock(), Mock(), Mock()
    monkeypatch.setattr(capacity, "admit_sampler", admission)
    monkeypatch.setattr(remote, "fit_on_modal", dispatch)
    monkeypatch.setattr(call_logs, "emit_message", emit)
    read = Mock(side_effect=AssertionError("Unsupported settings must stop before data loading"))
    monkeypatch.setattr(data, "read_data_history", read)
    result = runners._fit_selected(Mock(), {}, request.input, call_identity(request))
    assert isinstance(result, Rejected)
    assert result.code == "INCOMPATIBLE_CONTINUATION"
    assert "only MPGibbs" in result.detail
    read.assert_not_called()
    admission.assert_not_called()
    dispatch.assert_not_called()
    emit.assert_not_called()
