"""Runner requests keep their concrete type through Temporal."""

from datetime import date

import pytest

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget, SimulateInput
from grassmoth_server.actions.contracts.requests import FitRequest, SimulateRequest
from grassmoth_server.actions.temporal.client import pydantic_data_converter
from grassmoth_server.actions.temporal.study.messages import ActionInput
from grassmoth_server.study.state import StudyState
from tests.support.git import git_oid
from tests.support.helpers import run_async

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
