"""Cold worker payload round trips must preserve native distribution constructors."""

import json
import subprocess
import sys
from textwrap import dedent
from uuid import uuid4

import pytest

from nof1_causal_lab.artifacts.identity import scientific_id
from tests.support.helpers import graph_constructs, make_model

pytestmark = pytest.mark.contract


def test_prior_request_round_trips_through_a_cold_workflow_sandbox():
    model = make_model(["symptoms"]).model_dump(mode="json")
    parameter_id = scientific_id("parameter", "baseline")
    graph_constructs(model)[0]["coefficients"] = [
        {"kind": "coefficient", "role": "initial_mean", "value": parameter_id}
    ]
    model["parameters"] = {
        parameter_id: {
            "name": "baseline",
            "description": "Initial symptom burden",
            "distribution": "distribution:baseline",
        }
    }
    model["distributions"] = {
        "distribution:baseline": {
            "distribution": "Beta",
            # The facade emits typed arrays, whose dtype encoding lazily imports
            # NumPy even when decoding plain Python scalar arguments succeeds.
            "params": {
                "concentration1": {"array": 2.0, "dtype": "float32"},
                "concentration0": {"array": 2.0, "dtype": "float32"},
            },
        }
    }
    envelope = {
        "attempt_id": str(uuid4()),
        "request": {
            "action": "edit_model",
            "input": {"parent_ref": "3" * 40, "dynamical_model_spec": model},
        },
    }
    # Constructing this law before entering the sandbox warms JAX's lazy imports
    # and masks the rejection, so decode raw JSON in a fresh worker process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import asyncio
                import importlib
                import sys

                from temporalio.api.common.v1 import Payload
                from temporalio.contrib.pydantic import pydantic_data_converter
                from temporalio.worker.workflow_sandbox._importer import Importer
                from temporalio.worker.workflow_sandbox._restrictions import RestrictionContext

                from nof1_causal_service.actions.contracts.requests import FitRequest, SimulateRequest
                from nof1_causal_service.actions.contracts.inputs import NewFitTarget, FitInput, SimulateInput
                from nof1_causal_lab.artifacts.identity import GitOid
                from nof1_causal_lab.artifacts.simulation import SimulationSpec
                from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
                from nof1_causal_service.study.records import AttemptRecord, Applied, EditAttempt
                from nof1_causal_service.actions.temporal.study.messages import (
                    ActionInput, ActionRequest, EditModelActivityInput, EvaluateChecksInput, AttemptPublication,
                )
                from nof1_causal_service.study.state import ArtifactRecord, StudyState
                from nof1_causal_service.actions.contracts.effects import ActionEffects
                from nof1_causal_service.actions.temporal.worker import study_workflow_runner

                payload = Payload(metadata={"encoding": b"json/plain"}, data=sys.stdin.buffer.read())
                converter = pydantic_data_converter.payload_converter
                importer = Importer(study_workflow_runner().restrictions, RestrictionContext())
                with importer.applied():
                    importer.restriction_context.is_runtime = True
                    restored, = converter.from_payloads([payload], [ActionRequest])
                    model_record = ArtifactRecord(artifact_id="model", revision="1" * 40)
                    panel_record = ArtifactRecord(artifact_id="panel", revision="2" * 40)
                    state = StudyState(current={"model": model_record, "panel": panel_record})
                    inputs = [
                        EditModelActivityInput(workspace_id="test", request=restored.request),
                        # These activities carry revision references, not a DynamicalModelSpec.
                        ActionInput(
                            workspace_id="test", state=state,
                            request=FitRequest[GitOid](input=FitInput[GitOid](target=NewFitTarget(
                                dynamical_model_spec_ref=model_record.revision,
                                data_ref={"revision": panel_record.revision, "replicate_index": 0}),)),
                        ),
                        ActionInput(
                            workspace_id="test", state=state,
                            request=SimulateRequest[GitOid](input=SimulateInput[GitOid](
                                dynamical_model_spec_ref=model_record.revision,
                                simulation=SimulationSpec(start="2026-01-01", horizon="2d"),
                            )),
                        ),
                        EvaluateChecksInput(
                            workspace_id="test", state=state, request=restored.request,
                            applied=Applied(result=GitOid("3" * 40), effects=ActionEffects(produced=[model_record])),
                        ),
                    ]
                    activity_payloads = converter.to_payloads(inputs)
                    journal = AttemptPublication(
                        workspace_id="test",
                        record=AttemptRecord(
                            seq=1, ts="2026-10-01T00:00:00Z", attempt_id=restored.attempt_id,
                            attempt=EditAttempt(action="edit_model", request=restored.request,
                                outcome=Applied(result=GitOid("4" * 40), effects=ActionEffects(produced=[model_record]))),
                        ),
                    )
                    inputs.append(journal)
                    activity_payloads.extend(converter.to_payloads([journal]))
                    activity_inputs = converter.from_payloads(
                        activity_payloads, [type(value) for value in inputs]
                    )
                    assert converter.to_payloads(activity_inputs) == activity_payloads
                    update_payloads = converter.to_payloads([restored])
                    assert converter.from_payloads(update_payloads, [ActionRequest]) == [restored]
                # Sandbox configuration must not alter the existing wire representation.
                assert converter.to_payloads(inputs) == activity_payloads
                assert converter.to_payloads([restored]) == update_payloads
                law = restored.request.input.dynamical_model_spec.distributions["distribution:baseline"]
                assert type(law).__name__ == "Beta"
                assert law.batch_shape == law.event_shape == ()

                # A later activity imports the fitter in an ordinary thread,
                # without pytest's jaxtyping registry or the workflow importer.
                async def import_fitter():
                    module = await asyncio.to_thread(
                        importlib.import_module,
                        "nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.fit",
                    )
                    assert callable(module.fit_marginal_particle_gibbs)

                asyncio.run(import_fitter())
                print("round trip verified")
                """),
        ],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "round trip verified"


@pytest.mark.parametrize("action", ["fit", "simulate"])
def test_numerical_results_cross_workflows_only_as_saved_report_references(action):
    import numpy as np
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.arrays import NumericalArray
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from nof1_causal_lab.artifacts.simulation import SimulationEvidence
    from nof1_causal_service.actions.contracts.effects import ActionEffects
    from nof1_causal_service.actions.contracts.requests import FitRequest, SimulateRequest
    from nof1_causal_service.study.records import ActionAttempt, Applied, applied_attempt

    ref = {"workspace_id": "TEST", "revision": "1" * 40, "path": "model.json"}
    data = {"revision": "2" * 40, "replicate_index": 0}
    values = NumericalArray.from_numpy(np.ones((1, 2, 1)))
    if action == "fit":
        request = FitRequest.model_validate(
            {
                "input": {
                    "target": {
                        "kind": "new",
                        "dynamical_model_spec_ref": ref["revision"],
                        "data_ref": data,
                    }
                }
            }
        )
        InferenceEvidence(chain_extra_fields={"energy": values})
    else:
        design = {"start": "2026-01-01", "horizon": "1d"}
        request = SimulateRequest.model_validate(
            {"input": {"dynamical_model_spec_ref": ref["revision"], "simulation": design}}
        )
        SimulationEvidence.model_validate(
            {
                "assignments": [],
                "time_origin": "2026-01-01T00:00:00Z",
                "times": [0, 1],
                "draws": 1,
                "seed": 0,
                "state_ids": [],
                "parameter_draws": {},
                "arms": {
                    "kind": "single",
                    "action": {"latent_paths": values, "observations": values},
                },
                "observation_layout": {
                    "variables": [],
                    "support_start_times": values,
                    "support_end_times": values,
                    "mask": values,
                },
            }
        )
    staged = applied_attempt(
        request,
        Applied(
            result="3" * 40,
            effects=ActionEffects(),
        ),
    )
    payload = staged.model_dump_json()
    assert '"result":"' + "3" * 40 + '"' in payload
    assert '"npy"' not in payload
    assert TypeAdapter(ActionAttempt).validate_json(payload) == staged
