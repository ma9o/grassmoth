"""One forward-generation contract with dated interventions and durable histories."""

from datetime import UTC, date, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from grassmoth.artifacts.scenarios import InterventionSpec
from grassmoth.artifacts.simulation import SimulationReport, SimulationSpec
from grassmoth_server.actions.contracts.inputs import SimulateInput
from grassmoth_server.actions.contracts.requests import SimulateRequest
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from tests.support.action import empty_simulation_summary
from tests.support.inference import compile_model_fixture, particle_posterior
from tests.support.model import x_y_model

pytestmark = pytest.mark.contract


def test_simulation_request_is_a_dated_window_with_optional_interventions():
    request = SimulateRequest[GitOid](
        input=SimulateInput[GitOid](
            simulation=SimulationSpec(start=date(2026, 1, 1), horizon="12d"),
            dynamical_model_spec_ref="a" * 40,
        )
    )
    assert request.input.simulation.interventions == ()
    assert set(request.model_dump()) == {"action", "input", "reasoning"}
    assert set(request.input.model_dump()) == {
        "dynamical_model_spec_ref",
        "simulation",
        "seed",
        "num_replicates",
    }
    with pytest.raises(ValidationError, match="panel_ref"):
        request.input.revised(panel_ref="b" * 40)
    with pytest.raises(ValidationError, match="Extra inputs"):
        request.revised(comparison_panel_revision="b" * 40)


def test_interventions_are_placed_after_the_start_in_model_days():
    spec = SimulationSpec(
        start=date(2026, 1, 3),
        horizon="3d",
        interventions=(
            InterventionSpec(target="construct:x", value=2),
            InterventionSpec(target="construct:x", after="1d", value=4),
        ),
    )
    assert SimulationSpec.model_validate_json(spec.model_dump_json()) == spec
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    assert (spec.start_day(origin), spec.end_day(origin)) == (2.0, 5.0)
    assert [(event.time, event.value) for event in spec.assignments(origin)] == [
        (2.0, 2.0),
        (3.0, 4.0),
    ]


@pytest.mark.parametrize(
    "events",
    [
        [{"after": "-1d", "value": 2}],
        [{"after": "3d", "value": 2}],
        [{"after": "4d", "value": 2}],
        [{"after": "1d", "value": float("nan")}],
        [{"after": "1d", "value": float("inf")}],
        [{"after": "1d", "value": [[1, 2], [2, 4]]}],
        [{"kind": "hold", "start": 0, "end": 2, "value": 1}],
        [{"value": 1}, {"value": 2}],
        [{"after": "1d", "value": 1}, {"after": "24h", "value": 2}],
    ],
)
def test_intervention_offsets_values_and_conflicts_are_explicit(events):
    with pytest.raises(ValidationError):
        SimulationSpec.model_validate(
            {
                "start": "2026-01-01",
                "horizon": "3d",
                "interventions": [{"target": "construct:x", **event} for event in events],
            }
        )


def _response(origin: datetime = datetime(2026, 1, 1, tzinfo=UTC)):
    start = (datetime(2026, 1, 1, tzinfo=UTC) - origin).total_seconds() / 86400
    days = [start, start + 1, start + 2]

    result = {
        "outcome": "construct:y",
        "labels": {"construct:x": "Treatment", "construct:y": "Outcome"},
        "differences": NumericalArray.from_numpy(np.ones((100, 3))),
        "frame": [0, 2],
        "summary": {"mean": 1, "median": 1, "lower_95": 0, "upper_95": 2, "prob_positive": 0.9},
        "reference_mean": 3,
        "manifest_effects": {},
        "warnings": [],
    }

    evidence = {
        "time_origin": origin.isoformat(),
        "assignments": [{"target": "construct:x", "time": start, "value": 1}],
        "times": days,
        "draws": 100,
        "seed": 0,
        "state_ids": ["construct:x", "construct:y"],
        "observation_layout": {
            "variables": [
                {
                    "id": "indicator:y",
                    "name": "y",
                    "measurement_dtype": "continuous",
                    "aggregation": "last",
                    "observation_window": "1d",
                }
            ],
            "support_start_times": NumericalArray.from_numpy(np.asarray(days)[:, None] - 1),
            "support_end_times": NumericalArray.from_numpy(np.asarray(days)[:, None]),
            "mask": NumericalArray.from_numpy(np.ones((3, 1), dtype=bool)),
        },
        "parameter_draws": {},
        "arms": {
            "kind": "paired",
            "causal": result,
            "action": {
                "latent_paths": NumericalArray.from_numpy(np.ones((100, 3, 2))),
                "observations": NumericalArray.from_numpy(np.ones((100, 3, 1))),
            },
            "reference": {
                "latent_paths": NumericalArray.from_numpy(np.zeros((100, 3, 2))),
                "observations": NumericalArray.from_numpy(np.zeros((100, 3, 1))),
            },
        },
    }
    return {
        "evidence": evidence,
        "summary": empty_simulation_summary().model_dump(mode="json"),
        "law": AuthoredLawProvenance().model_dump(mode="json"),
        "fit_reliability": "converged",
    }


def test_simulation_report_keeps_computed_coordinates():
    report = SimulationReport.model_validate(_response())
    assert report.evidence.arms.kind == "paired"
    assert SimulationReport.model_validate(unpack_result(pack_result(report))) == report


def test_causal_absence_is_structural_and_paired_results_are_required():
    payload = _response()
    arms = payload["evidence"]["arms"]
    del arms["causal"]
    with pytest.raises(ValidationError, match="causal"):
        SimulationReport.model_validate(payload)
    arms["kind"] = "single"
    del arms["reference"]
    SimulationReport.model_validate(payload)
    arms["causal"] = {
        "kind": "not_evaluated",
        "code": "causal_effect",
        "subject": "causal_effect",
        "reason": "CAUSAL_EVALUATION_FAILED",
        "detail": "No retained fit.",
    }
    with pytest.raises(ValidationError):
        SimulationReport.model_validate(payload)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("current_panel", [False, True])
def test_runner_uses_model_time_binding_without_reading_panels(
    tmp_path, monkeypatch, conditioned, current_panel
):
    from importlib import import_module

    import jax.numpy as jnp

    from grassmoth.actions.fit.conditioning import condition_model
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth.models.ssm.inference.types import JointPosteriorDraws
    from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
    from grassmoth_server.actions.errors import ActionExecutionError
    from grassmoth_server.actions.runners import run_action
    from grassmoth_server.storage import paths as data
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.state import StudyState
    from tests.support.helpers import run_async, write_question
    from tests.support.inference import parameter_draws
    from tests.support.runner_fixtures import panel_frame, panel_metadata

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("ORIGIN")
    dynamical_model_spec = x_y_model()
    fit_origin = datetime(2024, 1, 2, tzinfo=UTC)
    if conditioned:
        dynamical_model_spec, _ = condition_model(
            dynamical_model_spec,
            compile_model_fixture(dynamical_model_spec),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(dynamical_model_spec, 2), jnp.zeros((2, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
            time_origin=fit_origin,
        )
    # An imported conditioned model must be executable without its original fit history.
    from grassmoth_server.study.persistence.writes import write_model_revision

    record = write_model_revision(
        store, dynamical_model_spec, derived_from={}, produced_by="edit_model"
    )

    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=2)},
    )
    state = StudyState().with_artifacts(
        [
            write_question(
                store,
                QuestionSpec(text="Effect on sleep", outcome=dynamical_model_spec.constructs[1].id),
            ),
            record,
            *([panel] if current_panel else []),
        ]
    )
    design = SimulationSpec(start=date(2026, 1, 1), horizon="2d")
    expected_origin = fit_origin if conditioned else design.start_instant
    calls = []

    def generate(_model, *, start, end, assignments, time_origin, compute_budget, seed, draws):
        assert (seed, draws) == (73, 1)
        calls.append(time_origin)
        assert time_origin == expected_origin
        assert (start, end) == (design.start_day(expected_origin), design.end_day(expected_origin))
        assert assignments == ()
        return ObservationPreflightFailure.rejected("checked model-owned coordinates")

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("Simulation must not read a data panel")

    monkeypatch.setattr(
        import_module("grassmoth.actions.simulate.action"),
        "generate_simulation_batch",
        generate,
    )
    monkeypatch.setattr("grassmoth_server.study.queries.data.read_data_history", unexpected_read)
    with pytest.raises(ActionExecutionError, match="checked model-owned coordinates"):
        run_async(
            run_action(
                "ORIGIN",
                SimulateRequest[GitOid](
                    input=SimulateInput[GitOid](
                        dynamical_model_spec_ref=record.revision,
                        simulation=design,
                        seed=73,
                        num_replicates=1,
                    )
                ),
                state,
            )
        )
    assert calls == [expected_origin]
