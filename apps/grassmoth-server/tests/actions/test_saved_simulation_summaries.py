"""Saved scientific summaries retain masks, paired uncertainty and absolute time."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from grassmoth.actions.contracts.bodies import SimulateOutput
from grassmoth.actions.simulate.summaries import simulation_summary
from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.reports.checks import NotEvaluated
from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from grassmoth.artifacts.simulation import (
    PairedArmSimulation,
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from grassmoth.models.ssm.predictive.simulation import generate_simulation_batch
from grassmoth.models.structure import StructuralSelection, selected_state_ids
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import SimulateInput
from grassmoth_server.actions.contracts.requests import SimulateRequest
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied
from tests.support.action import applied_record, empty_simulation_summary
from tests.support.compute import COMPUTE_BUDGET
from tests.support.data import metadata_for_model
from tests.support.helpers import make_model, write_question
from tests.support.inference import compile_model_fixture
from tests.support.model import x_y_model

pytestmark = pytest.mark.inference(concern="simulation")


def test_full_categories_and_paired_paths_are_saved_once(tmp_path, monkeypatch):
    from grassmoth_server.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    dynamical_model_spec = make_model(
        ["x", "binary", "ordinal", "category"],
        [("x", "binary"), ("binary", "ordinal"), ("ordinal", "category")],
    )
    dtypes = {
        "x": "continuous",
        "binary": "binary",
        "ordinal": "ordinal",
        "category": "categorical",
    }
    constructs = tuple(
        c.revised(
            indicators=tuple(
                i.revised(
                    observation=i.observation.revised(
                        aggregation="last",
                        measurement_dtype=dtypes[c.name],
                        ordinal_levels=("low", "medium", "high") if c.name == "ordinal" else None,
                        categorical_levels=("a", "b") if c.name == "category" else None,
                    )
                )
                for i in c.indicators
            )
        )
        for c in dynamical_model_spec.constructs
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, constructs)
    )
    store, history = ArtifactStore("SUMMARY"), StudyRepository("SUMMARY")
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    history.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=None,
                effects=ActionEffects(produced=[write_question(store), definition]),
            ),
            seq=1,
            ts="2026-01-01T00:00:00Z",
            trace_ids=[],
        )
    )
    by_type = {
        "continuous": [1.0, 2.0, 3.0],
        "binary": [0.0, 1.0, 1.0],
        "ordinal": [0.0, 1.0, 2.0],
        "categorical": [0.0, 1.0, 1.0],
    }
    observations = np.stack(
        [
            np.tile(
                by_type[dynamical_model_spec.indicator(i).observation.measurement_dtype], (2, 1)
            ).T
            for i in (indicator.observation.id for indicator in dynamical_model_spec.indicators)
        ],
        axis=-1,
    )
    mask = np.ones_like(observations, dtype=bool)
    mask[:, 0, 0] = False
    observations[:, 0, 0] = np.nan
    states = np.arange(observations.size, dtype=float).reshape(observations.shape)
    states[0, 0, 0] = np.nan
    support_times = np.broadcast_to(
        np.array([[5.0], [7.0]]), (2, len(dynamical_model_spec.indicators))
    )
    layout = SimulationObservationLayout(
        variables=metadata_for_model(dynamical_model_spec).variables,
        support_start_times=NumericalArray.from_numpy(support_times),
        support_end_times=NumericalArray.from_numpy(support_times),
        mask=NumericalArray.from_numpy(mask),
    )
    report = SimulationReport(
        summary=empty_simulation_summary(),
        fit_reliability="not_fitted",
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            times=(5, 7),
            draws=3,
            seed=0,
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            state_ids=tuple(selected_state_ids(StructuralSelection(dynamical_model_spec, None))),
            parameter_draws={},
            arms=PairedArmSimulation(
                action=SimulationArm(
                    latent_paths=NumericalArray.from_numpy(states),
                    observations=NumericalArray.from_numpy(observations),
                ),
                reference=SimulationArm(
                    latent_paths=NumericalArray.from_numpy(states - 1),
                    observations=NumericalArray.from_numpy(observations),
                ),
                causal=NotEvaluated(
                    code="causal_effect",
                    subject="causal_effect",
                    reason="CAUSAL_EVALUATION_FAILED",
                    detail="No certified effect in this fixture.",
                ),
            ),
            observation_layout=layout,
            assignments=SimulationSpec(
                start=date(2026, 1, 6),
                horizon="2d",
                interventions=(
                    {
                        "target": selected_state_ids(
                            StructuralSelection(dynamical_model_spec, None)
                        )[0],
                        "value": 1,
                    },
                ),
            ).assignments(datetime(2026, 1, 1, tzinfo=UTC)),
        ),
    )
    report = report.revised(
        summary=simulation_summary(
            report.evidence, states, observations, mask, states - 1, observations
        )
    )
    published = history.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=report.evidence,
                effects=ActionEffects(reports={"simulation": store.write_result(report)}),
            ),
            request=SimulateRequest(
                input=SimulateInput(
                    dynamical_model_spec_ref=definition.revision,
                    simulation=SimulationSpec(
                        start=date(2026, 1, 6),
                        horizon="2d",
                        interventions=(
                            {"target": report.evidence.assignments[0].target, "value": 1},
                        ),
                    ),
                )
            ),
            seq=2,
            ts="2026-01-01T01:00:00Z",
            trace_ids=[],
        )
    )

    assert published.record.attempt.outcome.status == "applied"
    result_ref = published.record.attempt.outcome.result
    saved = store.read_result(result_ref, SimulateOutput)
    assert set(type(saved).model_fields) == {"data", "report"}
    evidence = saved.report.evidence
    assert evidence.draws == 3
    assert evidence.times == (5, 7)
    assert evidence.arms.kind == "paired"
    from grassmoth.artifacts.arrays import resolve_vector

    np.testing.assert_array_equal(evidence.arms.action.latent_paths.values, states)
    np.testing.assert_array_equal(evidence.arms.reference.latent_paths.values, states - 1)
    assert len(saved.data) == 3
    for index, variable in enumerate(layout.variables):
        expected = tuple(
            float(value) if observed else None
            for value, observed in zip(observations[1, :, index], mask[1, :, index], strict=True)
        )
        assert resolve_vector(saved.data[1][variable.id].values) == expected
    path_data = saved.report.summary.model_dump(mode="json")
    for variable in layout.variables:
        if variable.measurement_dtype == "binary":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"]["1"][
                1
            ] == pytest.approx(2 / 3)
        if variable.measurement_dtype == "ordinal":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"][
                "medium"
            ][1] == pytest.approx(1 / 3)
        if variable.measurement_dtype == "categorical":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"]["b"][
                1
            ] == pytest.approx(2 / 3)

    first = saved.report
    assert first.evidence == report.evidence

    def no_generation(*_args, **_kwargs):
        pytest.fail("A saved simulation read must not generate histories")

    monkeypatch.setattr(
        "grassmoth.models.ssm.predictive.simulation.generate_simulation_batch", no_generation
    )
    assert ArtifactStore("SUMMARY").read_result(result_ref, SimulateOutput).report == first
    monkeypatch.setattr("grassmoth_server.actions.output_builder.build_output", no_generation)
    assert ArtifactStore("SUMMARY").read_result(result_ref, SimulateOutput).report == first


def test_authored_law_advances_from_zero_before_a_later_requested_start():
    dynamical_model_spec = x_y_model()

    fixed = {
        p.id: 0.5 if p.name.startswith("rho") else 0.0 if p.name.startswith("beta") else 1e-8
        for p in dynamical_model_spec.parameters
    }

    def literal_coefficients(value):
        if isinstance(value, list):
            return [literal_coefficients(item) for item in value]
        if isinstance(value, dict):
            if value.get("kind") == "coefficient" and value.get("value") in fixed:
                return {**value, "value": fixed[value["value"]]}
            return {key: literal_coefficients(item) for key, item in value.items()}
        return value

    payload = literal_coefficients(dynamical_model_spec.model_dump(mode="json"))
    payload.update(parameters={}, distributions={})
    dynamical_model_spec = DynamicalModelSpec.model_validate(payload).materialized()
    constructs = tuple(
        c.revised(
            coefficients=tuple(
                coefficient.revised(value=10.0 if coefficient.role == "initial_mean" else 1e-8)
                if coefficient.role in {"initial_mean", "initial_scale"}
                else coefficient
                for coefficient in c.coefficients
            )
        )
        for c in dynamical_model_spec.constructs
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, constructs)
    )
    batch = generate_simulation_batch(
        compile_model_fixture(dynamical_model_spec),
        start=2.0,
        end=3.0,
        draws=2,
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        compute_budget=COMPUTE_BUDGET,
    )
    from grassmoth.models.ssm.predictive.simulation import SimulationBatch

    assert isinstance(batch, SimulationBatch)
    np.testing.assert_allclose(
        batch.prediction.trajectory.latents[:, 0], 10 * np.exp(-1), rtol=0.002
    )
    with pytest.raises(ValueError, match="before the initial law"):
        generate_simulation_batch(
            compile_model_fixture(dynamical_model_spec),
            start=-1.0,
            end=1.0,
            draws=2,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            compute_budget=COMPUTE_BUDGET,
        )
