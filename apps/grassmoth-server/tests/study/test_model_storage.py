"""Saved model revisions preserve scientific identity and their owned numerical laws."""

from datetime import UTC, datetime

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.identity import ConstructId, IndicatorId
from grassmoth.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    replace_constructs,
)
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import ObservationLawSpec
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.probability.serialization import (
    empirical_atoms,
    empirical_distribution,
)
from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
from grassmoth.models.ssm.compile.bindings import joint_law_layout, parameter_bindings
from grassmoth.models.structure import StructuralSelection, selected_state_ids
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.study.artifact_files import json_filename
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_model, read_question
from grassmoth_server.study.records import Applied, AttemptRecord, EditAttempt, Raised
from grassmoth_server_test_support.action import applied_record, question_root
from grassmoth_server_test_support.git import artifact_revision, commit_id
from grassmoth_server_test_support.helpers import write_question
from grassmoth_server_test_support.model import load_study_model_fixture
from grassmoth_test_support.fixtures import model_fixture_json
from grassmoth_test_support.helpers import graph_constructs
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import x_y_model


def _fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound_complete_test_model() -> (
    DynamicalModelSpec
):
    return load_study_model_fixture(
        "snapshots/fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound_complete_test_model.json"
    )


pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "MODELS"


def _model():
    return DynamicalModelSpec.model_validate(
        {
            "constructs": {
                f"construct:{key}": {
                    "name": key.upper(),
                    "description": key,
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": {
                        f"indicator:{key}": {
                            "observation": {
                                "name": f"{key.upper()}_obs",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "construct_polarity": "positive",
                        }
                    },
                }
                for key in ("x", "y")
            },
            "measurement_clock": "1d",
            "edges": {
                "edge:xy": {
                    "cause": "construct:x",
                    "effect": "construct:y",
                    "description": "effect",
                }
            },
        }
    ).materialized()


def _drop_x(model):
    return model.with_entities(
        edges=(
            CausalEdgeSpec(
                id="edge:yz",
                cause=model.get_construct("construct:y"),
                effect=ConstructSpec(
                    id="construct:z",
                    name="Z",
                    description="Downstream response",
                    role="endogenous",
                    temporal_status="time_varying",
                ),
                description="Y affects Z",
            ),
        )
    )


def test_predictive_findings_use_entity_ids_and_keep_served_reasons(monkeypatch):
    from grassmoth.artifacts.reports.checks import NumericCriterionEvidence
    from grassmoth.models.ssm.diagnostics.reachability import CheckResult
    from grassmoth.models.ssm.diagnostics.simulation import DesignInfo
    from grassmoth.models.ssm.predictive import simulation
    from grassmoth.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory

    dynamical_model_spec = x_y_model()
    edge = dynamical_model_spec.edges[0]

    def measured(_model, _prediction, _design, target, **_kwargs):
        if target.construct.id != edge.effect.id:
            return []
        return [
            CheckResult.measured(
                "measured",
                target,
                "1",
                "0",
                "Served reason.",
                outcome="failed",
                measurements=(NumericCriterionEvidence(criterion="measured", value=1, upper=0),),
            )
            for target in (
                edge.effect.name,
                edge.effect.indicators[0].observation.id,
                f"{edge.cause.name}->{edge.effect.name}",
            )
        ]

    monkeypatch.setattr(simulation, "measure_construct_simulation", measured)
    paths = jnp.zeros((1, 2, 2))
    prediction = PredictiveDraws(
        {}, PredictiveTrajectory(paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths)
    )
    batch = simulation.SimulationBatch.from_draws(
        prediction,
        DesignInfo(jnp.array([0.0, 1.0]), (), {}, {}),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    findings, _ = simulation.measure_simulation_batch(
        compile_model_fixture(dynamical_model_spec),
        batch,
        groups=("measurement",),
    )
    targets = [finding.subject.target for finding in findings]
    assert all(not isinstance(target, str) for target in targets)
    assert [target.id for target in targets if not isinstance(target, str)] == [
        edge.effect.id,
        edge.effect.indicators[0].observation.id,
        edge.id,
    ]
    for finding in findings:
        assert finding.kind == "evaluated"
        assert finding.evidence[0].note == "Served reason."


def _commit(workspace, artifact_id, payload, *, pins=None, retracted=(), reports=None):
    journal = StudyRepository(workspace)
    store = ArtifactStore(workspace)
    info = store.write_artifact(
        artifact_id,
        derived_from=pins or {},
        produced_by=None,
        json_files={json_filename(artifact_id, artifact_id): payload},
    )
    # Every lineage carries its question; the first commit adds it alongside.
    rooted = journal.state().has("question")
    journal.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=None,
                effects=ActionEffects(
                    produced=[info] if rooted else [write_question(store), info],
                    retracted=list(retracted),
                    reports=reports or {},
                ),
            ),
            seq=journal.latest_seq() + 1,
            ts="2026-09-12T12:00:00Z",
            trace_ids=[],
        ),
    )
    return info


def _measured(workspace):
    _commit(workspace, "model", _model().model_dump(mode="json"))


def _published_model(workspace, at=None):
    repository = StudyRepository(workspace)
    state = repository.state(at)
    return read_model(ArtifactStore(workspace), state.current["model"].revision)


def _identity_owners(model):
    return (
        *model.constructs,
        *model.edges,
        *(item.observation for item in model.indicators),
        *model.parameters,
    )


def test_saved_model_keeps_joint_arrays_lazy_and_self_contained(workspace, monkeypatch):
    from grassmoth.actions.contracts.bodies import EditModelOutput
    from grassmoth_server.study.records import AttemptRecord, EditAttempt

    dynamical_model_spec = (
        _fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound_complete_test_model()
    )
    bindings, _ = parameter_bindings(compile_model_fixture(dynamical_model_spec))
    layout = joint_law_layout(
        bindings,
        parameters=[p.id for p in dynamical_model_spec.parameters],
        constructs=selected_state_ids(StructuralSelection(dynamical_model_spec, None)),
        time_points=(0, 1),
    )
    store = ArtifactStore(workspace)
    atoms = np.ones((2, layout.width))
    law = empirical_distribution(
        atoms,
    )
    conditioned = dynamical_model_spec.with_entities(
        parameters=tuple(
            p.revised(distribution=layout.distribution_id, transform={"kind": "identity"})
            for p in dynamical_model_spec.parameters
        ),
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                c.revised(distribution=layout.distribution_id)
                for c in dynamical_model_spec.constructs
            ),
        ),
        distributions={layout.distribution_id: law},
        law_layouts={layout.distribution_id: layout},
    )
    question_root(workspace)
    staged = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    result = store.write_result(
        EditModelOutput(
            dynamical_model_spec=conditioned,
            checks={"specification": (), "question": {"findings": ()}},
            identification={"outcome": None, "treatments": {}},
            pruning={},
        )
    )
    attempt = EditAttempt(
        action="edit_model",
        request=None,
        outcome=Applied(
            result=result,
            effects=ActionEffects(produced=(store.result_artifact(staged, result),)),
        ),
    )
    StudyRepository(workspace).append(
        AttemptRecord(seq=2, ts="2026-09-12T12:00:00Z", attempt=attempt)
    )
    read_array = ArtifactStore.read_array
    reads = []

    def tracked_read(self, identity):
        reads.append(identity)
        return read_array(self, identity)

    monkeypatch.setattr(ArtifactStore, "read_array", tracked_read)
    restored_model = _published_model(workspace)
    assert restored_model.model_dump(mode="json") == conditioned.model_dump(mode="json")
    assert reads == []
    # The retained model owns its buffers and needs no workspace lookup to read draws.
    restored = restored_model.distributions[layout.distribution_id]
    np.testing.assert_array_equal(empirical_atoms(restored), atoms)
    assert reads == []


def test_partial_model_can_be_read_before_any_compilation(workspace):
    repository = StudyRepository(workspace)
    assert not repository.state().current
    question_root(
        workspace, QuestionSpec(text="Does X change Y?", outcome=ConstructId("construct:y"))
    )
    info = _commit(workspace, "model", {"measurement_clock": "1d"})
    state = repository.state()
    assert (
        read_question(ArtifactStore(workspace), state.current["question"].revision).text
        == "Does X change Y?"
    )
    model = _published_model(workspace)
    assert model == DynamicalModelSpec(measurement_clock="1d").materialized()
    assert state.current["model"].revision == info.revision
    assert not _identity_owners(model)


def test_ownership_is_explicit_from_first_structure(workspace):
    _measured(workspace)
    model = _published_model(workspace)
    assert {entity.id for entity in _identity_owners(model)} == {
        "construct:x",
        "construct:y",
        "edge:xy",
        "indicator:x",
        "indicator:y",
    }
    dynamical_model_spec = model
    assert (
        dynamical_model_spec.indicator(IndicatorId("indicator:x"))
        is dynamical_model_spec.get_construct(ConstructId("construct:x")).indicators[0]
    )
    assert DynamicalModelSpec.model_validate_json(model.model_dump_json()).materialized() == model


def test_rename_preserves_identity_and_historical_content(workspace):
    _measured(workspace)
    before_commit = StudyRepository(workspace).head()
    before = _published_model(workspace, before_commit)
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": artifact_revision(workspace, "model", 1)})
    after = _published_model(workspace)
    assert [entity.id for entity in _identity_owners(before)] == [
        entity.id for entity in _identity_owners(after)
    ]
    assert _published_model(workspace, before_commit) == before
    assert after.get_construct(ConstructId("construct:x")).name == "Treatment"
    assert after.indicator_owner(IndicatorId("indicator:x")).id == "construct:x"
    assert "construct_id" not in after.indicator(IndicatorId("indicator:x")).model_dump()


def test_planning_preserves_ids_across_name_and_role_edits(workspace):
    original = _model()
    payload = _model().model_dump(mode="json")
    given = graph_constructs(payload)[0]
    given.update(name="Renamed", role="exogenous", coefficients=[], dynamics={}, distribution=None)
    for indicator in given["indicators"].values():
        indicator["likelihood"] = {
            "law": {
                "distribution": "Delta",
                "v": {"kind": "state", "construct_id": "construct:x"},
            },
            "standardized": False,
            "reasoning": "The renamed input is given exactly.",
            "sources": [],
        }
    revised = DynamicalModelSpec.model_validate(payload).materialized()
    assert set(selected_state_ids(StructuralSelection(original, None))) == set(
        selected_state_ids(StructuralSelection(revised, None))
    )
    assert original.edges[0].id == revised.edges[0].id == "edge:xy"
    _commit(workspace, "model", revised.model_dump(mode="json"))
    assert _published_model(workspace) == revised
    dynamic_ids = selected_state_ids(StructuralSelection(revised, None))
    from grassmoth.models.causal.identification import unroll_temporal_dag

    dag = unroll_temporal_dag(revised.constructs, revised.edges, {"Renamed", "Y"})
    for construct_id in dynamic_ids:
        name = revised.get_construct(construct_id).name
        assert dag.has_edge(f"{name}_{{t-1}}", f"{name}_t")
    assert dag.has_edge("Renamed_{t-1}", "Y_t")


def test_removed_construct_removes_its_owned_indicators(workspace):
    _measured(workspace)
    _commit(
        workspace,
        "model",
        _drop_x(_model()).model_dump(mode="json"),
    )
    assert {entity.id for entity in _identity_owners(_published_model(workspace))} == {
        "construct:y",
        "indicator:y",
        "construct:z",
        "edge:yz",
    }
    assert len(_identity_owners(_published_model(workspace, commit_id(workspace, 1)))) == 5


def test_uncommitted_versions_and_failed_attempts_never_change_published_model(workspace):
    _measured(workspace)
    before_commit = StudyRepository(workspace).head()
    before = _published_model(workspace, before_commit)
    ArtifactStore(workspace).write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": {"measurement_clock": "2d"}},
    )
    StudyRepository(workspace).append(
        AttemptRecord(
            seq=3,
            ts="2026-09-12T13:00:00Z",
            trace_ids=[],
            attempt=EditAttempt(
                action="edit_model",
                request=None,
                outcome=Raised(error_type="SavedError", error_message="failed"),
            ),
        )
    )
    assert _published_model(workspace) == before
    with pytest.raises(StudyLookupError):
        StudyRepository(workspace).resolve(at=commit_id(workspace, 3))


@pytest.mark.parametrize("violation", ["owner", "identity"])
def test_model_rejects_inconsistent_facts(violation):
    payload = _model().model_dump(mode="json")
    if violation == "owner":
        next(iter(payload["edges"].values()))["effect"] = "construct:missing"
    else:
        next(iter(graph_constructs(payload)[0]["indicators"].values()))["observation"]["id"] = (
            "indicator:y"
        )
    with pytest.raises(ValidationError):
        DynamicalModelSpec.model_validate(payload).materialized()


def test_owned_likelihood_survives_reused_names(workspace, monkeypatch):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    next(iter(graph_constructs(payload)[1]["indicators"].values()))["likelihood"] = {
        "law": TypeAdapter(ObservationLawSpec)
        .validate_json(model_fixture_json("common/y_gaussian_observation_law.json"))
        .model_dump(mode="json"),
        "reasoning": "Test",
    }
    _commit(workspace, "model", payload)
    before_commit = StudyRepository(workspace).head()
    before = _published_model(workspace, before_commit)
    next(iter(graph_constructs(payload)[0]["indicators"].values()))["observation"]["name"] = "Y_obs"
    next(iter(graph_constructs(payload)[1]["indicators"].values()))["observation"]["name"] = "X_obs"
    _commit(workspace, "model", payload)

    after = _published_model(workspace)
    assert after.indicator(IndicatorId("indicator:x")).likelihood is None
    assert (
        after.indicator(IndicatorId("indicator:y")).likelihood
        == before.indicator(IndicatorId("indicator:y")).likelihood
    )
    assert after.indicator(IndicatorId("indicator:y")).observation.name == "X_obs"
    assert _published_model(workspace, before_commit) == before


@pytest.mark.parametrize("owner", ["constructs", "edges"])
def test_owned_mechanisms_survive_rename_and_disappear_with_owner(workspace, owner):
    _measured(workspace)
    payload = _model().model_dump(mode="json")
    field = "dynamics" if owner == "constructs" else "mechanisms"
    from grassmoth.artifacts.model.expressions import hill, restoring_force

    entity = next(iter(payload[owner].values()))
    mechanism = DriftMechanismSpec(
        id="mechanism:owned-term",
        expression=restoring_force(next(iter(payload[owner])), center=0, stiffness=1, quartic=0)
        if owner == "constructs"
        else hill(state(entity["cause"]), emax=2, ec50=1, n=2),
    ).model_dump(mode="json")
    mechanism_id = mechanism.pop("id")
    entity[field] = {mechanism_id: mechanism}
    _commit(workspace, "model", payload)
    before_commit = StudyRepository(workspace).head()
    before = _published_model(workspace, before_commit)
    graph_constructs(payload)[0]["name"] = "Renamed X"
    _commit(workspace, "model", payload)
    after = _published_model(workspace)
    encoded = after.model_dump(mode="json")
    assert next(iter(encoded[owner].values()))[field] == {mechanism_id: mechanism}
    payload = _drop_x(DynamicalModelSpec.model_validate(payload).materialized()).model_dump(
        mode="json"
    )
    _commit(workspace, "model", payload)
    assert list(_published_model(workspace).iter_mechanisms()) == []
    assert _published_model(workspace, before_commit) == before


def test_rename_changes_only_construct_label():
    payload = _model().model_dump(mode="json")
    original = DynamicalModelSpec.model_validate(payload).materialized()
    graph_constructs(payload)[0]["name"] = "Renamed"
    renamed = DynamicalModelSpec.model_validate(payload).materialized()
    assert original.indicators == renamed.indicators
    assert original.edges[0].id == renamed.edges[0].id
    assert original.edges[0].effect == renamed.edges[0].effect
    assert original.edges[0].cause.revised(name="Renamed") == renamed.edges[0].cause
    assert selected_state_ids(StructuralSelection(original, None)) == selected_state_ids(
        StructuralSelection(renamed, None)
    )


def _identification():
    return {
        "outcome": "construct:y",
        "treatments": {
            "construct:x": {
                "status": "identified",
                "estimand": "P(Y | do(X))",
            },
        },
    }


def test_identification_keeps_entity_ids_after_rename(workspace):
    from grassmoth.actions.contracts.bodies import EditModelOutput
    from grassmoth.artifacts.reports.model_checks import ModelCheckReport

    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    reports = {
        "checks": store.write_result(
            ModelCheckReport(specification=(), question=QuestionCheckReport(findings=()))
        ),
        "identification": store.write_result(
            IdentificationReport.model_validate(_identification())
        ),
    }
    original = _commit(workspace, "model", _model().model_dump(mode="json"), reports=reports)
    before = repository.record(repository.head()).record.attempt.outcome
    assert before.status == "applied"
    before_output = store.read_result(before.result, EditModelOutput)
    payload = _model().model_dump(mode="json")
    graph_constructs(payload)[0]["name"] = "Treatment"
    _commit(workspace, "model", payload, pins={"model": original.revision}, reports=reports)
    after = repository.record(repository.head()).record.attempt.outcome
    assert after.status == "applied"
    after_output = store.read_result(after.result, EditModelOutput)
    assert after_output.identification == before_output.identification
    assert "construct:x" in after_output.identification.estimable_treatments
    assert store.read_result(before.result, EditModelOutput) == before_output


@pytest.mark.parametrize(
    "field", ["outcome", "treatment", "marginalized_confounders", "instruments", "confounders"]
)
def test_identification_rejects_unresolved_construct_ids(field):
    payload = _identification()
    if field == "outcome":
        payload["outcome"] = "construct:missing"
    elif field == "treatment":
        payload["treatments"]["construct:missing"] = {
            "status": "identified",
            "estimand": "test",
        }
    elif field == "confounders":
        payload["treatments"]["construct:y"] = {
            "status": "not_identified",
            "confounders": ["construct:missing"],
        }
    else:
        payload["treatments"]["construct:x"][field] = ["construct:missing"]
    with pytest.raises(ValueError, match="unknown construct IDs"):
        IdentificationReport.model_validate(payload).validate_model(_model())


def test_identification_round_trip_preserves_tagged_evidence():
    payload = _identification()
    payload["treatments"]["construct:y"] = {
        "status": "not_identified",
        "confounders": ["construct:x"],
        "notes": "Blocking confounding",
    }
    report = IdentificationReport.model_validate(payload)
    restored = IdentificationReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.estimable_treatments == ("construct:x",)
    identified = restored.treatments[ConstructId("construct:x")]
    assert identified.status == "identified"
    assert identified.estimand == "P(Y | do(X))"
    assert restored.non_identifiable[ConstructId("construct:y")].confounders == ("construct:x",)
    assert restored.non_identifiable[ConstructId("construct:y")].notes == "Blocking confounding"

    payload["treatments"]["construct:y"]["estimand"] = "Contradictory positive evidence"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IdentificationReport.model_validate(payload)
