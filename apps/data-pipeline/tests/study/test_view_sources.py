"""Read findings follow their scientific revision and observational inputs."""

import pytest

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model
from nof1_causal_lab.study.records import Applied, inference_record
from tests.support.action import applied_record
from tests.support.git import artifact_revision
from tests.support.helpers import write_question
from tests.support.model import x_y_model


@pytest.mark.contract
def test_authored_model_has_no_inference_report(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("ABSENT"), StudyRepository("ABSENT")
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": x_y_model().model_dump(mode="json", round_trip=True)},
    )
    journal.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=(write_question(store), info))),
            seq=1,
        )
    )
    assert inference_record(journal.records(journal.head()), info.revision) is None


@pytest.mark.inference(concern="sampling")
def test_joint_reports_and_raw_draws_use_production_labels_without_compiling(monkeypatch, tmp_path):
    from nof1_causal_lab.actions.contracts.bodies import NewFitTarget, FitInput, FitOutput
    from nof1_causal_lab.actions.contracts.requests import FitRequest
    from nof1_causal_lab.actions.fit.action import read_inference_report
    from nof1_causal_lab.artifacts.observations.reference import DataRef
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from nof1_causal_lab.storage import paths as data
    from tests.support.inference import inference_metadata
    from tests.support.model import load_model_fixture

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("LABELS"), StudyRepository("LABELS")
    dynamical_model_spec = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    identity, layout = next(iter(dynamical_model_spec.law_layouts.items()))
    labels = {element: f"Production label {index}" for index, element in enumerate(layout.labels)}
    dynamical_model_spec = dynamical_model_spec.with_entities(
        law_layouts={identity: layout.revised(labels=labels)}
    )
    prior = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={
            "model.json": load_model_fixture("causal_proofs/treatment_outcome.json").model_dump(
                mode="json", round_trip=True
            )
        },
    )
    from tests.integration.runner_fixtures import seed_panel

    panel = seed_panel(store, model_revision=prior.revision)
    info = store.write_artifact(
        "model",
        derived_from={"model": prior.revision, "panel": panel.revision},
        produced_by="fit",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json", round_trip=True)},
    )
    result = InferenceEvidence()
    published = journal.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=result,
                effects=ActionEffects(
                    produced=(write_question(store), panel, info),
                    reports={
                        "inference": store.write_report(
                            read_inference_report(
                                store,
                                info.revision,
                                result,
                                inference_metadata(dynamical_model_spec),
                            )
                        )
                    },
                ),
            ),
            request=FitRequest(
                input=FitInput(target=NewFitTarget(
                    dynamical_model_spec_ref=prior.revision,
                    data_ref=DataRef(revision=panel.revision, replicate_index=0)),)
            ),
            seq=1,
        ),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.compile.inputs.compile_model",
        lambda *_args: pytest.fail("Reports and raw atoms do not need the compiler"),
    )
    assert published.record.attempt.outcome.status == "applied"
    output = store.read_result(published.record.attempt.outcome.result, FitOutput)
    report = output.inference
    marginals = report.core.posterior_marginals
    assert marginals is not None
    assert {row.parameter for row in marginals} == set(labels.values())
    assert report.core.inference_metadata.num_chains == 1
    assert report.core.inference_diagnostics is not None
    assert set(output.model_dump()) == {"dynamical_model_spec", "checks", "inference"}
    from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms

    np_atoms = empirical_atoms(output.dynamical_model_spec.distributions[identity])
    assert len(np_atoms) == report.core.inference_metadata.num_samples_total
    assert {row.subject.element_id for row in marginals} == set(
        dynamical_model_spec.law_layouts[identity].labels
    )
    assert all(row.empirical[-1].probability == 1 for row in marginals)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_model_view_reads_canonical_science_without_a_compiled_plan(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("DEFINITION")
    dynamical_model_spec = x_y_model()
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    repository = StudyRepository("DEFINITION")
    repository.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=None,
                effects=ActionEffects(produced=[write_question(store), info]),
            ),
            seq=1,
            ts="2026-09-14T12:00:00Z",
            trace_ids=[],
        )
    )
    assert read_model(store, info.revision) == dynamical_model_spec
    assert all(
        "prior_density_points" not in parameter.model_dump()
        for parameter in dynamical_model_spec.parameters
    )
    changed = dynamical_model_spec.revised(measurement_clock="2d")
    revision = store.write_artifact(
        "model",
        derived_from={"model": artifact_revision("DEFINITION", "model", 1)},
        produced_by=None,
        json_files={"model.json": changed.model_dump(mode="json")},
    )
    repository.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=[revision])),
            seq=2,
            ts="2026-09-14T13:00:00Z",
            trace_ids=[],
        )
    )
    assert read_model(store, revision.revision) == changed
    assert read_model(store, info.revision) == dynamical_model_spec
