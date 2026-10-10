"""Model-panel limitations are checked when fitting, independently of model edits."""

import time
from datetime import UTC, datetime

import polars as pl
import pytest

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.io import EditModelInput
from nof1_causal_lab.actions.messages import edit_messages
from nof1_causal_lab.artifacts.data_preparation import DataPreparationResult
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observation_data import ObservationDataset
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.sampler_config import SamplerSpec
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import edit_and_check
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.helpers import write_question
from tests.inference_fixtures import compile_fit_fixture
from tests.integration.runner_fixtures import panel_frame, panel_metadata
from tests.model_fixtures import construct_named, stress_sleep_causal_model

pytestmark = pytest.mark.contract


def test_specification_reports_each_distinct_fit_law_reason_once(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import inputs as compilation
    from nof1_causal_lab.models.ssm.compile import prior_compilation

    def unsupported(_compiled, _model):
        raise prior_compilation.PriorCompilationError(
            ["Unsupported joint law.", "Unsupported joint law.", "Invalid scale."]
        )

    monkeypatch.setattr(compilation, "compile_priors", unsupported)
    selection = StructuralSelection(
        stress_sleep_causal_model(),
        None,
    )
    compiled = compilation.compile_model(selection)
    report = check_specification(compiled, compilation.compile_fit_inputs(compiled, selection))
    finding = next(finding for finding in report if finding.code == "fit_laws")
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert finding.evidence.count("Unsupported joint law.") == 1
    assert finding.evidence.count("Invalid scale.") == 1


def test_interval_summary_fails_shared_preflight_before_particle_dispatch():
    dynamical_model_spec, panel = (
        stress_sleep_causal_model(),
        panel_frame(n_days=4),
    )
    panel = ObservationDataset.from_frame(
        panel, panel_metadata().variables, time_origin=panel_metadata().time_origin
    )
    inputs = compile_fit_fixture(dynamical_model_spec)
    report = check_model_data(inputs, panel, time_origin=panel_metadata().time_origin)
    finding = report[0]
    assert finding.code == "fit_preflight"
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert "interval summaries" in finding.evidence
    assert "stress_score" in finding.evidence
    from nof1_causal_lab.models.ssm.runtime import PreparedFit, prepare_fit

    prepared = prepare_fit(inputs, panel, time_origin=panel_metadata().time_origin)
    assert isinstance(prepared, PreparedFit)
    assert prepared.inputs is inputs
    assert prepared.panel.compiled_dynamical_model is inputs.compiled_dynamical_model
    failure = fit(
        prepared.inputs.prior_runtime_bundle,
        prepared.panel,
        sampler=SamplerSpec(),
        compute_budget=COMPUTE_BUDGET,
        clock=time.monotonic,
    )
    assert isinstance(failure, ObservationPreflightFailure)
    assert "interval summaries" in failure.message


def test_edit_with_missing_panel_variable_has_no_data_findings(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    full = panel_metadata()
    metadata = full.revised(
        variables=full.variables[:1],
        preparation=full.preparation.revised(variables=full.preparation.variables[:1]),
    )
    panel = panel_frame(n_days=4).filter(pl.col("indicator_id") == metadata.variables[0].id)
    record = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": panel},
    )
    evaluate_data_checks(
        "TEST",
        StudyState(),
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[record])),
    )
    state = StudyState(
        data=DataRef[GitOid, int](revision=record.revision, replicate_index=0)
    ).with_artifacts(
        [
            write_question(
                store,
                QuestionSpec(
                    text="How does sleep change?",
                    outcome=construct_named(stress_sleep_causal_model(), "Sleep").id,
                ),
            ),
            record,
        ]
    )
    edited = edit_and_check(
        "TEST",
        EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=state.current["question"].revision,
                dynamical_model_spec=stress_sleep_causal_model(),
            )
        ),
        state,
    )
    assert "model" in {item.artifact_id for item in edited.effects.produced}
    from nof1_causal_lab.actions.model_checks import read_model_checks

    checks, identification = read_model_checks(
        "TEST", state.with_artifacts(edited.effects.produced), action="edit_model"
    )
    assert checks.question is not None
    assert all(finding.code not in {"window", "range"} for finding in checks.question.findings)
    messages = edit_messages(checks, identification, edited.result, datetime.now(UTC))
    assert "MODEL_DATA_INCOMPATIBLE" not in {message.code for message in messages}
