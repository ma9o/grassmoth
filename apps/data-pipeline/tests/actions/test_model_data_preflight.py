import logging
import time
from datetime import UTC, datetime
from unittest.mock import Mock

import polars as pl
import pytest

from nof1_causal_lab.actions.checks import check_specification
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.models.ssm.inference.fit import fit
from nof1_causal_lab.models.structure import StructuralSelection
from nof1_causal_service.actions.contracts.effects import ActionEffects
from nof1_causal_service.actions.contracts.inputs import EditModelInput
from nof1_causal_service.actions.contracts.requests import EditModelRequest
from nof1_causal_service.actions.messages import edit_messages
from nof1_causal_service.compute.compilation_cache import compile_staged, prepare_program
from nof1_causal_service.study.persistence.store import ArtifactStore
from nof1_causal_service.study.records import Applied
from nof1_causal_service.study.state import StudyState
from tests.integration.runner_fixtures import panel_frame, panel_metadata
from tests.support.action import edit_and_check, prepared_profile
from tests.support.compute import COMPUTE_BUDGET
from tests.support.helpers import write_question
from tests.support.inference import compile_fit_fixture
from tests.support.model import construct_named, stress_sleep_causal_model

"""Model-panel limitations are checked when fitting, independently of model edits."""


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


def test_noisy_interval_summary_passes_fit_admission_and_dispatches(monkeypatch):
    from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs import (
        fit as marginal_particle_gibbs,
    )

    dispatch = Mock(return_value=object())
    monkeypatch.setattr(marginal_particle_gibbs, "fit_marginal_particle_gibbs", dispatch)
    dynamical_model_spec, panel = (
        stress_sleep_causal_model(),
        panel_frame(n_days=4),
    )
    panel = ObservationDataset.from_frame(
        panel, panel_metadata().variables, time_origin=panel_metadata().time_origin
    )
    inputs = compile_fit_fixture(dynamical_model_spec)
    from nof1_causal_lab.models.ssm.preparation.panel import PreparedFit, prepare_fit

    prepared = prepare_fit(inputs, panel, time_origin=panel_metadata().time_origin)
    assert isinstance(prepared, PreparedFit)
    assert prepared.inputs is inputs
    assert prepared.panel.compiled_dynamical_model is inputs.compiled_dynamical_model
    sampler = SamplerSpec()
    result = fit(
        prepared.inputs.prior_runtime_bundle,
        prepared.panel,
        sampler=sampler,
        reparam=None,
        clock=time.monotonic,
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_staged,
        prepare_program=prepare_program,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert result is dispatch.return_value
    dispatch.assert_called_once_with(
        prepared.inputs.prior_runtime_bundle,
        prepared.panel,
        compute_budget=COMPUTE_BUDGET,
        sampler=sampler,
        reparam=None,
        clock=time.monotonic,
        logger=logging.getLogger(__name__),
        wall_clock=dispatch.call_args.kwargs["wall_clock"],
        continuation=None,
        compiler=compile_staged,
        prepare_program=prepare_program,
    )


def test_edit_with_missing_panel_variable_has_no_data_findings(tmp_path, monkeypatch):
    from nof1_causal_service.storage import paths as data_module

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
    prepared_profile(
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
    from tests.support.action import model_findings

    checks, identification = model_findings(
        "TEST", state.with_artifacts(edited.effects.produced), action="edit_model"
    )
    assert checks.question is not None
    assert all(finding.code not in {"window", "range"} for finding in checks.question.findings)
    from nof1_causal_lab.actions.contracts.bodies import EditModelOutput

    messages = edit_messages(
        checks,
        identification,
        store.read_result(edited.result, EditModelOutput).pruning,
        datetime.now(UTC),
    )
    assert "MODEL_DATA_INCOMPATIBLE" not in {message.code for message in messages}
