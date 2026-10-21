"""Model edits evaluate and reuse data-independent checks without numerical execution."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.bodies import EditModelOutput
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_service.actions.contracts.effects import ActionEffects
from nof1_causal_service.actions.contracts.inputs import EditModelInput
from nof1_causal_service.actions.contracts.requests import EditModelRequest
from nof1_causal_service.actions.edit_model import edit_model
from nof1_causal_service.study.persistence.history import StudyRepository
from nof1_causal_service.study.persistence.store import ArtifactStore
from nof1_causal_service.study.records import Applied
from nof1_causal_service.study.state import StudyState, apply_effects
from tests.support.action import question_root
from tests.support.git import git_oid
from tests.support.model import construct_named, stress_sleep_causal_model

pytestmark = pytest.mark.contract


def test_edit_checks_depend_only_on_model_and_question_inputs(tmp_path, monkeypatch):
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_service.storage import paths as data
    from nof1_causal_service.study.queries import data as model_data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    dynamical_model_spec = stress_sleep_causal_model()
    question_root(
        "checks",
        QuestionSpec(
            text="How does sleep change?", outcome=construct_named(dynamical_model_spec, "Sleep").id
        ),
    )
    repository = StudyRepository("checks")
    root = repository.state()

    def unexpected(*_args, **_kwargs):
        pytest.fail("Model edits must not access observations or generate predictions")

    monkeypatch.setattr(model_data, "read_data_history", unexpected)
    monkeypatch.setattr(simulation, "generate_simulation_batch", unexpected)

    def edit(state):
        staged = edit_model(
            "checks",
            EditModelRequest[GitOid](
                input=EditModelInput[GitOid](
                    parent_ref=state.current["model"].revision
                    if state.has("model")
                    else root.current["question"].revision,
                    dynamical_model_spec=dynamical_model_spec,
                )
            ),
        )
        assert isinstance(staged, Applied)
        output = ArtifactStore("checks").read_result(staged.result, EditModelOutput)
        reports = output.checks, output.identification
        return apply_effects(state, staged.effects.produced, staged.effects.retracted), reports

    # Even a selected, unreadable panel has no bearing on a model edit.
    state, (checks, identification) = edit(
        root.revised(data=DataRef[GitOid, int](revision=git_oid(99), replicate_index=0))
    )
    assert checks.specification
    assert identification is not None
    assert checks.question is not None
    assert all(f.code not in {"window", "range"} for f in checks.question.findings)
    _, (reused, _) = edit(
        state.revised(data=DataRef[GitOid, int](revision=git_oid(100), replicate_index=0))
    )
    assert reused.question == checks.question


def test_edit_model_contract_rejects_panel_selection():
    with pytest.raises(ValidationError, match="panel_ref"):
        EditModelInput[GitOid].model_validate(
            {
                "parent_ref": git_oid(1),
                "dynamical_model_spec": {},
                "panel_ref": git_oid(2),
            }
        )


def test_fit_checks_reuse_the_retained_input_report_without_rebuilding(tmp_path, monkeypatch):
    from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
    from nof1_causal_lab.artifacts.reports.inference import FitCheckReport
    from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
    from nof1_causal_service.storage import paths as data
    from nof1_causal_service.study.persistence.store import ArtifactStore

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    report = FitCheckReport(
        data=DataProfileReport(indicators={}, findings=()),
        preflight=(),
        question=QuestionCheckReport(findings=()),
    )
    from nof1_causal_lab.actions import checks as library_checks
    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_service.actions.contracts.inputs import FitInput, NewFitTarget
    from nof1_causal_service.actions.contracts.requests import FitRequest
    from nof1_causal_service.actions.temporal.study.activities import evaluate_model_checks_activity
    from nof1_causal_service.actions.temporal.study.messages import EvaluateChecksInput
    from tests.support.helpers import run_async
    from tests.support.inference import _report

    model = stress_sleep_causal_model()
    output = FitOutput(dynamical_model_spec=model, checks=report, inference=_report(model))
    reference = ArtifactStore("checks").write_result(output)

    def unexpected(*_args, **_kwargs):
        pytest.fail("A completed fit must reuse its selected-input findings")

    monkeypatch.setattr(library_checks, "fit_check_report", unexpected)
    result = run_async(
        evaluate_model_checks_activity(
            EvaluateChecksInput(
                workspace_id="checks",
                state=StudyState(),
                applied=Applied(result=reference, effects=ActionEffects()),
                request=FitRequest(
                    input=FitInput(
                        target=NewFitTarget(
                            dynamical_model_spec_ref=git_oid(1),
                            data_ref=DataRef(revision=git_oid(2), replicate_index=0),
                        )
                    )
                ),
            )
        )
    )
    assert result.reports == {}
