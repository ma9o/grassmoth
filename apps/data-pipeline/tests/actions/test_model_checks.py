"""Model edits evaluate and reuse data-independent checks without numerical execution."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.bodies import EditModelInput
from nof1_causal_lab.actions.contracts.requests import EditModelRequest
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.model_checks import evaluate_model_checks
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import apply_effects
from tests.support.action import question_root
from tests.support.git import git_oid
from tests.support.model import construct_named, stress_sleep_causal_model

pytestmark = pytest.mark.contract


def test_edit_checks_depend_only_on_model_and_question_inputs(tmp_path, monkeypatch):
    from nof1_causal_lab.actions import model_checks
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_lab.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    dynamical_model_spec = stress_sleep_causal_model()
    question_root(
        "checks",
        QuestionSpec(
            text="How does sleep change?", outcome=construct_named(dynamical_model_spec, "Sleep").id
        ),
    )
    repository = StudyRepository("checks")
    root = repository.state(repository.head())

    def unexpected(*_args, **_kwargs):
        pytest.fail("Model edits must not access observations or generate predictions")

    monkeypatch.setattr(model_checks, "read_data_history", unexpected)
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
        reports = evaluate_model_checks("checks", state, staged, action="edit_model")
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
