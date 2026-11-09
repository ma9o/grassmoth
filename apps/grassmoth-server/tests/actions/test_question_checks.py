"""The question's quick checks report what the model and the record still lack."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from grassmoth.actions.edit_model import ModelEditFailure, edit_model
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from grassmoth.models.causal.question_checks import question_findings
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.actions.contracts.inputs import EditModelInput
from grassmoth_test_support.helpers import make_model, observation_dataset
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract

_ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def _model():
    """A given dose read exactly each day, and the mood it may change."""
    dynamical_model_spec = make_model(["Dose", "Mood"], [("Dose", "Mood")])
    dose = dynamical_model_spec.constructs[0]
    reading = revise_fixture(
        dose.indicators[0],
        reasoning={"text": "The dose is read exactly."},
        likelihood=LikelihoodSpec(law=DeltaLawSpec(v=state(dose.id))),
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (revise_fixture(dose, role="exogenous", indicators=(reading,)),),
        )
    )


def _question(model, *, start="2024-01-05", value=0.0, outcome=None):
    dose = model.constructs[0]
    return revise_fixture(
        model,
        question="Does lowering my dose change my mood?",
        outcome=outcome or model.outcome,
        core_query={
            "window": {"start": start, "horizon": "2w"},
            "interventions": [{"target": dose.id, "after": "1w", "value": value}],
        },
    ).materialized()


def _panel(model):
    dose = model.constructs[0]
    days = range(10)
    return pl.DataFrame(
        {
            "indicator_id": [dose.indicators[0].observation.id] * len(days),
            "value": [10.0 if day < 5 else 20.0 for day in days],
            "support_start": [(_ORIGIN + timedelta(days=day)).replace(tzinfo=None) for day in days],
            "support_end": [
                (_ORIGIN + timedelta(days=day + 1)).replace(tzinfo=None) for day in days
            ],
        }
    )


def _findings(model, panel):
    definitions = tuple(
        indicator.observation.resolved(model.time.step) for indicator in model.indicators
    )
    variable = definitions[0]
    history = (
        None
        if panel is None
        else observation_dataset(
            panel.with_columns(
                pl.col("support_end").alias("anchor_time"),
                pl.lit(variable.support_kind.value).alias("support_kind"),
                pl.lit(variable.summary_operator.value).alias("summary_operator"),
                pl.lit(variable.anchor_policy.value).alias("anchor_policy"),
                pl.lit(variable.observation_window.source).alias("observation_window"),
            ),
            definitions,
            time_origin=_ORIGIN,
        )
    )
    return {
        (finding.code, finding.kind, getattr(finding, "outcome", finding.kind))
        for finding in question_findings(StructuralSelection.for_model(model), history=history)
    }


def test_model_only_checks_undefined_targets_without_data_findings():
    model = make_model(["Stress"])
    model = revise_fixture(
        model,
        core_query={
            "window": {"start": "2024-01-01", "horizon": "2d"},
            "interventions": [{"target": "construct:missing", "value": 1}],
        },
    )
    findings = question_findings(StructuralSelection.for_model(model), history=None)
    assert {(item.code, item.kind) for item in findings} == {
        ("outcome", "evaluated"),
        ("target", "not_evaluated"),
        ("identification", "not_evaluated"),
    }
    assert {item.reason for item in findings if item.kind == "not_evaluated"} == {
        "CONSTRUCT_UNDEFINED"
    }


def test_core_query_is_checked_against_the_model_and_the_record():
    model = _model()
    panel = _panel(model)
    assert _findings(_question(model, value=15.0), panel) == {
        ("window", "evaluated", "passed"),
        ("range", "evaluated", "passed"),
    }
    assert ("window", "not_evaluated", "not_evaluated") in _findings(
        _question(model), panel.head(0)
    )
    assert ("range", "evaluated", "failed") in _findings(_question(model, value=0), panel)
    with pytest.raises(ValueError, match="before the model"):
        _question(model, start="2023-12-25")
    dose = model.constructs[0]
    with pytest.raises(ValueError, match="endogenous construct"):
        _question(model, outcome=dose.id)


def test_an_edit_keeps_an_unresolved_query_target_as_a_readiness_finding():
    model = _question(_model())
    result = edit_model(None, model)
    assert not isinstance(result, ModelEditFailure)
    result = edit_model(
        model,
        DynamicalModelSpec(
            core_query={"interventions": [{"target": "construct:missing", "value": 1}]}
        ),
    )
    assert not isinstance(result, ModelEditFailure)
    assert any(
        item.code == "target" and item.kind == "not_evaluated"
        for item in result.checks.question.findings
    )


@pytest.mark.parametrize(
    ("extra_edges", "outside"),
    [
        ([("Dose", "Side")], ("Side",)),
        ([("Mood", "Side")], ("Side",)),
        ([("Dose", "Side"), ("Side", "Loop"), ("Loop", "Side")], ("Loop", "Side")),
    ],
)
def test_an_edit_allows_connected_branches_without_a_path_to_the_outcome(extra_edges, outside):
    dynamical_model_spec = make_model(["Dose", "Mood", *outside], [("Dose", "Mood"), *extra_edges])
    draft = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                revise_fixture(construct, indicators=())
                for construct in dynamical_model_spec.constructs
            ),
        ),
    )
    assert not isinstance(edit_model(None, draft), ModelEditFailure)


def test_an_edit_allows_upstream_causes_confounders_and_feedback_that_reaches_the_outcome():
    dynamical_model_spec = revise_fixture(
        make_model(
            ["U", "Dose", "Mediator", "Mood", "Moderator"],
            [
                ("U", "Dose"),
                ("U", "Mood"),
                ("Dose", "Mediator"),
                ("Mediator", "Mood"),
                ("Mood", "Mediator"),
                ("Moderator", "Mood"),
            ],
        ),
    )
    confounder = next(
        construct for construct in dynamical_model_spec.constructs if construct.name == "U"
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges, (revise_fixture(confounder, indicators=()),)
        )
    )
    assert not isinstance(edit_model(None, dynamical_model_spec), ModelEditFailure)


def test_an_off_path_model_edit_retains_the_connected_branch(tmp_path, monkeypatch):
    from grassmoth_server.actions.contracts.requests import EditModelRequest
    from grassmoth_server.actions.edit_model import edit_model
    from grassmoth_server.storage import paths as data
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.state import StudyState
    from grassmoth_server_test_support.helpers import write_initial_model

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("question-path")
    question = write_initial_model(store, _question(_model()))
    state = StudyState().with_artifacts([question])
    dynamical_model_spec = make_model(
        ["Dose", "Mood", "Side"], [("Dose", "Mood"), ("Dose", "Side")]
    )
    result = edit_model(
        store.workspace_id,
        EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=state.current["model"].revision,
                dynamical_model_spec=dynamical_model_spec,
            )
        ),
    )
    from grassmoth_server.study.records import Applied

    assert isinstance(result, Applied)
    from grassmoth.actions.contracts.io import EditModelOutput

    saved = store.read_result(result.result, EditModelOutput)
    assert saved.pruning.constructs == ()
    assert (
        saved.dynamical_model_spec.get_construct(dynamical_model_spec.constructs[2].id)
        == (dynamical_model_spec.constructs[2])
    )
    assert len(result.effects.produced) == 1


def test_unobserved_outcome_explanation_is_an_identification_finding():
    from grassmoth.artifacts.model.dynamical_model_spec import ModelEditResult
    from grassmoth.artifacts.reports.checks import QueryTargetSubject
    from grassmoth.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport
    from grassmoth_server.actions.messages import _findings as announce_findings
    from grassmoth_server.actions.messages import edit_messages

    model = _model()
    dose, mood = model.constructs
    model = model.with_entities(
        edges=replace_constructs(model.edges, (revise_fixture(mood, indicators=()),))
    )
    model = _question(model)
    findings = question_findings(StructuralSelection(model, mood.id), history=None)
    (finding,) = (item for item in findings if item.code == "identification")
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert isinstance(finding.subject, QueryTargetSubject)
    assert finding.subject.target.id == dose.id
    assert "the outcome is unobserved" in finding.evidence
    (message,) = announce_findings((finding,), _ORIGIN)
    assert message.severity == "warning"
    assert message.detail == finding.evidence
    selection = StructuralSelection(model, mood.id)
    (unqueried,) = edit_messages(
        ModelCheckReport(specification=(), question=QuestionCheckReport(findings=())),
        selection.identification,
        ModelEditResult(),
        _ORIGIN,
    )
    assert unqueried.code == "identification"
    assert "the outcome is unobserved" in unqueried.detail
