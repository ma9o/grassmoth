"""The question's quick checks report what the model and the record still lack."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts.bodies import EditModelInput
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.expressions import state
from nof1_causal_lab.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.models.causal.question_checks import question_edit_reason, question_findings
from nof1_causal_lab.models.structure import StructuralSelection
from tests.support.helpers import make_model

pytestmark = pytest.mark.contract

_ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def _model():
    """A given dose read exactly each day, and the mood it may change."""
    dynamical_model_spec = make_model(["Dose", "Mood"], [("Dose", "Mood")])
    dose = dynamical_model_spec.constructs[0]
    reading = dose.indicators[0].revised(
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(dose.id)), reasoning="The dose is read exactly."
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges, (dose.revised(role="exogenous", indicators=(reading,)),)
        )
    )


def _question(model, *, start="2024-01-05", value=0.0, outcome=None):
    dose, mood = model.constructs
    return QuestionSpec(
        text="Does lowering my dose change my mood?",
        outcome=outcome or mood.id,
        queries={
            "taper": {
                "start": start,
                "horizon": "2w",
                "interventions": [{"target": dose.id, "after": "1w", "value": value}],
            }
        },
    )


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


def _findings(question, model, panel):
    definitions = tuple(
        indicator.observation.resolved(model.measurement_clock) for indicator in model.indicators
    )
    variable = definitions[0]
    history = (
        None
        if panel is None
        else ObservationDataset.from_frame(
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
        for finding in question_findings(
            question, StructuralSelection.for_question(model, question), history=history
        )
    }


def test_model_only_checks_undefined_constructs_without_data_findings():
    dynamical_model_spec = make_model(["Stress"])
    question = _question(_model())
    findings = question_findings(
        question,
        StructuralSelection.for_question(dynamical_model_spec, question),
        history=None,
    )
    assert {(item.code, item.kind) for item in findings} == {
        ("outcome", "not_evaluated"),
        ("target", "not_evaluated"),
        ("identification", "not_evaluated"),
    }
    reasons = {item.code: item.reason for item in findings if item.kind == "not_evaluated"}
    assert reasons["outcome"] == "CONSTRUCT_UNDEFINED"


def test_queries_are_checked_against_the_model_and_the_record():
    dynamical_model_spec = _model()
    panel = _panel(dynamical_model_spec)
    assert _findings(_question(dynamical_model_spec, value=15.0), dynamical_model_spec, panel) == {
        ("window", "evaluated", "passed"),
        ("range", "evaluated", "passed"),
    }
    outside = _findings(
        _question(dynamical_model_spec, start="2023-12-25", value=0.0), dynamical_model_spec, panel
    )
    assert ("window", "evaluated", "failed") in outside
    assert ("range", "evaluated", "failed") in outside
    dose = dynamical_model_spec.constructs[0]
    with pytest.raises(ValueError, match="intervenes on the outcome"):
        _question(dynamical_model_spec, outcome=dose.id)
    exogenous_outcome = QuestionSpec(text="Does the dose change?", outcome=dose.id)
    assert _findings(exogenous_outcome, dynamical_model_spec, None) == {
        ("outcome", "evaluated", "failed")
    }


def test_an_edit_defines_the_question_and_models_its_outcome():
    dynamical_model_spec = _model()
    dose, _ = dynamical_model_spec.constructs
    assert question_edit_reason(dynamical_model_spec, _question(dynamical_model_spec)) is None
    missing = question_edit_reason(make_model(["Mood"]), _question(dynamical_model_spec))
    assert missing is not None
    assert dose.id in missing
    given = QuestionSpec(text="Does my dose change?", outcome=dose.id)
    assert (
        question_edit_reason(dynamical_model_spec, given)
        == "The question's outcome must reference an endogenous construct"
    )


@pytest.mark.parametrize(
    ("extra_edges", "outside"),
    [
        ([("Dose", "Side")], ("Side",)),
        ([("Mood", "Side")], ("Side",)),
        ([("Dose", "Side"), ("Side", "Loop"), ("Loop", "Side")], ("Loop", "Side")),
    ],
)
def test_an_edit_rejects_connected_branches_without_a_path_to_the_outcome(extra_edges, outside):
    dynamical_model_spec = make_model(["Dose", "Mood", *outside], [("Dose", "Mood"), *extra_edges])
    draft = dynamical_model_spec.revised(measurement_clock=None).with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(indicators=()) for construct in dynamical_model_spec.constructs
            ),
        ),
    )
    reason = question_edit_reason(draft, _question(_model()))
    assert reason is not None
    assert "directed path to the question's outcome 'Mood'" in reason
    assert reason.split("No directed path: ")[1] == ", ".join(
        sorted(
            f"{construct.name!r} ({construct.id})"
            for construct in draft.constructs
            if construct.name in outside
        )
    )


def test_an_edit_allows_upstream_causes_confounders_and_feedback_that_reaches_the_outcome():
    dynamical_model_spec = make_model(
        ["U", "Dose", "Mediator", "Mood", "Moderator"],
        [
            ("U", "Dose"),
            ("U", "Mood"),
            ("Dose", "Mediator"),
            ("Mediator", "Mood"),
            ("Mood", "Mediator"),
            ("Moderator", "Mood"),
        ],
    ).revised(measurement_clock=None)
    confounder = next(
        construct for construct in dynamical_model_spec.constructs if construct.name == "U"
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (confounder.revised(indicators=()),))
    )
    assert question_edit_reason(dynamical_model_spec, _question(_model())) is None


def test_an_off_path_model_edit_retains_its_pruning_evidence(tmp_path, monkeypatch):
    from nof1_causal_lab.actions.contracts.requests import EditModelRequest
    from nof1_causal_lab.actions.edit_model import edit_model
    from nof1_causal_lab.storage import paths as data
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from nof1_causal_lab.study.state import StudyState
    from tests.support.helpers import write_question

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("question-path")
    state = StudyState().with_artifacts([write_question(store, _question(_model()))])
    dynamical_model_spec = make_model(
        ["Dose", "Mood", "Side"], [("Dose", "Mood"), ("Dose", "Side")]
    )
    result = edit_model(
        store.workspace_id,
        EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=state.current["question"].revision,
                dynamical_model_spec=dynamical_model_spec,
            )
        ),
    )
    from nof1_causal_lab.study.records import Applied

    assert isinstance(result, Applied)
    assert result.result.constructs == (dynamical_model_spec.constructs[2].id,)
    assert len(result.effects.produced) == 1
