"""One document supports atomic creation, partial edits and explicit pruning."""

from datetime import UTC, datetime

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.bodies import EditModelInput
from nof1_causal_lab.actions.contracts.requests import EditModelRequest
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.messages import edit_messages
from nof1_causal_lab.artifacts.identity import GitOid, scientific_id
from nof1_causal_lab.artifacts.model.dynamical_model_spec import (
    DynamicalModelSpec,
    apply_model_edit,
)
from nof1_causal_lab.artifacts.model.expressions import coefficient, state
from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model
from nof1_causal_lab.study.records import Applied, Rejected
from tests.support.helpers import make_model, write_question

pytestmark = pytest.mark.contract


def test_partial_document_round_trip_and_nested_merge_preserve_omission():
    original = make_model(["X", "Y"], [("X", "Y")])
    source, outcome = original.constructs
    mechanism = DriftMechanismSpec(
        id="mechanism:xy",
        expression=(coefficient(1, "weight") + 2) * state(source.id),
    )
    original = original.with_entities(edges=(original.edges[0].revised(mechanisms=(mechanism,)),))
    indicator = source.indicators[0].observation.id
    payload = {
        "constructs": {
            source.id: {
                "name": "Renamed",
                "indicators": {indicator: {"observation": {"name": "new reading"}}},
            }
        },
        "edges": {
            original.edges[0].id: {
                "mechanisms": {mechanism.id: {"expression": {"left": {"left": {"value": 4}}}}}
            }
        },
    }
    Draft202012Validator(DynamicalModelSpec.model_json_schema()).validate(payload)
    supplied = DynamicalModelSpec.model_validate(payload)
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](parent_ref="a" * 40, dynamical_model_spec=supplied)
    )
    restored = EditModelRequest[GitOid].model_validate_json(request.model_dump_json())
    assert restored.input.dynamical_model_spec.model_dump(mode="json") == payload
    edited = apply_model_edit(
        original, restored.input.dynamical_model_spec, outcome.id
    ).dynamical_model_spec
    assert edited.measurement_clock == original.measurement_clock
    assert edited.edges[0].id == original.edges[0].id
    assert edited.get_construct(source.id).description == source.description
    assert edited.get_construct(source.id).name == "Renamed"
    assert edited.indicator(indicator).observation.name == "new reading"
    assert edited.mechanism(mechanism.id).expression == (coefficient(4, "weight") + 2) * state(
        source.id
    )
    assert original.get_construct(source.id).name == "X"


def test_null_edge_prunes_ancestors_and_their_owned_parameters_and_laws():
    parent = make_model(["A", "B", "Y"], [("A", "B"), ("B", "Y")])
    ancestor, middle, outcome = parent.constructs
    identity = scientific_id("parameter", "initial A")
    document = parent.model_dump(mode="json")
    document["constructs"][ancestor.id]["coefficients"] = [
        {"kind": "coefficient", "role": "initial_mean", "value": identity}
    ]
    document["parameters"] = {
        identity: {
            "name": "initial A",
            "description": "Baseline A",
            "distribution": "distribution:initial",
        }
    }
    document["distributions"] = {
        "distribution:initial": {"distribution": "Normal", "params": {"loc": 0, "scale": 1}}
    }
    parent = DynamicalModelSpec.model_validate(document).materialized()
    edit = DynamicalModelSpec.model_validate({"edges": {parent.edges[1].id: None}})
    result = apply_model_edit(parent, edit, outcome.id)
    assert result.dynamical_model_spec.constructs == (outcome,)
    assert result.dynamical_model_spec.edges == result.dynamical_model_spec.parameters == ()
    assert not result.dynamical_model_spec.distributions
    assert result.pruning.constructs == (ancestor.id, middle.id)
    assert result.pruning.edges == (parent.edges[0].id,)
    assert result.pruning.parameters == (identity,)
    assert result.pruning.distributions == ("distribution:initial",)
    from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
    from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport

    (warning,) = edit_messages(
        ModelCheckReport(specification=(), question=QuestionCheckReport(findings=())),
        IdentificationReport(outcome=None),
        result.pruning,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert (warning.severity, warning.code) == ("warning", "MODEL_COMPONENTS_PRUNED")
    assert warning.subject == "model"


def test_disconnected_creation_prunes_before_whole_model_validation():
    source = make_model(["A", "Y", "X", "Z"], [("A", "Y"), ("Y", "X"), ("X", "Z")])
    payload = source.model_dump(mode="json")
    payload["edges"].pop(source.edges[1].id)
    result = apply_model_edit(
        DynamicalModelSpec(), DynamicalModelSpec.model_validate(payload), source.constructs[1].id
    )
    assert tuple(node.name for node in result.dynamical_model_spec.constructs) == ("A", "Y")
    assert len(result.pruning.constructs) == 2


def test_null_nested_entity_and_nullable_field_remove_only_the_supplied_values():
    parent = make_model(["X", "Y"], [("X", "Y")])
    source, outcome = parent.constructs
    indicator = source.indicators[0].observation.id
    supplied = DynamicalModelSpec.model_validate(
        {"constructs": {source.id: {"indicators": {indicator: None}}}, "measurement_clock": None}
    )
    result = apply_model_edit(parent, supplied, outcome.id).dynamical_model_spec
    assert result.get_construct(source.id).indicators == ()
    assert result.measurement_clock is None
    assert result.get_construct(outcome.id).indicators == outcome.indicators


@pytest.mark.parametrize(
    "violation",
    ["missing_definition", "dangling_endpoint", "duplicate_identity", "invalid_indicator"],
)
def test_retained_invalid_edits_fail_only_at_materialization(violation):
    parent = make_model(["X", "Y"], [("X", "Y")])
    source, outcome = parent.constructs
    payload = {
        "missing_definition": {"constructs": {source.id: None}},
        "dangling_endpoint": {"edges": {parent.edges[0].id: {"cause": "construct:missing"}}},
        "duplicate_identity": {"constructs": {source.id: {"id": "construct:other"}}},
        "invalid_indicator": {
            "constructs": {source.id: {"indicators": {source.indicators[0].observation.id: 42}}}
        },
    }[violation]
    supplied = DynamicalModelSpec.model_validate(payload)
    with pytest.raises(ValidationError):
        apply_model_edit(parent, supplied, outcome.id)


def test_action_uses_selected_parent_and_rejects_without_writing(tmp_path, monkeypatch):
    from nof1_causal_lab.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("EDIT")
    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    source, outcome = dynamical_model_spec.constructs
    question = write_question(
        store,
        QuestionSpec(
            text="What affects Y?",
            outcome=outcome.id,
            queries={
                "raise X": {
                    "start": "2026-01-01",
                    "horizon": "2d",
                    "interventions": [{"target": source.id, "value": 1}],
                }
            },
        ),
    )

    def run(parent, payload):
        return edit_model(
            "EDIT",
            EditModelRequest[GitOid](
                input=EditModelInput[GitOid](
                    parent_ref=parent,
                    dynamical_model_spec=DynamicalModelSpec.model_validate(payload),
                )
            ),
        )

    created = run(question.revision, dynamical_model_spec.model_dump(mode="json"))
    assert isinstance(created, Applied)
    parent = created.effects.produced[0]
    changed = run(parent.revision, {"constructs": {source.id: {"name": "renamed"}}})
    assert isinstance(changed, Applied)
    saved = changed.effects.produced[0]
    assert saved.derived_from == {"question": question.revision, "model": parent.revision}
    assert read_model(store, saved.revision).get_construct(source.id).name == "renamed"
    assert read_model(store, parent.revision).get_construct(source.id).name == "X"
    standalone = run(question.revision, {"constructs": {source.id: {"name": "renamed"}}})
    assert isinstance(standalone, Rejected)
    before = tuple(store.repo.references)
    rejected = run(parent.revision, {"edges": {dynamical_model_spec.edges[0].id: None}})
    assert isinstance(rejected, Rejected)
    assert str(source.id) in rejected.detail
    assert tuple(store.repo.references) == before


def test_partial_law_parameters_merge_and_a_changed_family_replaces_them():
    import numpyro.distributions as dist

    parent = make_model(["X", "Y"], [("X", "Y")])
    source, outcome = parent.constructs
    parameter = scientific_id("parameter", "baseline")
    parent = apply_model_edit(
        parent,
        DynamicalModelSpec.model_validate(
            {
                "constructs": {
                    source.id: {
                        "coefficients": [
                            {"kind": "coefficient", "role": "initial_mean", "value": parameter}
                        ]
                    }
                },
                "parameters": {
                    parameter: {
                        "name": "baseline",
                        "description": "Initial X",
                        "distribution": "distribution:baseline",
                    }
                },
                "distributions": {
                    "distribution:baseline": {
                        "distribution": "Normal",
                        "params": {"loc": 0, "scale": 2},
                    }
                },
            }
        ),
        outcome.id,
    ).dynamical_model_spec
    supplied = DynamicalModelSpec.model_validate(
        {"distributions": {"distribution:baseline": {"params": {"loc": 3}}}}
    )
    edited = apply_model_edit(parent, supplied, outcome.id).dynamical_model_spec
    law = edited.distribution_for(parameter)
    assert isinstance(law, dist.Normal)
    assert float(law.loc) == 3
    assert float(law.scale) == 2
    replacement = DynamicalModelSpec.model_validate(
        {
            "distributions": {
                "distribution:baseline": {
                    "distribution": "Uniform",
                    "params": {"low": -1, "high": 1},
                }
            }
        }
    )
    changed = apply_model_edit(edited, replacement, outcome.id).dynamical_model_spec
    assert isinstance(changed.distribution_for(parameter), dist.Uniform)
