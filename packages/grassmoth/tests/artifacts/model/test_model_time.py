"""One saved model owns creation, partial authoring and its aligned temporal domain."""

import pytest
from pydantic import ValidationError

from grassmoth.actions.edit_model import ModelEditFailure, edit_model
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.time import ModelTimeSpec, TimeGridFailure
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth_test_support.purpose import initial_model

pytestmark = pytest.mark.contract


def test_creation_and_nested_query_override_share_one_model_document():
    document = initial_model()
    created = edit_model(None, document)
    assert not isinstance(created, ModelEditFailure)
    assert created.dynamical_model_spec.outcome == document.outcome
    edited = edit_model(
        created.dynamical_model_spec,
        DynamicalModelSpec(
            question="A revised question?",
            core_query={
                "window": {"horizon": "10d"},
                "interventions": [{"target": "construct:treatment", "value": 2}],
            },
        ),
    )
    assert not isinstance(edited, ModelEditFailure)
    model = edited.dynamical_model_spec
    assert model.question == "A revised question?"
    assert model.core_query.window.start == document.core_query.window.start
    assert model.core_query.window.horizon.source == "10d"
    assert model.core_query.grid(model.time) == tuple(float(value) for value in range(11))
    assert any(
        finding.code == "target" and finding.kind == "not_evaluated"
        for finding in edited.checks.question.findings
    )
    assert created.dynamical_model_spec == document


@pytest.mark.parametrize("field", ["question", "outcome", "time", "core_query"])
def test_creation_requires_complete_metadata(field):
    document = initial_model().model_dump(mode="json")
    del document[field]
    result = edit_model(None, DynamicalModelSpec.model_validate(document))
    assert isinstance(result, ModelEditFailure)


def test_model_grid_is_fixed_and_query_extends_from_the_same_origin():
    time = ModelTimeSpec.model_validate(
        {"window": {"start": "2026-01-01", "horizon": "4d"}, "step": "1d"}
    )
    query = SimulationSpec.model_validate(
        {
            "window": {"start": "2026-01-04", "horizon": "4d"},
            "interventions": [{"target": "construct:treatment", "after": "2d", "value": 1}],
        }
    )
    assert time.time_points == (0, 1, 2, 3, 4)
    assert query.grid(time) == (3, 4, 5, 6, 7)
    assert query.assignments(time.window.start_instant)[0].time == 5


@pytest.mark.parametrize(
    "query",
    [
        {"window": {"start": "2025-12-31", "horizon": "2d"}},
        {"window": {"start": "2026-01-02", "horizon": "2d"}},
        {"window": {"start": "2026-01-01", "horizon": "3d"}},
        {
            "window": {"start": "2026-01-01", "horizon": "4d"},
            "interventions": [{"target": "construct:x", "after": "1d", "value": 1}],
        },
    ],
)
def test_queries_preserve_exact_step_alignment(query):
    time = ModelTimeSpec.model_validate(
        {"window": {"start": "2026-01-01", "horizon": "4d"}, "step": "2d"}
    )
    assert isinstance(SimulationSpec.model_validate(query).grid(time), TimeGridFailure)


def test_model_horizon_cannot_stretch_its_step():
    with pytest.raises(ValidationError, match="whole number"):
        ModelTimeSpec.model_validate(
            {"window": {"start": "2026-01-01", "horizon": "3d"}, "step": "2d"}
        )


def test_grid_edits_prune_incompatible_retained_joint_laws():
    from grassmoth_test_support.model import load_model_fixture

    parent = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    retained = tuple(
        identity for identity, layout in parent.law_layouts.items() if layout.constructs
    )
    unchanged = edit_model(parent, DynamicalModelSpec(question="Reworded question"))
    assert not isinstance(unchanged, ModelEditFailure)
    assert unchanged.dynamical_model_spec.law_layouts == parent.law_layouts
    changed = edit_model(parent, DynamicalModelSpec(time={"window": {"horizon": "2d"}}))
    assert not isinstance(changed, ModelEditFailure)
    assert set(changed.pruning.distributions) == set(retained)
    assert not changed.dynamical_model_spec.law_layouts
    assert all(node.distribution is None for node in changed.dynamical_model_spec.constructs)
    assert parent.law_layouts
