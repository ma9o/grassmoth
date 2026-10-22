"""Coefficients used only through a sum or a product cannot be told apart by any data."""

import re

import pytest

from grassmoth.artifacts.identity import scientific_id
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import CallExpression, coefficient, state
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from tests.support.helpers import fixture_entity_id, make_model

pytestmark = pytest.mark.contract

_A, _B, _C = (scientific_id("parameter", name) for name in "abc")


def _model(*mechanisms, reused=False):
    """X -> Y carrying the given drift terms; optionally b also drives Y's own drift."""
    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    edge = dynamical_model_spec.edges[0]
    y = edge.effect
    terms = tuple(
        DriftMechanismSpec(id=fixture_entity_id("mechanism", f"m{index}"), expression=expression)
        for index, expression in enumerate(mechanisms)
    )
    own = (
        (
            DriftMechanismSpec(
                id=fixture_entity_id("mechanism", "own"),
                expression=coefficient(_B, "weight") * state(y.id),
            ),
        )
        if reused
        else ()
    )
    used = {
        identity
        for item in mechanisms
        for identity in (_A, _B, _C)
        if identity in item.model_dump_json()
    }
    return DynamicalModelSpec.from_entities(
        edges=(edge.revised(effect=y.revised(dynamics=own), mechanisms=terms),),
        parameters=tuple(
            ParameterSpec(id=identity, name=name, description=name)
            for identity, name in zip((_A, _B, _C), "abc", strict=True)
            if identity in used
        ),
        measurement_clock="1d",
    )


def _x():
    return state(make_model(["X", "Y"], [("X", "Y")]).constructs[0].id)


@pytest.mark.parametrize(
    ("mechanisms", "form"),
    [
        ((coefficient(_A, "weight") * _x(), coefficient(_B, "weight") * _x()), "a + b"),
        (((coefficient(_A, "weight") + coefficient(_B, "weight")) * _x(),), "a + b"),
        (((coefficient(_A, "weight") - coefficient(_B, "weight")) * _x(),), "a - b"),
        ((coefficient(_A, "weight") * coefficient(_B, "weight") * _x(),), "a·b"),
        (
            tuple(coefficient(identity, "weight") * _x() for identity in (_A, _B, _C)),
            "a + b + c",
        ),
    ],
)
def test_combinations_no_data_can_separate_are_rejected(mechanisms, form):
    with pytest.raises(ValueError, match=re.escape(f"enter the model only through {form}")):
        _model(*mechanisms)


def test_a_separate_use_identifies_each_coefficient():
    dynamical_model_spec = _model(
        coefficient(_A, "weight") * _x(), coefficient(_B, "weight") * _x(), reused=True
    )
    assert {item.name for item in dynamical_model_spec.parameters} == {"a", "b"}


def test_opaque_uses_do_not_certify_a_product_redundancy():
    dynamical_model_spec = _model(
        coefficient(_A, "weight") * coefficient(_B, "weight") * _x(),
        CallExpression(function="exp", arguments=(coefficient(_B, "weight") * _x(),)),
    )
    assert {item.name for item in dynamical_model_spec.parameters} == {"a", "b"}
