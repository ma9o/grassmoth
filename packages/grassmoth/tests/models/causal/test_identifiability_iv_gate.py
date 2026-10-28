"""Tests for the ``iv_allowed`` gate on ``check_identifiability``.

The flag controls whether graph-theoretic IV candidates can be reported under
an explicit parametric linearity assumption.
"""

from __future__ import annotations

import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.reports.identification import IdentifiedTreatmentStatus
from grassmoth.models.causal.identification import check_identifiability
from grassmoth.models.structure import StructuralSelection
from grassmoth_test_support.causal_graphs import make_graph
from grassmoth_test_support.helpers import fixture_entity_id

pytestmark = pytest.mark.contract

X = fixture_entity_id("construct", "X")


def _iv_graph():
    """A static IV pattern: U → X → Y and Z → X, with U unobserved."""
    return make_graph(
        [{"name": name, "temporal_status": "time_invariant"} for name in ("X", "Y", "Z", "U")],
        [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    )


class TestIVAllowedDefault:
    def test_default_does_not_promote_a_linear_iv_candidate(self):
        result = check_identifiability(
            *_iv_graph(),
            outcome_id=fixture_entity_id("construct", "Y"),
            observed_constructs={"X", "Y", "Z"},
        )
        assert X not in result.estimable_treatments
        assert X in result.non_identifiable


class TestIVAllowedFalse:
    def test_disabled_iv_gate_skips_iv(self):
        """An IV-only pattern remains unidentified without a linearity assumption."""
        graph = _iv_graph()
        result_with_iv = check_identifiability(
            *graph,
            outcome_id=fixture_entity_id("construct", "Y"),
            observed_constructs={"X", "Y", "Z"},
            iv_allowed=True,
        )
        result_no_iv = check_identifiability(
            *graph,
            outcome_id=fixture_entity_id("construct", "Y"),
            observed_constructs={"X", "Y", "Z"},
            iv_allowed=False,
        )

        finding = result_with_iv.treatments[X]
        assert isinstance(finding, IdentifiedTreatmentStatus)
        assert finding.instruments == (fixture_entity_id("construct", "Z"),)
        assert X not in result_no_iv.estimable_treatments
        assert X in result_no_iv.non_identifiable

    def test_disabled_iv_gate_preserves_do_calculus_identifications(self):
        """The IV gate does not change nonparametric identification."""
        graph = make_graph(
            [{"name": name, "temporal_status": "time_invariant"} for name in ("X", "Y")],
            [{"cause": "X", "effect": "Y"}],
        )
        result_with_iv = check_identifiability(
            *graph,
            outcome_id=fixture_entity_id("construct", "Y"),
            observed_constructs={"X", "Y"},
            iv_allowed=True,
        )
        result_no_iv = check_identifiability(
            *graph,
            outcome_id=fixture_entity_id("construct", "Y"),
            observed_constructs={"X", "Y"},
            iv_allowed=False,
        )

        with_iv, no_iv = result_with_iv.treatments[X], result_no_iv.treatments[X]
        assert isinstance(with_iv, IdentifiedTreatmentStatus)
        assert isinstance(no_iv, IdentifiedTreatmentStatus)
        assert with_iv.estimand == no_iv.estimand


def test_model_reporting_keeps_nonparametric_findings_without_linear_iv_assumptions():
    from grassmoth.artifacts.model.construct import TemporalStatus
    from grassmoth.models.causal.analysis import identify_model
    from grassmoth_test_support.helpers import make_model

    dynamical_model_spec = make_model(
        ["X", "Y", "Z", "U"], [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")]
    )
    identities = {construct.name: construct.id for construct in dynamical_model_spec.constructs}
    x_id, y_id, u_id = (identities[name] for name in ("X", "Y", "U"))
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(
                    temporal_status=TemporalStatus.TIME_INVARIANT,
                    indicators=() if construct.id == u_id else construct.indicators,
                )
                for construct in dynamical_model_spec.constructs
            ),
        ),
    )
    report = identify_model(StructuralSelection(dynamical_model_spec, y_id))
    assert x_id not in report.estimable_treatments
    assert x_id in report.non_identifiable

    unconfounded = dynamical_model_spec.with_entities(
        edges=tuple(edge for edge in dynamical_model_spec.edges if edge.cause.id != u_id)
    )
    identified = identify_model(StructuralSelection(unconfounded, y_id))
    finding = identified.treatments[x_id]
    assert finding.status == "identified"
    assert finding.estimand
    assert finding.instruments == ()
