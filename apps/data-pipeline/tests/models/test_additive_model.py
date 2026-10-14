"""Scientific entities gain detail through stable component references."""

from pathlib import Path

import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, scientific_id
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.expressions import coefficient, hill, state
from nof1_causal_lab.artifacts.model.likelihood import ObservationLawSpec
from nof1_causal_lab.models.parameters import require_priors
from nof1_causal_lab.models.structure import StructuralSelection
from tests.support.helpers import graph_constructs
from tests.support.inference import compile_model_fixture
from tests.support.model import load_model_fixture


def _entities_gain_detail_with_one_owner_and_native_prior_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "additive_model/entities_gain_detail_with_one_owner_and_native_prior_with_parameter_distributions.json"
    )


pytestmark = pytest.mark.contract


def _model():
    weight = scientific_id("parameter", "weight")
    emax = scientific_id("parameter", "emax")
    return DynamicalModelSpec.model_validate(
        {
            "constructs": {
                "construct:x": {
                    "name": "X",
                    "description": "X",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                "construct:y": {
                    "name": "Y",
                    "description": "Y",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": {
                        "indicator:y": {
                            "observation": {
                                "name": "measured_y",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "construct_polarity": "positive",
                        }
                    },
                },
            },
            "edges": {
                "edge:xy": {
                    "cause": "construct:x",
                    "effect": "construct:y",
                    "description": "X influences Y",
                    "mechanisms": {
                        "mechanism:linear": {
                            "kind": "drift",
                            "expression": (
                                coefficient(weight, "weight") * state(ConstructId("construct:x"))
                            ).model_dump(mode="json"),
                        },
                        "mechanism:hill": {
                            "kind": "drift",
                            "expression": hill(
                                state(ConstructId("construct:x")), emax=emax, ec50=1, n=2
                            ).model_dump(mode="json"),
                        },
                    },
                }
            },
            "parameters": {
                weight: {"name": "effect", "description": "Linear effect"},
                emax: {"name": "emax", "description": "Maximum Hill effect"},
            },
        }
    ).materialized()


def test_entities_gain_detail_with_one_owner_and_native_prior():
    before = _model()
    payload = before.model_dump(mode="python")
    next(iter(graph_constructs(payload)[1]["indicators"].values()))["likelihood"] = {
        "law": TypeAdapter(ObservationLawSpec)
        .validate_json(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/models"
                / "common/y_gaussian_observation_law.json"
            ).read_text()
        )
        .model_dump(mode="json"),
        "reasoning": "Continuous measurement",
    }

    after = _entities_gain_detail_with_one_owner_and_native_prior_with_parameter_distributions()
    assert after.indicator_owner(IndicatorId("indicator:y")) is after.get_construct(
        ConstructId("construct:y")
    )
    assert (
        after.indicator(IndicatorId("indicator:y")).observation.id
        == before.indicator("indicator:y").observation.id
    )
    assert after.indicator(IndicatorId("indicator:y")).likelihood is not None
    assert before.indicator("indicator:y").likelihood is None
    assert after.parameters[0].id == before.parameters[0].id
    assert isinstance(after.distribution_for(after.parameters[0].id), dist.Normal)
    assert before.parameters[0].distribution is None
    encoded = after.model_dump(mode="json")
    assert "construct_id" not in next(iter(graph_constructs(encoded)[1]["indicators"].values()))
    assert (
        "indicator_id"
        not in next(iter(graph_constructs(encoded)[1]["indicators"].values()))["likelihood"]
    )
    assert all(
        set(term) == {"kind", "expression"}
        for term in next(iter(encoded["edges"].values()))["mechanisms"].values()
    )
    assert all(
        "edge_id" not in term
        for term in next(iter(encoded["edges"].values()))["mechanisms"].values()
    )
    assert after.edges[0].id == before.edges[0].id
    assert after.edges[0].mechanisms == before.edges[0].mechanisms
    assert "policies" not in encoded
    assert {"quantity", "owners", "elements"}.isdisjoint(next(iter(encoded["parameters"].values())))


def test_partial_model_is_valid_but_operation_requirements_are_explicit():
    initial = DynamicalModelSpec()
    assert initial.edges == initial.constructs == ()
    assert DynamicalModelSpec.model_validate({}).materialized().edges == ()
    with pytest.raises(ValueError, match="clock"):
        compile_model_fixture(initial)
    partial = _model()
    with pytest.raises(ValueError, match="clock"):
        partial.require_measurements()
    with pytest.raises(ValueError, match="prior laws"):
        require_priors(StructuralSelection(partial, None))
    complete_measurements = partial.revised(measurement_clock="1d")
    complete_measurements.require_measurements()
    assert partial.measurement_clock is None


@pytest.mark.parametrize(
    "change",
    [
        "delete_owner",
        "coefficient_owner",
        "dangling_parameter",
        "duplicate_indicator",
        "incompatible_likelihood",
    ],
)
def test_inconsistent_enrichment_is_rejected(change):
    payload = _model().model_dump(mode="json")
    if change == "delete_owner":
        payload["constructs"].pop("construct:x")
    elif change == "coefficient_owner":
        next(iter(payload["parameters"].values()))["owners"] = [
            {"kind": "construct", "id": "construct:y"}
        ]
    elif change == "dangling_parameter":
        payload["parameters"].pop(next(iter(payload["parameters"])))
    elif change == "duplicate_indicator":
        graph_constructs(payload)[0]["indicators"] = graph_constructs(payload)[1]["indicators"]
    else:
        next(iter(graph_constructs(payload)[1]["indicators"].values()))["likelihood"] = {
            "law": TypeAdapter(ObservationLawSpec)
            .validate_json(
                (
                    Path(__file__).resolve().parents[1]
                    / "fixtures/models"
                    / "additive_model/inconsistent_enrichment_is_rejected_observation_law.json"
                ).read_text()
            )
            .model_dump(mode="json"),
            "reasoning": "Wrong type",
        }
    with pytest.raises(ValidationError):
        DynamicalModelSpec.model_validate(payload).materialized()


def test_shared_endpoints_round_trip_once_and_resolve_forward_references():
    from tests.support.helpers import make_model

    dynamical_model_spec = make_model(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
    assert dynamical_model_spec.edges[0].effect is dynamical_model_spec.edges[1].effect
    payload = dynamical_model_spec.model_dump(mode="json")
    assert len(payload["constructs"]) == 3
    assert "constructs" in DynamicalModelSpec.model_json_schema()["properties"]
    assert list(payload["edges"].values())[1]["effect"] == dynamical_model_spec.edges[0].effect.id
    payload["edges"] = dict(reversed(payload["edges"].items()))
    restored = DynamicalModelSpec.model_validate(payload).materialized()
    assert restored.edges[0].effect is restored.edges[1].effect
    assert (
        DynamicalModelSpec.model_validate_json(
            dynamical_model_spec.model_dump_json()
        ).materialized()
        == dynamical_model_spec
    )
    changed = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            [dynamical_model_spec.edges[0].effect.revised(name="Renamed Y")],
        )
    )
    assert changed.edges[0].effect is changed.edges[1].effect
    assert changed.edges[1].effect.name == "Renamed Y"
    assert dynamical_model_spec.edges[1].effect.name == "Y"


def test_endpoint_identity_rejects_conflicting_definitions():
    from tests.support.helpers import make_model

    dynamical_model_spec = make_model(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
    payload = dynamical_model_spec.model_dump(mode="json")
    payload["constructs"][dynamical_model_spec.edges[0].effect.id]["id"] = "construct:different"
    with pytest.raises(ValidationError, match="identities belong in their map keys"):
        DynamicalModelSpec.model_validate(payload).materialized()


def test_graph_membership_follows_edges_and_revisions_preserve_connectivity():
    from tests.support.helpers import make_model

    dynamical_model_spec = make_model(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")])
    with pytest.raises(ValidationError, match="connected causal graph"):
        dynamical_model_spec.with_entities(
            edges=(dynamical_model_spec.edges[0], dynamical_model_spec.edges[2])
        )
    assert dynamical_model_spec.with_entities(edges=()).constructs == ()
    shortened = dynamical_model_spec.with_entities(edges=dynamical_model_spec.edges[:-1])
    assert [construct.name for construct in shortened.constructs] == ["A", "B", "C"]
    assert [construct.name for construct in dynamical_model_spec.constructs] == ["A", "B", "C", "D"]
