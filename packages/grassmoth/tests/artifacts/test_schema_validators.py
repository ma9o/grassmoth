"""Whole-model validation reports intrinsic errors; measurement readiness is a separate requirement."""

import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.compilation_errors import IncompleteModelError
from grassmoth_test_support.helpers import graph_constructs, invalid_dict_payload, make_model

pytestmark = pytest.mark.contract


def test_dependent_alternatives_reject_impossible_fields_at_the_schema_boundary():
    from grassmoth.artifacts.model.likelihood import ObservationLawSpec
    from grassmoth.artifacts.model.parameter_spec import ParameterTransformSpec
    from grassmoth.artifacts.observations.preparation import ExtractionSpec
    from grassmoth.artifacts.observations.reference import DataRef
    from grassmoth.artifacts.reports.predictive_provenance import PredictiveLawProvenance

    expression = {"kind": "state", "construct_id": "construct:x"}
    for tag, argument, excluded in (
        ("BernoulliLogits", "logits", "probs"),
        ("BernoulliProbs", "probs", "logits"),
    ):
        adapter = TypeAdapter(ObservationLawSpec)
        law = adapter.validate_python({"distribution": tag, argument: expression})
        assert set(law.model_dump()) == {"distribution", argument}
        with pytest.raises(ValidationError, match="extra_forbidden"):
            adapter.validate_python(
                {"distribution": tag, argument: expression, excluded: expression}
            )
    for adapter, payload in (
        (
            TypeAdapter(ExtractionSpec),
            {"kind": "semantic", "how_to_measure": "Read", "computed_rule": None},
        ),
        (
            TypeAdapter(ExtractionSpec),
            {"kind": "semantic", "how_to_measure": "Read", "fill_null": "forward"},
        ),
        (
            TypeAdapter(ExtractionSpec),
            {
                "kind": "computed",
                "how_to_measure": "Read",
                "source_columns": ["value"],
                "fill_null": "mean",
                "fill_null_limit": 2,
            },
        ),
        (TypeAdapter(ParameterTransformSpec), {"kind": "identity", "interval_days": 7}),
        (TypeAdapter(ParameterTransformSpec), {"kind": "dt_effect_to_ct_rate"}),
        (
            TypeAdapter(PredictiveLawProvenance),
            {"kind": "fitted", "interpretation": "posterior_predictive"},
        ),
        (
            TypeAdapter(DataRef[GitOid, int | None]),
            {"kind": "panel", "revision": "a" * 40, "replicate": None},
        ),
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(payload)

    dynamical_model_spec = make_model(["x", "y"], [("x", "y")])
    edge = dynamical_model_spec.edges[0]
    with pytest.raises(ValidationError, match="drift"):
        edge.revised(
            mechanisms=[
                {"id": "mechanism:potential", "kind": "potential", "expression": expression}
            ]
        )


def test_valid_partial_model_can_be_enriched_for_measurement():
    dynamical_model_spec = make_model(["stress", "sleep"], [("stress", "sleep")])
    candidate = DynamicalModelSpec.model_validate(
        dynamical_model_spec.model_dump(mode="json")
    ).materialized()
    candidate.require_measurements()
    assert candidate == dynamical_model_spec
    partial = dynamical_model_spec.revised(measurement_clock=None).with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(c.revised(indicators=()) for c in dynamical_model_spec.constructs),
        ),
    )
    reloaded = DynamicalModelSpec.model_validate(partial.model_dump(mode="json")).materialized()
    with pytest.raises(IncompleteModelError, match="clock and indicators"):
        reloaded.require_measurements()


@pytest.mark.parametrize("payload", ["not a dict", {"constructs": "bad"}])
def test_invalid_authored_structure_returns_errors(payload):
    with pytest.raises(ValidationError):
        DynamicalModelSpec.model_validate(invalid_dict_payload(payload)).materialized()


@pytest.mark.parametrize("bad", ["bad", [42], [{"name": "bad"}]])
def test_invalid_owned_indicator_returns_errors(bad):
    data = make_model(["stress"]).model_dump(mode="json")
    graph_constructs(data)[0]["indicators"] = bad
    with pytest.raises(ValidationError, match="indicators"):
        DynamicalModelSpec.model_validate(data).materialized()


def test_duplicate_indicator_identity_across_constructs_is_rejected():
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    first, second = graph_constructs(data)
    second["indicators"] = first["indicators"]
    with pytest.raises(ValidationError, match="Duplicate indicator IDs"):
        DynamicalModelSpec.model_validate(data).materialized()


def test_duplicate_names_with_distinct_identities_are_rejected():
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    graph_constructs(data)[1]["name"] = "stress"
    with pytest.raises(ValidationError, match="Duplicate construct names"):
        DynamicalModelSpec.model_validate(data).materialized()
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    next(iter(graph_constructs(data)[1]["indicators"].values()))["observation"]["name"] = (
        "stress_obs"
    )
    with pytest.raises(ValidationError, match="Duplicate indicator names"):
        DynamicalModelSpec.model_validate(data).materialized()


def test_validation_collects_errors_from_multiple_entities():
    data = {
        "constructs": {"construct:first": {"name": "bad1"}, "construct:second": {"name": "bad2"}}
    }
    with pytest.raises(ValidationError) as exc:
        DynamicalModelSpec.model_validate(data).materialized()
    assert {error["loc"][1] for error in exc.value.errors() if len(error["loc"]) > 2} == {0, 1}


def test_edge_payload_must_be_a_valid_entity():
    data = make_model(["stress"]).model_dump(mode="json")
    data["edges"] = ["not a dict"]
    with pytest.raises(ValidationError, match="edges"):
        DynamicalModelSpec.model_validate(data).materialized()


@pytest.mark.contract
@pytest.mark.parametrize(
    ("identity_name", "wire"),
    [
        ("GitOid", "a" * 40),
        ("ConstructId", "construct:outcome"),
        ("EdgeId", "edge:exposure-outcome"),
        ("IndicatorId", "indicator:rating"),
        ("MechanismId", "mechanism:drift"),
        ("DistributionId", "distribution:prior"),
        ("ParameterId", "parameter:" + "b" * 64),
        ("ParameterElementId", "element:" + "c" * 64),
    ],
)
def test_identity_construction_and_parsing_share_wire_grammar(
    identity_name: str, wire: str
) -> None:
    from pydantic import TypeAdapter, ValidationError

    from grassmoth.artifacts import identity

    constructor = getattr(identity, identity_name)
    value = constructor(wire)
    adapter = TypeAdapter(constructor)
    assert type(value) is constructor
    assert adapter.validate_json(adapter.dump_json(value)) == value
    with pytest.raises(ValueError, match=f"Invalid {identity_name}"):
        constructor("wrong:identity")
    with pytest.raises(ValidationError):
        adapter.validate_python("wrong:identity")


@pytest.mark.contract
def test_response_presence_is_independent_of_request_defaults_and_excluded_fields() -> None:
    from pydantic import Field

    from grassmoth.artifacts.base import Value
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec

    dynamical_model_spec = DynamicalModelSpec().materialized()
    assert dynamical_model_spec.model_dump(mode="json")["measurement_clock"] is None
    assert (
        "measurement_clock"
        in DynamicalModelSpec.model_json_schema(mode="serialization")["required"]
    )
    assert "measurement_clock" not in DynamicalModelSpec.model_json_schema(mode="validation").get(
        "required", []
    )

    class PrivateField(Value):
        visible: int = 0  # noqa: V107 -- The serialization assertion below reads this field through model_dump.
        excluded: str | None = Field(default=None, exclude=True)

    value = PrivateField(excluded="private")
    assert value.model_dump(mode="json") == {"visible": 0}
    assert "excluded" not in PrivateField.model_json_schema(mode="serialization")["properties"]


def test_owned_mappings_detach_nested_inputs_defaults_and_revisions() -> None:
    from collections.abc import Mapping
    from types import MappingProxyType

    from pydantic import Field

    from grassmoth.artifacts.base import Value

    class Contract(Value):
        groups: Mapping[str, Mapping[str, tuple[int, ...]]] = Field(default_factory=dict)

    source: dict[str, dict[str, tuple[int, ...]]] = {"first": {"values": (1, 2)}}
    value = Contract(groups=source)
    source["first"]["values"] = (9,)
    assert value.groups["first"]["values"] == (1, 2)
    assert isinstance(value.groups, MappingProxyType)
    assert isinstance(value.groups["first"], MappingProxyType)
    assert isinstance(Contract().groups, MappingProxyType)

    changes = {"second": {"values": (3,)}}
    revised = value.revised(groups=changes)
    changes["second"]["values"] = (4,)
    assert revised.groups["second"]["values"] == (3,)
    assert revised.model_dump(mode="json") == {"groups": {"second": {"values": [3]}}}
    assert Contract.model_validate_json(revised.model_dump_json()) == revised
    assert (
        Contract.model_json_schema(mode="serialization")["properties"]["groups"]["type"] == "object"
    )
