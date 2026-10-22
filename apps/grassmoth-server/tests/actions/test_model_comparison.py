"""Spec comparisons are lossless edit documents with no execution interpretation."""

from itertools import product

import numpyro.distributions as dist
import pytest

from grassmoth.actions.contracts.bodies import ModelDiffOutput
from grassmoth.artifacts.model.document import diff_fields, merge_fields
from grassmoth.artifacts.model.dynamical_model_spec import (
    DynamicalModelSpec,
    apply_model_edit,
)
from tests.support.helpers import make_model
from tests.support.model import load_model_fixture

pytestmark = pytest.mark.contract


def test_spec_diffs_replay_nested_edits_pruning_and_empty_checkpoints_in_both_directions():
    before = make_model(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")])
    source, outcome, pruned = before.constructs
    indicator = outcome.indicators[0].observation
    after = apply_model_edit(
        before,
        DynamicalModelSpec.model_validate(
            {
                "constructs": {
                    source.id: {"name": "Exposure"},
                    outcome.id: {
                        "indicators": {indicator.id: {"observation": {"name": "New measure"}}}
                    },
                },
                "edges": {before.edges[1].id: None},
                "measurement_clock": None,
            }
        ),
        outcome.id,
    ).dynamical_model_spec
    patch = after.changes_from(before).model_dump(mode="json")
    assert patch == {
        "constructs": {
            source.id: {"name": "Exposure"},
            outcome.id: {"indicators": {indicator.id: {"observation": {"name": "New measure"}}}},
            pruned.id: None,
        },
        "edges": {before.edges[1].id: None},
        "measurement_clock": None,
    }
    for left, right in product((DynamicalModelSpec.from_entities(), before, after), repeat=2):
        changes = right.changes_from(left)
        assert merge_fields(left.model_dump(mode="json"), changes.model_dump(mode="json")) == (
            right.model_dump(mode="json")
        )
        assert (
            ModelDiffOutput.model_validate_json(
                ModelDiffOutput(changes=changes).model_dump_json()
            ).changes
            == changes
        )
    assert before.changes_from(before).model_dump(mode="json") == {}


def test_law_changes_stay_at_their_owner_and_changed_families_replace_the_variant():
    dynamical_model_spec = load_model_fixture("model_comparison/y_z_model.json")
    parameter = dynamical_model_spec.parameters_for(dynamical_model_spec.edges[0].id)[0]
    identity = parameter.distribution
    assert identity is not None
    before = dynamical_model_spec.with_entities(
        distributions={**dynamical_model_spec.distributions, identity: dist.Normal(0, 1)}
    )
    for law in (dist.Normal(2, 1), dist.Uniform(-1, 1)):
        after = before.with_entities(distributions={**before.distributions, identity: law})
        patch = after.changes_from(before).model_dump(mode="json")
        assert set(patch) == {"distributions"}
        assert merge_fields(before.model_dump(mode="json"), patch) == after.model_dump(mode="json")
        if isinstance(law, dist.Normal):
            assert patch["distributions"][identity] == {"params": {"loc": 2.0}}
        else:
            assert (
                patch["distributions"][identity]
                == after.model_dump(mode="json")["distributions"][identity]
            )


def test_map_order_is_ignored_and_ordered_values_are_replaced():
    before = make_model(["X", "Y"], [("X", "Y")])
    saved = before.model_dump(mode="json")
    reordered = DynamicalModelSpec.model_validate(
        {**saved, "constructs": dict(reversed(tuple(saved["constructs"].items())))}
    )
    assert reordered.changes_from(before).model_dump(mode="json") == {}
    assert diff_fields({"time_points": [0, 1]}, {"time_points": [1, 0]}) == {"time_points": [1, 0]}


def test_referenced_laws_compare_without_resolving_numerical_buffers():
    identity = "distribution:stored"
    law = {
        "distribution": "Delta",
        "params": {
            "v": {"array_ref": "a" * 64, "shape": [2], "dtype": "float64", "index": []},
            "event_dim": 1,
        },
    }
    before = DynamicalModelSpec.model_validate({"distributions": {identity: law}})
    after = DynamicalModelSpec.model_validate(
        {
            "distributions": {
                identity: {
                    **law,
                    "params": {**law["params"], "v": {**law["params"]["v"], "array_ref": "b" * 64}},
                }
            }
        }
    )
    assert after.changes_from(before).model_dump(mode="json") == {
        "distributions": {identity: {"params": {"v": {"array_ref": "b" * 64}}}}
    }


def test_element_label_maps_reuse_identity_deletion_semantics():
    before = {"labels": {"element:a": "A", "element:b": "B"}}
    after = {"labels": {"element:a": "Renamed"}}
    patch = diff_fields(before, after)
    assert patch == {"labels": {"element:a": "Renamed", "element:b": None}}
    assert merge_fields(before, patch) == after


def test_output_serialization_preserves_the_partial_model_contract():
    assert ModelDiffOutput(changes=DynamicalModelSpec()).model_dump(mode="json") == {"changes": {}}
