"""Independent additive terms survive revision without tying their free coefficients."""

from pathlib import Path

import numpy as np
import pytest

from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.expressions import BinaryExpression
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors
from nof1_causal_lab.models.structure import StructuralSelection
from tests.support.inference import compile_model_fixture
from tests.support.model import load_model_fixture


def _two_hills_model() -> DynamicalModelSpec:
    return load_model_fixture("mechanism_identity/two_hills_model.json")


@pytest.fixture(scope="module")
def two_hills():
    return _two_hills_model()


@pytest.mark.inference(concern="sampling")
def test_independent_hill_coefficients_survive_reorder_rename_and_submission(two_hills):
    model = two_hills
    edge = model.edges[0]
    compile_model_fixture(model)
    before = parameter_bindings(compile_model_fixture(model))[0]
    before_priors = compile_priors(compile_model_fixture(model), StructuralSelection(model, None))[
        0
    ]
    revised = model.with_entities(
        edges=(edge.revised(mechanisms=tuple(reversed(edge.mechanisms))),),
        parameters=tuple(p.revised(name=f"new label {i}") for i, p in enumerate(model.parameters)),
    )
    compile_model_fixture(revised)
    after = parameter_bindings(compile_model_fixture(revised))[0]
    after_priors = compile_priors(
        compile_model_fixture(revised), StructuralSelection(revised, None)
    )[0]
    identities = {p.id for term in edge.mechanisms for p in model.parameters_for(term.id)}
    assert len(identities) == 6
    old = {b.parameter_id: b for b in before if b.parameter_id in identities}
    new = {b.parameter_id: b for b in after if b.parameter_id in identities}
    assert old.keys() == new.keys() == identities
    assert len({b.site.name for b in old.values()}) == 6
    for identity in identities:
        assert old[identity].elements.keys() == new[identity].elements.keys()
        assert old[identity].coordinates != new[identity].coordinates
        np.testing.assert_allclose(
            before_priors[old[identity].site.name].log_prob(0.7),
            after_priors[new[identity].site.name].log_prob(0.7),
        )
    # Incremental edits submit the whole authored model directly.

    submitted = DynamicalModelSpec.model_validate(revised.model_dump(mode="json")).materialized()
    assert submitted.edges[0].mechanisms == revised.edges[0].mechanisms
    removed_ids = {p.id for p in model.parameters_for(edge.mechanisms[1].id)}
    remaining = tuple(p for p in model.parameters if p.id not in removed_ids)
    edited = model.with_entities(
        edges=(edge.revised(mechanisms=(edge.mechanisms[0],)),),
        parameters=remaining,
        distributions={
            key: law
            for key, law in model.distributions.items()
            if key in {p.distribution for p in remaining}
        },
    )
    assert {p.id for p in edited.parameters} == {p.id for p in model.parameters} - removed_ids
    compile_model_fixture(edited)


@pytest.mark.contract
@pytest.mark.parametrize("change", ["duplicate", "wrong_coefficient", "delete"])
def test_term_identity_rejects_ambiguous_or_dangling_revisions(two_hills, change):
    model = two_hills
    edge = model.edges[0]
    first, second = edge.mechanisms
    if change == "duplicate":
        terms = (first, second.revised(id=first.id))
    elif change == "wrong_coefficient":
        terms = (
            first,
            second.revised(
                expression=BinaryExpression.model_validate_json(
                    (
                        Path(__file__).resolve().parents[3]
                        / "fixtures/models"
                        / "mechanism_identity/term_identity_rejects_ambiguous_or_dangling_revisions_map_expression.json"
                    ).read_text()
                )
            ),
        )
    else:
        terms = (first,)
    with pytest.raises(ValueError, match=r"Duplicate mechanism|not referenced by component slots"):
        model.with_entities(edges=(edge.revised(mechanisms=terms),))
    assert model.mechanism(second.id) is second
