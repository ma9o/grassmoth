"""Execution planning reads canonical entities and preserves their source identities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from grassmoth.artifacts.model.construct import Role, TemporalStatus, replace_constructs
from grassmoth.artifacts.model.expressions import coefficient, state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from grassmoth.models.parameters import execution_parameters
from grassmoth.models.ssm.compile.bindings import parameter_bindings
from grassmoth.models.ssm.compile.layout.static_factors import _build_static_factor_structure
from grassmoth.models.structure import (
    StructuralCompilationError,
    StructuralSelection,
    reference_indicators,
    selected_edges,
    selected_indicators,
    selected_state_ids,
    validate_execution_structure,
)
from tests.support.helpers import make_model
from tests.support.inference import compile_fit_fixture, compile_model_fixture, particle_posterior
from tests.support.model import load_model_fixture, x_y_model


def _severed_components_do_not_require_priors_or_bind_numerical_parameters_complete_test_model() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "structural_compiler/severed_components_do_not_require_priors_or_bind_numerical_parameters_complete_test_model.json"
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


@pytest.mark.contract
def test_planner_rejects_retained_static_target_edge():
    dynamical_model_spec = make_model(
        ["X", "Baseline", "Y"], [("X", "Baseline"), ("Baseline", "Y")]
    )
    x, baseline, y = dynamical_model_spec.constructs
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                x.revised(role=Role.ENDOGENOUS, temporal_status=TemporalStatus.TIME_INVARIANT),
                baseline.revised(temporal_status=TemporalStatus.TIME_INVARIANT),
                y,
            ),
        )
    )
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        validate_execution_structure(StructuralSelection(dynamical_model_spec, None))


@pytest.mark.contract
def test_model_rejects_duplicate_endpoint_pairs():
    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    duplicate = dynamical_model_spec.edges[0].revised(id="edge:another")
    with pytest.raises(ValueError, match="one causal edge per endpoint pair"):
        dynamical_model_spec.with_entities(edges=(*dynamical_model_spec.edges, duplicate))


@pytest.mark.contract
def test_projected_coefficients_require_literals():
    import numpy as np

    from grassmoth.artifacts.identity import scientific_id
    from grassmoth.models.ssm.compile.mechanisms import iter_mechanism_components

    dynamical_model_spec = make_model(["U", "X", "Y"], [("U", "Y"), ("X", "Y")])
    root = next(item for item in dynamical_model_spec.constructs if item.name == "U")
    root = root.revised(
        role=Role.ENDOGENOUS,
        temporal_status=TemporalStatus.TIME_INVARIANT,
        indicators=(),
        coefficients=(coefficient(0.0, "initial_mean"), coefficient(1.0, "initial_scale")),
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (root,))
    )
    parameter = ParameterSpec(
        id=scientific_id("parameter", "fixed-loading"),
        name="loading",
        description="Known loading",
    )

    def _with_loading(weight, parameters):
        edges = tuple(
            edge.revised(
                mechanisms=(
                    DriftMechanismSpec(
                        id="mechanism:fixed-loading",
                        expression=coefficient(weight, "weight") * state(root.id),
                    ),
                )
            )
            if edge.cause.id == root.id
            else edge
            for edge in dynamical_model_spec.edges
        )
        return dynamical_model_spec.with_entities(edges=edges, parameters=parameters)

    literal = StructuralSelection(_with_loading(0.5, ()), None)
    assert np.any(
        np.asarray(
            _build_static_factor_structure(
                literal,
                tuple(
                    literal.dynamical_model_spec.get_construct(identity).name
                    for identity in selected_state_ids(literal)
                ),
            )[2]
        )
        == 0.5
    )
    tuple(iter_mechanism_components(literal.dynamical_model_spec, selected_state_ids(literal)))
    unresolved = StructuralSelection(_with_loading(parameter.id, (parameter,)), None)
    with pytest.raises(ValueError, match="fixed linear"):
        _build_static_factor_structure(
            unresolved,
            tuple(
                unresolved.dynamical_model_spec.get_construct(identity).name
                for identity in selected_state_ids(unresolved)
            ),
        )[2]
    with pytest.raises(ValueError, match="fixed linear"):
        tuple(
            iter_mechanism_components(
                unresolved.dynamical_model_spec, selected_state_ids(unresolved)
            )
        )


def _model_with_exact_measurement():

    dynamical_model_spec = make_model(
        ["X", "Y", "Driver", "History", "U"],
        [("X", "Y"), ("Driver", "Y"), ("History", "Y"), ("U", "X"), ("U", "Y")],
    )
    x, y, driver, history, u = dynamical_model_spec.constructs
    indicator = driver.indicators[0].revised(
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(driver.id)),
            reasoning="Direct exact driver observation",
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                x,
                y,
                driver.revised(role=Role.EXOGENOUS, indicators=(indicator,)),
                history,
                u.revised(role=Role.ENDOGENOUS, indicators=()),
            ),
        )
    )


@pytest.mark.contract
def test_exact_measurements_retain_scientific_states_and_project_only_supported_roots():
    dynamical_model_spec = _model_with_exact_measurement()
    selection = StructuralSelection(dynamical_model_spec, None)
    by_name = {c.name: c for c in dynamical_model_spec.constructs}
    assert [
        dynamical_model_spec.get_construct(key).name for key in selected_state_ids(selection)
    ] == [
        "X",
        "Y",
        "Driver",
        "History",
    ]
    assert by_name["Driver"].id in selection.retained_construct_ids
    assert by_name["History"].id in selection.retained_construct_ids
    assert by_name["U"].id in selection.marginalized_construct_ids
    assert by_name["Driver"].indicators[0].observation.id in tuple(
        indicator.observation.id for indicator in selected_indicators(selection)
    )
    assert selection.induced_dependencies
    assert selected_edges(selection)[0] is dynamical_model_spec.edges[0]


@pytest.mark.contract
def test_source_ids_are_stable_across_authoring_reordering():
    dynamical_model_spec = _model_with_exact_measurement()
    original = StructuralSelection(dynamical_model_spec, None)
    reordered = StructuralSelection(
        dynamical_model_spec.with_entities(
            edges=replace_constructs(
                tuple(reversed(dynamical_model_spec.edges)),
                tuple(reversed(dynamical_model_spec.constructs)),
            )
        ),
        None,
    )
    assert original.retained_construct_ids == reordered.retained_construct_ids
    assert original.marginalized_construct_ids == reordered.marginalized_construct_ids
    assert original.induced_dependencies == reordered.induced_dependencies
    assert reference_indicators(original) == reference_indicators(reordered)


@pytest.mark.contract
def test_execution_checks_preserve_the_scientific_model():
    dynamical_model_spec = x_y_model()
    before = dynamical_model_spec.model_dump(mode="json")
    compile_model_fixture(dynamical_model_spec)
    from grassmoth.models.ssm import compiled_access as numeric

    assert numeric.observation_names(compile_model_fixture(dynamical_model_spec)) == (
        "X_obs",
        "Y_obs",
    )
    assert {
        b.parameter_id for b in parameter_bindings(compile_model_fixture(dynamical_model_spec))[0]
    } == {p.id for p in dynamical_model_spec.parameters}
    assert dynamical_model_spec.model_dump(mode="json") == before


@pytest.mark.contract
def test_required_unmeasured_mediator_cannot_be_silently_excluded():
    dynamical_model_spec = make_model(
        ["X", "Mediator", "Y"], [("X", "Mediator"), ("Mediator", "Y")]
    )
    x, mediator, y = dynamical_model_spec.constructs
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                x,
                mediator.revised(indicators=()),
                y,
            ),
        ),
    )
    selection = StructuralSelection(dynamical_model_spec, y.id)
    with pytest.raises(StructuralCompilationError, match="Required constructs"):
        validate_execution_structure(selection)


@pytest.mark.inference(concern="predictive")
def test_severed_components_do_not_require_priors_or_bind_numerical_parameters():
    import jax
    import jax.numpy as jnp

    from grassmoth.actions.fit.conditioning import condition_model
    from grassmoth.models.ssm import compiled_access as numeric
    from grassmoth.models.ssm.inference.types import JointPosteriorDraws
    from grassmoth.models.ssm.predictive.parameters import sample_model_laws
    from tests.support.inference import model_draws

    dynamical_model_spec = make_model(
        ["A", "B", "Sink", "X", "Y"],
        [("A", "B"), ("B", "Sink"), ("Y", "Sink"), ("X", "Y")],
    )
    nodes = {item.name: item for item in dynamical_model_spec.constructs}
    dynamical_model_spec = (
        _severed_components_do_not_require_priors_or_bind_numerical_parameters_complete_test_model()
    )
    island_parameter = dynamical_model_spec.parameters_for(nodes["A"].id)[0]
    outcome = nodes["Y"].id
    unassigned = dynamical_model_spec.with_entities(
        parameters=tuple(
            item.revised(distribution=None) if item.id == island_parameter.id else item
            for item in dynamical_model_spec.parameters
        ),
        distributions={
            key: value
            for key, value in dynamical_model_spec.distributions.items()
            if key != island_parameter.distribution
        },
    )
    selected = StructuralSelection(unassigned, outcome)
    before = unassigned.model_dump(mode="json")
    assert set(numeric.state_names(compile_model_fixture(unassigned, outcome))) == {"X", "Y"}
    assert len(tuple(indicator.observation.id for indicator in selected_indicators(selected))) == 2
    assert island_parameter.id not in {item.id for item in execution_parameters(selected)}
    bindings = compile_fit_fixture(unassigned, outcome).compiled_dynamical_model.bindings
    assert {item.parameter_id for item in bindings} == {
        item.id for item in execution_parameters(selected)
    }
    assert unassigned.model_dump(mode="json") == before

    draws = sample_model_laws(
        compile_model_fixture(unassigned, outcome), draws=2, key=jax.random.PRNGKey(0)
    )
    assert set(draws.state_ids) == {nodes["X"].id, nodes["Y"].id}
    # Exercise persistence with synthetic draws; no fitting or trajectory simulation.
    conditioned, _ = condition_model(
        unassigned,
        compile_model_fixture(unassigned, outcome),
        particle_posterior(
            JointPosteriorDraws(draws.parameters, jnp.zeros((2, 2, 2)), draws.state_ids)
        ),
        times=jnp.array([0.0, 1.0]),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert conditioned.parameter(island_parameter.id) == unassigned.parameter(island_parameter.id)
    assert set(conditioned.distributions) & set(unassigned.distributions)
    assert model_draws(compile_model_fixture(conditioned, outcome)).state_ids == draws.state_ids
    assert (
        sample_model_laws(
            compile_model_fixture(conditioned, outcome), draws=2, key=jax.random.PRNGKey(1)
        ).state_ids
        == draws.state_ids
    )

    # Without an outcome, the operation covers every measured component.
    assert set(numeric.state_names(compile_model_fixture(dynamical_model_spec))) == {
        "A",
        "B",
        "X",
        "Y",
    }


@pytest.mark.contract
def test_projected_latent_dependencies_keep_their_connected_states():
    dynamical_model_spec = make_model(["U", "X", "Y", "A"], [("U", "X"), ("U", "A"), ("X", "Y")])
    nodes = {item.name: item for item in dynamical_model_spec.constructs}
    selected = StructuralSelection(
        dynamical_model_spec.with_entities(
            edges=replace_constructs(
                dynamical_model_spec.edges,
                (nodes["U"].revised(role=Role.ENDOGENOUS, indicators=()),),
            ),
        ),
        nodes["Y"].id,
    )
    assert nodes["U"].id in selected.marginalized_construct_ids
    assert set(selected_state_ids(selected)) == {nodes[name].id for name in ("X", "Y", "A")}
