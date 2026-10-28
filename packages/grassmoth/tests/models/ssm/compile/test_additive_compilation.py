"""Retained scientific identity and explicit execution requirements; no numerical runs."""

from datetime import UTC, datetime

import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import hill as expr_hill
from grassmoth.artifacts.model.expressions import state as expr_state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.diagnostics.simulation import (
    ConstructSimulationTarget,
    _incoming_edge_off_target,
)
from grassmoth.models.structure import (
    StructuralCompilationError,
    StructuralSelection,
    selected_state_ids,
    validate_execution_structure,
)
from grassmoth_test_support.helpers import make_model
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import additive_a_b_model

pytestmark = pytest.mark.contract


def test_static_target_is_retained_and_reported_as_unsupported():
    draft = make_model(["A", "B"], [("A", "B")])
    static_constructs = tuple(
        node.revised(temporal_status="time_invariant") for node in draft.constructs
    )
    dynamical_model_spec = draft.with_entities(
        edges=replace_constructs(draft.edges, static_constructs)
    )
    before = dynamical_model_spec.model_dump(mode="json")
    selection = StructuralSelection(dynamical_model_spec, None)
    assert set(selected_state_ids(selection)) == {
        node.id for node in dynamical_model_spec.constructs
    }
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        validate_execution_structure(selection)
    assert dynamical_model_spec.model_dump(mode="json") == before


def test_exact_observations_preserve_state_and_parameter_identity():
    draft = additive_a_b_model()
    source = draft.edges[0].cause
    observed = source.revised(
        indicators=(
            source.indicators[0].revised(
                likelihood=LikelihoodSpec(
                    law=DeltaLawSpec(v=expr_state(source.id)),
                    standardized=False,
                    reasoning="The source is observed exactly.",
                )
            ),
        )
    )
    dynamical_model_spec = draft.with_entities(edges=replace_constructs(draft.edges, (observed,)))
    selection = StructuralSelection(dynamical_model_spec, None)
    assert source.id in selected_state_ids(selection)
    assert dynamical_model_spec.parameters == draft.parameters
    edge_parameters = dynamical_model_spec.parameters_for(dynamical_model_spec.edges[0].id)
    assert edge_parameters
    for parameter in edge_parameters:
        assert dynamical_model_spec.parameter(parameter.id) == draft.parameter(parameter.id)
        assert (
            dynamical_model_spec.parameter_context(parameter.id).quantity
            == SiteKind.DYNAMICS_WEIGHT
        )


def test_edge_off_targets_every_additive_contribution_without_running_a_simulation():
    dynamical_model_spec = additive_a_b_model()
    edge = dynamical_model_spec.edges[0]
    fixed_hill = DriftMechanismSpec(
        id="mechanism:fixed-hill-a",
        expression=expr_hill(expr_state(edge.cause.id), emax=0.4, ec50=1, n=2),
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=(
            edge.revised(
                mechanisms=(
                    *edge.mechanisms,
                    fixed_hill,
                    fixed_hill.revised(id="mechanism:fixed-hill-b"),
                )
            ),
        )
    )
    native = dynamical_model_spec
    target = dynamical_model_spec.get_construct(edge.effect.id)
    source = dynamical_model_spec.get_construct(edge.cause.id)
    contribution = ConstructSimulationTarget(
        construct=compile_model_fixture(dynamical_model_spec).states[
            selected_state_ids(StructuralSelection(dynamical_model_spec, None)).index(target.id)
        ],
        edge_parents=(source.name,),
    )
    assert numeric.state_names(compile_model_fixture(native)) is not None
    off = _incoming_edge_off_target(
        compile_model_fixture(native),
        contribution,
        numeric.state_names(compile_model_fixture(native)),
        numeric.state_names(compile_model_fixture(native)).index(target.name),
    )
    assert len(off.components) == 3


def test_predictive_edge_findings_exclude_the_response_state(monkeypatch):
    """State-dependent edge mechanisms do not invent a response-to-itself edge."""
    import jax.numpy as jnp

    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.diagnostics.reachability import CheckResult
    from grassmoth.models.ssm.diagnostics.simulation import DesignInfo
    from grassmoth.models.ssm.predictive import simulation
    from grassmoth.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory

    dynamical_model_spec = additive_a_b_model()
    edge = dynamical_model_spec.edges[0]
    mechanism = DriftMechanismSpec(
        id="mechanism:response-dependent-hill",
        expression=expr_hill(expr_state(edge.cause.id), emax=0.4, ec50=1, n=2)
        * expr_state(edge.effect.id),
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=(edge.revised(mechanisms=(*edge.mechanisms, mechanism)),)
    )
    compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
    received: list[ConstructSimulationTarget] = []

    def measured(
        _model: CompiledDynamicalModel,
        _prediction: PredictiveDraws,
        _design: DesignInfo,
        target: ConstructSimulationTarget,
        **_kwargs: object,
    ) -> list[CheckResult]:
        received.append(target)
        return [
            CheckResult.measured(
                "edge attribution",
                f"{parent}->{target.construct.name}",
                "1",
                "0",
                "Retain the declared parent edge.",
                measurements=(),
            )
            for parent in target.edge_parents
        ]

    monkeypatch.setattr(simulation, "measure_construct_simulation", measured)
    latent = jnp.zeros((1, 2, len(compiled_dynamical_model.states)))
    observed = jnp.zeros((1, 2, len(compiled_dynamical_model.observations)))
    prediction = PredictiveDraws(
        {},
        PredictiveTrajectory(
            latent, observed, observed, jnp.ones_like(observed, dtype=bool), observed
        ),
    )
    batch = simulation.SimulationBatch.from_draws(
        prediction,
        DesignInfo(jnp.array([0.0, 1.0]), (), {}, {}),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    findings, _ = simulation.measure_simulation_batch(
        compiled_dynamical_model, batch, groups=("dynamics",)
    )
    response = next(target for target in received if target.construct.id == edge.effect.id)
    assert response.edge_parents == (edge.cause.name,)
    assert response.hill_parents == (edge.cause.name,)
    assert [
        finding.subject.target.id
        for finding in findings
        if not isinstance(finding.subject.target, str)
    ] == [edge.id]
