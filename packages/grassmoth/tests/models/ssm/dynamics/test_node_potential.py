"""Native node potentials preserve nonlinear drift, metadata, and causal interventions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.models.ssm.dynamics.intervention import (
    ConstantValueFn,
    Intervention,
    VariableOverride,
)
from grassmoth.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
from grassmoth.models.ssm.dynamics.vector_field import VectorFieldArgs
from grassmoth.models.ssm.execution.dynamical_model import continuous_state_evolution
from grassmoth.models.structure import StructuralSelection, selected_edges
from tests.support.dynamics import linear_term, potential_term
from tests.support.model import load_model_fixture


def _directed_edges_cannot_be_reinterpreted_as_potentials_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "node_potential/directed_edges_cannot_be_reinterpreted_as_potentials_model_fixture.json"
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize("quartic", [0.0, 0.5])
def test_native_potential_matches_restoring_force_and_its_derivatives(quartic):
    compiled = compile_dynamics(
        DynamicsSpec(
            2,
            (
                potential_term(
                    1,
                    center=0.5,
                    stiffness=1.2,
                    quartic=quartic,
                ),
            ),
        )
    )
    args = VectorFieldArgs(({},), Intervention.none())
    evolution = compiled.vector_field.evolution(args)
    assert evolution.potential is not None
    assert evolution.use_negative_gradient
    x = jnp.array([3.0, 2.0])

    def force(value):
        return evolution.total_drift(x=value, u=None, t=jnp.array(0.0))

    displacement = float(x[1]) - 0.5
    np.testing.assert_allclose(force(x), [0.0, -1.2 * displacement - quartic * displacement**3])
    np.testing.assert_allclose(
        jax.jacfwd(force)(x), [[0.0, 0.0], [0.0, -1.2 - 3 * quartic * displacement**2]]
    )
    np.testing.assert_allclose(force(jnp.array([3.0, 0.5])), [0.0, 0.0], atol=1e-7)


@pytest.mark.inference(concern="simulation")
def test_clamp_removes_potential_input_forcing_and_process_noise():
    compiled = compile_dynamics(
        DynamicsSpec(
            2,
            (
                potential_term(
                    0,
                    center=0,
                    stiffness=1,
                    quartic=0.2,
                ),
                linear_term(0, 1, weight=2),
            ),
        )
    )
    intervention = Intervention((VariableOverride(0, ConstantValueFn(jnp.array(3.0))),))
    evolution = continuous_state_evolution(
        compiled.vector_field,
        ({}, {}),
        jnp.eye(2),
        intervention=intervention,
    )
    np.testing.assert_allclose(
        evolution.total_drift(jnp.array([3.0, 0.0]), None, jnp.array(0.0)), [0.0, 6.0]
    )
    np.testing.assert_allclose(
        evolution.diffusion.as_matrix(x=jnp.ones(2), u=None, t=0.0, state_dim=2),
        [[0.0, 0.0], [0.0, 1.0]],
    )


@pytest.mark.contract
def test_potential_coefficients_keep_their_scientific_meanings():
    compiled = compile_dynamics(DynamicsSpec(1, (potential_term(0, quartic=None),)))
    assert {site.site_kind for site in compiled.site_registry} == {
        SiteKind.DYNAMICS_POTENTIAL_CENTER,
        SiteKind.DYNAMICS_DECAY,
        SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
    }
    assert all(site.positions == (0,) for site in compiled.site_registry)


@pytest.mark.contract
@pytest.mark.parametrize("kwargs", [{"stiffness": 0}, {"quartic": -1}])
def test_invalid_potential_coefficients_are_rejected(kwargs):
    with pytest.raises(ValueError, match=r"(positive|non-negative)"):
        potential_term(0, **kwargs)


@pytest.mark.contract
def test_directed_edges_cannot_be_reinterpreted_as_potentials():
    dynamical_model_spec = _directed_edges_cannot_be_reinterpreted_as_potentials_model_fixture()
    edge = selected_edges(StructuralSelection(dynamical_model_spec, None))[0]
    with pytest.raises(ValueError, match="literal_error"):
        edge.revised(
            mechanisms=tuple(
                {"id": term.id, "kind": "potential", "expression": term.expression}
                for term in edge.mechanisms
            )
        )
