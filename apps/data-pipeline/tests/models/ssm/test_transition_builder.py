"""Constant vector-field derivatives match their affine form."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx import (
    AffineDrift,
    StochasticContinuousTimeStateEvolution,
)

from nof1_causal_lab.models.ssm.dynamics.vector_field import (
    StructuralDrift,
    VectorField,
)
from nof1_causal_lab.models.ssm.execution.dynamical_model import (
    continuous_state_evolution,
)
from tests.dynamics_fixtures import (
    DenseLinear,
    DiagonalDecay,
    Intercept,
    LinearEdge,
    StateDecay,
    StateIntercept,
)


def _structural_drift(dynamics) -> StructuralDrift:
    assert isinstance(dynamics.drift, StructuralDrift)
    return dynamics.drift


def _expected_affine_drift():
    return AffineDrift(
        A=jnp.array([[-0.50, 0.05, 0.0], [-0.10, -0.55, 0.02], [0.0, 0.0, -0.65]]),
        b=jnp.array([0.10, -0.04, 0.02]),
    )


def _constant_runtime_dynamics() -> StochasticContinuousTimeStateEvolution:
    drift = jnp.array(
        [
            [-0.30, 0.05, 0.00],
            [0.00, -0.40, 0.02],
            [0.00, 0.00, -0.50],
        ],
        dtype=jnp.float32,
    )
    cint = jnp.array([0.03, -0.04, 0.02], dtype=jnp.float32)
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=3,
            components=(
                DenseLinear(),
                DiagonalDecay(),
                StateDecay(target=2),
                Intercept(),
                StateIntercept(target=0),
                LinearEdge(source=0, target=1),
            ),
        ),
        vf_params=(
            {"drift": drift},
            {"decay": jnp.array([0.20, 0.15, 0.10], dtype=jnp.float32)},
            {"decay": jnp.array(0.05, dtype=jnp.float32)},
            {"cint": cint},
            {"cint": jnp.array(0.07, dtype=jnp.float32)},
            {"weight": jnp.array(-0.10, dtype=jnp.float32)},
        ),
        diffusion=jnp.linalg.cholesky(jnp.diag(jnp.array([0.08, 0.06, 0.04], dtype=jnp.float32))),
    )


@pytest.mark.inference(concern="simulation")
def test_constant_vector_field_local_linearization_matches_affine_view():
    dynamics = _constant_runtime_dynamics()
    affine = _expected_affine_drift()
    x_lin = jnp.array([0.70, -0.20, 0.40], dtype=jnp.float32)

    args = _structural_drift(dynamics).args
    field = _structural_drift(dynamics).vector_field
    drift = jax.jacfwd(lambda x: field(jnp.array(0.0), x, args))(x_lin)
    cint = field(jnp.array(0.0), x_lin, args) - drift @ x_lin

    np.testing.assert_allclose(drift, affine.A, rtol=1e-6, atol=1e-6)
    assert affine.b is not None
    np.testing.assert_allclose(cint, affine.b, rtol=1e-6, atol=1e-6)
