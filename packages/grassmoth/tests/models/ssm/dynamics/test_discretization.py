"""Scientific drift derivatives for component-native dynamics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from grassmoth.artifacts.identity import scientific_id
from grassmoth.models.ssm.dynamics.intervention import Intervention
from grassmoth.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from grassmoth_test_support.dynamics import (
    DenseLinear,
    hill_term,
    interaction_term,
    reference_vector_field,
)

pytestmark = pytest.mark.inference(concern="simulation")


def _drift_derivatives(field, state, args):
    derivative = jax.jacfwd(lambda x: field(jnp.array(0.0), x, args))(state)
    return derivative, field(jnp.array(0.0), state, args) - derivative @ state


def _dense_matrix_vector_field(n_latent: int) -> VectorField:
    return reference_vector_field(n_latent=n_latent, components=(DenseLinear(),))


# =============================================================================
# Per-primitive linearization checks
# =============================================================================


class TestLinearizePrimitives:
    def test_dense_linear_recovers_matrix(self):
        """``linearize`` on a single ``DenseLinear`` component must return
        ``(A, c)`` exactly — autodiff through ``A @ x + c`` reproduces ``A``."""
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        c = jnp.array([0.1, -0.2])
        vf = _dense_matrix_vector_field(n_latent=2)
        args = VectorFieldArgs(params=({"drift": A, "cint": c},), intervention=Intervention.none())
        x_lin = jnp.array([0.7, -0.4])
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        assert jnp.allclose(A_loc, A, atol=1e-6)
        assert jnp.allclose(b_loc, c, atol=1e-6)

    def test_hill_jacobian_matches_analytic(self):
        """At ``x = EC50``, ``dHill/dx = Emax · n / (4 · EC50)``. The
        Jacobian entry for the source must match this."""
        Emax, EC50, n = 2.0, 1.0, 2.0
        vf = reference_vector_field(
            n_latent=2,
            components=(hill_term(source=0, target=1).build(),),
        )
        params: tuple[dict[str, jax.Array], ...] = (
            {
                scientific_id("parameter", "emax"): jnp.asarray(Emax),
                scientific_id("parameter", "ec50"): jnp.asarray(EC50),
                scientific_id("parameter", "exponent"): jnp.asarray(n),
            },
        )
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([EC50, 0.0])  # source at EC50
        A_loc, _ = _drift_derivatives(vf, x_lin, args)
        expected_slope = Emax * n / (4.0 * EC50)
        assert float(A_loc[1, 0]) == pytest.approx(expected_slope, abs=1e-4)
        # Other entries: target's effect on itself is 0 (no decay/feedback);
        # source has no self-influence either.
        assert float(A_loc[0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert float(A_loc[1, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_multiplicative_jacobian_off_diagonals(self):
        """For ``f(η) = w · η_a · η_b`` at ``(a₀, b₀)``: ``∂f/∂η_a = w · b₀``,
        ``∂f/∂η_b = w · a₀``."""
        w = 0.5
        a0, b0 = 3.0, 4.0
        vf = reference_vector_field(
            n_latent=3,
            components=(interaction_term(source_a=0, source_b=1, target=2).build(),),
        )
        params: tuple[dict[str, jax.Array], ...] = (
            {scientific_id("parameter", "weight"): jnp.asarray(w)},
        )
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([a0, b0, 0.0])
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        assert float(A_loc[2, 0]) == pytest.approx(w * b0, abs=1e-6)
        assert float(A_loc[2, 1]) == pytest.approx(w * a0, abs=1e-6)
        # Intercept: f(x_lin) - A · x_lin = w·a·b - (w·b·a + w·a·b) = -w·a·b
        f_at_x = w * a0 * b0
        expected_b = jnp.array([0.0, 0.0, f_at_x - (w * b0 * a0 + w * a0 * b0)])
        assert jnp.allclose(b_loc, expected_b, atol=1e-6)
