"""Test-only steady-state references for intervention and simulation checks.

Generalises the closed-form ``-A⁻¹c`` to any vector field by solving the
nonlinear root ``f(0, η*, args) = 0``. For linear vector fields the root
is unique and the solver converges in a couple of Newton steps from any
reasonable initial guess.
"""

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs


def compute_steady_state(
    vector_field: VectorField,
    params: tuple[dict[str, Array], ...],
    intervention: Intervention,
    initial_guess: Array | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 256,
) -> Array:
    """Find ``η*`` such that ``f(0, η*, args) = 0``.

    For a stable linear vector field, this reproduces
    ``-A⁻¹c`` (up to solver tolerance) and naturally extends to
    interventions, which simply alter the equation system whose root we
    seek.
    """
    args = VectorFieldArgs(params=params, intervention=intervention)
    if initial_guess is None:
        initial_guess = jnp.zeros(vector_field.n_latent)

    initial_guess = vector_field.initial_condition(initial_guess, args)

    def residual(eta: Array, residual_args: VectorFieldArgs) -> Array:
        value = vector_field(jnp.asarray(0.0), eta, residual_args)
        for override in residual_args.intervention.variable_overrides():
            target = override.value_fn(jnp.asarray(0.0))
            value = value.at[override.index].set(eta[override.index] - target)
        return value

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    solution = optx.root_find(
        residual,
        solver,
        initial_guess,
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    return solution.value
