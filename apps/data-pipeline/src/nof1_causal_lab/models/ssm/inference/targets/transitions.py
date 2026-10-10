"""Initialization-only Gaussian views of the declared Dynestyx state evolution."""

from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.models.ssm.covariance_utils import CHOL_JITTER, symmetrize_with_jitter
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
from nof1_causal_lab.models.ssm.shapes import Array, Float

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.vector_field import StructuralDrift


def build_discrete_transitions(
    dynamics: dsx.StochasticContinuousTimeStateEvolution,
    time_intervals: Float[Array, " T"],
    *,
    linearization_states: Array | None = None,
) -> dsx.LinearGaussianParams:
    """Ask Dynestyx for the local Gaussian model used to initialize particles.

    Affine fields use their state-independent Jacobian at zero. Nonlinear fields
    require an explicit reference trajectory. Neither view defines the production
    particle target, which uses Euler-Maruyama over the true nonlinear drift.
    """
    drift = cast("StructuralDrift", dynamics.drift)
    n_latent = drift.vector_field.n_latent
    time_intervals = jnp.asarray(time_intervals)
    shape = (time_intervals.shape[0], n_latent)
    if infer_linearisation(drift.vector_field) == "constant":
        states = jnp.zeros(shape, dtype=time_intervals.dtype)
    else:
        if linearization_states is None:
            raise ValueError(
                "Trajectory-dependent vector-field discretization requires "
                "linearization_states with one state per interval."
            )
        states = jnp.asarray(linearization_states)
        if states.shape != shape:
            raise ValueError(f"linearization_states must have shape {shape}, got {states.shape}")

    diffusion = dynamics.diffusion.as_matrix(x=None, u=None, t=0, state_dim=n_latent)
    starts = jnp.cumsum(time_intervals) - time_intervals - time_intervals[0]

    def at_interval(state: jax.Array, start: jax.Array, dt: jax.Array) -> dsx.LinearGaussianParams:
        affine_drift = dsx.linearize_drift(dynamics.total_drift, x=state, u=None, t=start)
        bias = cast("jax.Array", affine_drift.b)
        # A diagonal coordinate change keeps large accounting/log-rate couplings
        # from consuming the exponential's scaling budget purely through units.
        # The transformation is algebraically undone below; stopping its gradient
        # avoids differentiating the arbitrary choice of numerical coordinates.
        units = jax.lax.stop_gradient(
            jnp.maximum(
                1.0, dt * jnp.maximum(jnp.max(jnp.abs(affine_drift.A), axis=1), jnp.abs(bias))
            )
        )
        normalized_diffusion = diffusion / units[:, None]
        noise_scale = jax.lax.stop_gradient(
            jnp.maximum(1.0, jnp.max(jnp.abs(normalized_diffusion)))
        )
        normalized_drift = dsx.AffineDrift(
            A=affine_drift.A * units[None, :] / units[:, None], b=bias / units
        )
        affine_model = dsx.StochasticContinuousTimeStateEvolution(
            drift=normalized_drift, diffusion=dsx.FullDiffusion(normalized_diffusion / noise_scale)
        )
        evolution = cast(
            "dsx.LinearGaussianStateEvolution",
            dsx.discretize_state_evolution(
                affine_model, ExactAffineConfig(covariance_jitter=CHOL_JITTER)
            ),
        )
        params = evolution.params_at(start, start + dt)
        # Dynestyx requires a nondegenerate initialization Gaussian, including for
        # static coordinates. Remove its numerical-coordinate jitter before
        # restoring units, then apply the same jitter in the model coordinates.
        unregularized = params.cov - CHOL_JITTER * jnp.eye(n_latent, dtype=params.cov.dtype)
        covariance = unregularized * (noise_scale * units[:, None]) * (noise_scale * units[None, :])
        return params._replace(
            A=params.A * units[:, None] / units[None, :],
            bias=cast("jax.Array", params.bias) * units,
            cov=symmetrize_with_jitter(covariance, jitter=CHOL_JITTER),
        )

    return jax.vmap(at_interval)(states, starts, time_intervals)
