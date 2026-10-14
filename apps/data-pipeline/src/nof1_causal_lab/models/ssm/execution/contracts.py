"""Measurement parameters, observation laws, and NumPyro structural tracing."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from dynestyx import StochasticContinuousTimeStateEvolution
    from numpyro.distributions import MultivariateNormal

    from nof1_causal_lab.artifacts.likelihood import Law
    from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression
    from nof1_causal_lab.models.ssm.shapes import Array, Float

NUMERICAL_EPSILON = 1e-10

type ObservationLaws = tuple[Law[BoundExpression], ...]


class MeasurementParams(NamedTuple):
    """Measurement mapping before the family-specific link and observation law.

    ``lambda_mat @ state + manifest_means`` is the linear predictor.
    ``manifest_cov`` supplies Gaussian covariance or the configured family's
    dispersion inputs; the observation model determines the actual density.
    """

    lambda_mat: Float[Array, "M D"]
    manifest_means: Float[Array, " M"]
    manifest_cov: Float[Array, "M M"]  # Σ_R


class InitializationLikelihoodBackend(Protocol):
    """Likelihood placeholder used when tracing structural parameter sites."""

    def compute_log_likelihood(
        self,
        dynamics: StochasticContinuousTimeStateEvolution,
        _measurement_params: MeasurementParams,
        initial_state: MultivariateNormal,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        /,
        *,
        observation_laws: ObservationLaws,
    ) -> jnp.ndarray:
        """Return the cumulative likelihood contributions for the parameter trace."""
        ...


# Preserve immutable mappings when JAX flattens/rebuilds published numerical values.
jax.tree_util.register_pytree_node(
    MappingProxyType,
    lambda value: (tuple(value.values()), tuple(value.keys())),
    lambda keys, values: MappingProxyType(dict(zip(keys, values, strict=True))),
)
