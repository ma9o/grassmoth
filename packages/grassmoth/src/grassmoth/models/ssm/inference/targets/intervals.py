"""Exact interval emissions for conditional particle paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp

from grassmoth.models.ssm.execution.observation_model import compile_observation_model
from grassmoth.models.ssm.execution.observation_operator import (
    ObservationOperator,
    trajectory_observation_log_probs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import dynestyx as dsx
    import jax

    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.observation_operator import ObservationGradient
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext


@dataclass(frozen=True)
class ParticleIntervalObservations:
    """One observation contract shared by path densities and window memory.

    ``statistic_count`` is the number of response moments each window carries.
    """

    dynamical_model: Callable[[ParticleContext], dsx.DynamicalModel]
    operator: ObservationOperator
    statistic_count: int
    state_projection: Callable[[ParticleContext, jax.Array, jax.Array | int], jax.Array]

    def response(
        self, context: ParticleContext, state: jax.Array, index: jax.Array | int
    ) -> jax.Array:
        """Evaluate the authored response before temporal aggregation or noise."""
        observation = cast(
            "HeterogeneousObservation", self.dynamical_model(context).observation_model
        )
        return observation(self.state_projection(context, state, index), None, 0).response

    def trajectory_log_probs(
        self,
        context: ParticleContext,
        path: jax.Array,
        observations: jax.Array,
        *,
        gradient: ObservationGradient = "full",
    ) -> jax.Array:
        """Report each exact emission at its recorded anchor, including missing rows."""
        observation = cast(
            "HeterogeneousObservation", self.dynamical_model(context).observation_model
        )
        measurement = observation.measurement
        compiled = compile_observation_model(
            observation.laws,
            manifest_cov=measurement.manifest_cov,
            observation_support=self.operator.observation_support,
        )
        return trajectory_observation_log_probs(
            self.state_projection(context, path, jnp.arange(path.shape[0])),
            observations,
            measurement,
            compiled,
            gradient=gradient,
        )
