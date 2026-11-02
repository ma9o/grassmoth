"""Exact interval emissions and predecessor-only window memory for one parameter label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.execution.observation_model import compile_observation_model
from grassmoth.models.ssm.execution.observation_operator import (
    ObservationOperator,
    expected_observation_mean,
    trajectory_observation_log_probs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import dynestyx as dsx

    from grassmoth.models.ssm.compile.observation_windows import CompiledWindowMemory
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.observation_operator import ObservationGradient
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext


@dataclass(frozen=True)
class ParticleIntervalObservations:
    """One observation contract shared by path densities, window memory and linearization.

    ``memory`` owns the window schedule. Its accumulators depend only on earlier
    states, so particles, Gaussian proposals and the exact trajectory operator
    all evaluate the same window values.
    """

    dynamical_model: Callable[[ParticleContext], dsx.DynamicalModel]
    operator: ObservationOperator
    memory: CompiledWindowMemory
    state_projection: Callable[[ParticleContext, jax.Array, jax.Array | int], jax.Array]

    def response(
        self, context: ParticleContext, state: jax.Array, index: jax.Array | int
    ) -> jax.Array:
        """Evaluate the authored response before temporal aggregation or noise."""
        observation = cast(
            "HeterogeneousObservation", self.dynamical_model(context).observation_model
        )
        return observation(self.state_projection(context, state, index), None, 0).response

    def _moments(self, response: jax.Array) -> jax.Array:
        """Response moments of the windowed channels, with axes channel and moment."""
        windowed = response[jnp.asarray(self.memory.channels, dtype=jnp.int32)]
        return jnp.stack(
            (windowed, windowed**2) if self.memory.statistic_count == 2 else (windowed,), axis=-1
        )

    def advance(
        self,
        context: ParticleContext,
        memory: jax.Array,
        previous: jax.Array,
        index: jax.Array | int,
    ) -> jax.Array:
        """Carry memory from ``index - 1`` to ``index`` using only the earlier state."""
        moments = self._moments(self.response(context, previous, jnp.asarray(index) - 1))
        moments = moments[:, None, :]
        current = jnp.asarray(self.memory.current, dtype=memory.dtype)[jnp.asarray(index) - 1]
        following = jnp.asarray(self.memory.previous, dtype=memory.dtype)[index]
        reset = jnp.asarray(self.memory.reset)[jnp.asarray(index) - 1]
        carried = jnp.where(reset[..., None], 0.0, memory + current[..., None] * moments)
        return carried + following[..., None] * moments

    def expected_means(
        self,
        context: ParticleContext,
        memory: jax.Array,
        state: jax.Array,
        index: jax.Array | int,
    ) -> tuple[jax.Array, jax.Array]:
        """Complete the windows emitted at ``index`` and map them to observation means.

        Returns full-manifest means and the mask of interval channels emitting here.
        """
        response = self.response(context, state, index)
        channels = jnp.asarray(self.memory.channels, dtype=jnp.int32)
        slots = jnp.asarray(self.memory.emission_slots)[index]
        slot = jnp.maximum(slots, 0)
        current = jnp.take_along_axis(
            jnp.asarray(self.memory.current, dtype=response.dtype)[index], slot[:, None], axis=1
        )[:, 0]
        selected = jnp.take_along_axis(memory, slot[:, None, None], axis=1)[:, 0, :]
        statistics = selected + current[:, None] * self._moments(response)
        sums = jnp.zeros_like(response).at[channels].set(statistics[:, 0])
        squares = jnp.zeros_like(response)
        if self.memory.statistic_count == 2:
            squares = squares.at[channels].set(statistics[:, 1])
        weights = (
            jnp.ones_like(response)
            .at[channels]
            .set(jnp.asarray(self.memory.emission_weights, dtype=response.dtype)[index])
        )
        means = expected_observation_mean(
            response, sums, squares, weights, self.operator.summary_operator_codes
        )
        # The canonical trajectory operator emits its origin row on the response scale.
        means = jnp.where(jnp.asarray(index) == 0, response, means)
        emitted = jnp.zeros_like(response, dtype=bool).at[channels].set(slots >= 0)
        return means, emitted

    def memory_path(self, context: ParticleContext, path: jax.Array) -> jax.Array:
        """Window memory along a path, with axes time, channel, slot and moment."""
        initial = jnp.zeros(self.memory.shape, dtype=path.dtype)

        def step(
            previous: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            state, index = inputs
            current = self.advance(context, previous, state, index)
            return current, current

        _, tail = jax.lax.scan(step, initial, (path[:-1], jnp.arange(1, path.shape[0])))
        return jnp.concatenate((initial[None], tail))

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
