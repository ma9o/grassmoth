"""Exact sequential emissions and bounded window memory for guided particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import prepare_mean_density
from grassmoth.models.ssm.execution.observation_operator import expected_observation_mean

if TYPE_CHECKING:
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations
    from grassmoth.models.ssm.inference.targets.particle import (
        ParticleContext,
        ParticleNoiseFactors,
        ParticleTarget,
    )


@dataclass(frozen=True)
class WindowMemoryRuntime:
    """Fixed window schedule, with exact label-specific response statistics."""

    intervals: ParticleIntervalObservations
    channels: tuple[int, ...]
    previous: jax.Array
    current: jax.Array
    reset: jax.Array
    emission_slots: jax.Array
    emission_weights: jax.Array
    replay_stops: jax.Array
    max_replay: int
    shape: tuple[int, int, int, int]

    def moments(self, contexts: ParticleContext, state: jax.Array) -> jax.Array:
        """Evaluate the authored response before any window aggregation."""
        response = jax.vmap(self.intervals.response, in_axes=(0, None))(contexts, state)
        response = response[:, jnp.asarray(self.channels, dtype=jnp.int32)]
        return jnp.stack((response, response**2) if self.shape[-1] == 2 else (response,), axis=-1)[
            :, :, None, :
        ]

    def log_probs(
        self,
        contexts: ParticleContext,
        state: jax.Array,
        statistics: jax.Array,
        observations: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Emit each recorded window once, preserving its joint channel law."""
        slots = self.emission_slots[index]
        channels = jnp.asarray(self.channels, dtype=jnp.int32)
        selected = jnp.take_along_axis(
            statistics, jnp.maximum(slots, 0)[None, :, None, None], axis=2
        )[:, :, 0, :]

        def density(context: ParticleContext, moments: jax.Array) -> jax.Array:
            model = cast(
                "HeterogeneousObservation",
                self.intervals.dynamical_model(context).observation_model,
            )
            groups = compile_law_groups(model.laws, self.channels, interval=True)
            response = self.intervals.response(context, state)
            sums = jnp.zeros_like(response).at[channels].set(moments[:, 0])
            squares = jnp.zeros_like(response)
            if self.shape[-1] == 2:
                squares = squares.at[channels].set(moments[:, 1])
            weights = jnp.ones_like(response).at[channels].set(self.emission_weights[index])
            means = expected_observation_mean(
                response, sums, squares, weights, self.intervals.operator.summary_operator_codes
            )
            # The canonical trajectory operator emits its origin row on the response scale.
            means = jnp.where(index == 0, response, means)
            mask = jnp.zeros_like(response, dtype=bool).at[channels].set(slots >= 0)
            mask = mask & jnp.isfinite(observations[index])
            score = prepare_mean_density(
                groups, jnp.nan_to_num(observations[index]), model.measurement.manifest_cov, mask
            )
            return score(means)

        return jax.vmap(density)(contexts, selected)


def _window_memory(intervals: ParticleIntervalObservations, labels: int) -> WindowMemoryRuntime:
    operator = intervals.operator
    scan = operator.scan
    channels = operator.interval_summary_indices
    slots = scan.emission_slots[:, channels]
    reset = (slots[:, :, None] >= 0) & (
        np.arange(scan.max_active_windows)[None, None, :] == slots[:, :, None]
    )
    weights = np.ones((len(slots), len(channels)))
    stops = np.minimum(np.arange(len(slots)) + 1, len(slots) - 1)
    for window in operator.window_plan.windows:
        weights[window.anchor_index, channels.index(window.manifest_index)] = window.weights.sum()
        stops[window.start_index : window.anchor_index] = np.maximum(
            stops[window.start_index : window.anchor_index], window.anchor_index
        )
    return WindowMemoryRuntime(
        intervals,
        channels,
        jnp.asarray(scan.prev_coeffs[:, channels]),
        jnp.asarray(scan.curr_coeffs[:, channels]),
        jnp.asarray(reset),
        jnp.asarray(slots),
        jnp.asarray(weights),
        jnp.asarray(stops),
        max(1, int(np.max(stops - np.arange(len(stops))))),
        (labels, len(channels), scan.max_active_windows, intervals.tree.statistic_count),
    )


@dataclass(frozen=True)
class SequentialObservationRuntime:
    """Point factors composed with optional deterministic window bookkeeping."""

    target: ParticleTarget
    contexts: ParticleContext
    noise: ParticleNoiseFactors
    windows: WindowMemoryRuntime | None

    @property
    def dimension(self) -> int:
        """Dimension of the independent state plus derived observation memory."""
        return self.target.state_space.size + (
            0 if self.windows is None else int(np.prod(self.windows.shape))
        )

    def state(self, value: jax.Array) -> jax.Array:
        """Project the model coordinates from a particle's augmented state."""
        return value[..., : self.target.state_space.size]

    def initial(self, state: jax.Array) -> jax.Array:
        """Start every window with the canonical zero statistics."""
        return jnp.concatenate((state, jnp.zeros(self.dimension - state.size, dtype=state.dtype)))

    def advance(self, previous: jax.Array, state: jax.Array, index: jax.Array) -> jax.Array:
        """Reset completed slots and integrate the next exact response segment."""
        if self.windows is None:
            return state
        window = self.windows
        statistics = previous[self.target.state_space.size :].reshape(window.shape)
        statistics = jnp.where(window.reset[index - 1][None, ..., None], 0.0, statistics)
        statistics = (
            statistics
            + window.previous[index][None, ..., None]
            * window.moments(self.contexts, self.state(previous))
            + window.current[index][None, ..., None] * window.moments(self.contexts, state)
        )
        return jnp.concatenate((state, statistics.reshape(-1)))

    def replace_current(self, value: jax.Array, state: jax.Array, index: jax.Array) -> jax.Array:
        """Vary only the current response, holding all past window contributions fixed."""
        if self.windows is None:
            return state
        window = self.windows
        statistics = value[self.target.state_space.size :].reshape(window.shape)
        difference = window.moments(self.contexts, state) - window.moments(
            self.contexts, self.state(value)
        )
        statistics = statistics + window.current[index][None, ..., None] * difference
        return jnp.concatenate((state, statistics.reshape(-1)))

    def log_probs(self, value: jax.Array, index: jax.Array) -> jax.Array:
        """Score exact point and completed-window factors under each parameter label."""
        state = self.state(value)
        point = jax.vmap(
            lambda context, noise: self.target.observation_increment_with_noise(
                context, noise, state, index, self.target.observations
            )
        )(self.contexts, self.noise)
        if self.windows is None:
            return point
        return point + self.windows.log_probs(
            self.contexts,
            state,
            value[self.target.state_space.size :].reshape(self.windows.shape),
            self.target.observations,
            index,
        )

    def path(self, states: jax.Array) -> jax.Array:
        """Construct the exact prefix memory of each supplied path coordinate."""
        first = self.initial(states[0])

        def step(previous: jax.Array, inputs: tuple[jax.Array, jax.Array]):
            state, index = inputs
            value = self.advance(previous, state, index)
            return value, value

        _, tail = jax.lax.scan(step, first, (states[1:], jnp.arange(1, len(states))))
        return jnp.concatenate((first[None], tail))


def sequential_observations(
    target: ParticleTarget, contexts: ParticleContext
) -> SequentialObservationRuntime:
    """Bind exact per-label emissions and their fixed window schedule once per sweep."""
    noise = jax.vmap(lambda context: target.prepare_noise_factors(context, target.observations))(
        contexts
    )
    labels = jax.tree.leaves(contexts)[0].shape[0]
    return SequentialObservationRuntime(
        target,
        contexts,
        noise,
        None if target.intervals is None else _window_memory(target.intervals, labels),
    )
