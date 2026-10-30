"""Exact label-specific recurrence and window memory for guided particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import (
    prepare_grouped_noise,
    prepare_mean_density_with_noise,
)
from grassmoth.models.ssm.execution.observation_operator import expected_observation_mean

if TYPE_CHECKING:
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.gaussian_noise import GaussianNoiseFactors
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
    noise: tuple[GaussianNoiseFactors, ...]

    def moments(self, contexts: ParticleContext, states: jax.Array) -> jax.Array:
        """Evaluate the authored response before any window aggregation."""
        response = jax.vmap(self.intervals.response)(contexts, states)
        response = response[:, jnp.asarray(self.channels, dtype=jnp.int32)]
        return jnp.stack((response, response**2) if self.shape[-1] == 2 else (response,), axis=-1)[
            :, :, None, :
        ]

    def log_probs(
        self,
        contexts: ParticleContext,
        states: jax.Array,
        statistics: jax.Array,
        observations: jax.Array,
        index: jax.Array,
        log_precision: jax.Array | None = None,
    ) -> jax.Array:
        """Emit each recorded window once, preserving its joint channel law."""
        slots = self.emission_slots[index]
        channels = jnp.asarray(self.channels, dtype=jnp.int32)
        selected = jnp.take_along_axis(
            statistics, jnp.maximum(slots, 0)[None, :, None, None], axis=2
        )[:, :, 0, :]

        def density(
            context: ParticleContext,
            state: jax.Array,
            moments: jax.Array,
            noise: tuple[GaussianNoiseFactors, ...],
        ) -> jax.Array:
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
            score = prepare_mean_density_with_noise(
                groups,
                jnp.nan_to_num(observations[index]),
                model.measurement.manifest_cov,
                mask,
                tuple(factor.at(index) for factor in noise),
                log_precision,
            )
            return score(means)

        return jax.vmap(density)(contexts, states, selected, self.noise)


def _window_memory(
    intervals: ParticleIntervalObservations,
    contexts: ParticleContext,
    observations: jax.Array,
) -> WindowMemoryRuntime:
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
    observed = jnp.zeros_like(observations, dtype=bool).at[:, jnp.asarray(channels)].set(slots >= 0)
    observed = observed & jnp.isfinite(observations)
    readings = jnp.where(observed, observations, jnp.nan)

    def noise(context: ParticleContext) -> tuple[GaussianNoiseFactors, ...]:
        model = cast(
            "HeterogeneousObservation", intervals.dynamical_model(context).observation_model
        )
        return prepare_grouped_noise(
            compile_law_groups(model.laws, channels, interval=True),
            model.measurement.manifest_cov,
            readings,
        )

    labels = jax.tree.leaves(contexts)[0].shape[0]
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
        jax.vmap(noise)(contexts),
    )


@dataclass(frozen=True)
class SequentialObservationRuntime:
    """Independent coordinates composed with exact recurrence and window bookkeeping."""

    target: ParticleTarget
    contexts: ParticleContext
    guide: ParticleContext
    noise: ParticleNoiseFactors
    windows: WindowMemoryRuntime | None
    log_precisions: jax.Array | None = None

    @property
    def replay_shape(self) -> tuple[int, int]:
        """One reconstructed coordinate block per parameter label and shared guide."""
        return self.contexts[1].shape[0] + 1, len(self.target.state_space.replayed_coordinates)

    @property
    def model_dimension(self) -> int:
        """Independent coordinates plus reconstructed model memory."""
        return self.target.state_space.size + int(np.prod(self.replay_shape))

    @property
    def dimension(self) -> int:
        """Dimension of the independent state plus derived observation memory."""
        return self.model_dimension + (
            0 if self.windows is None else int(np.prod(self.windows.shape))
        )

    def state(self, value: jax.Array) -> jax.Array:
        """Project the coordinates whose independent entries are sampled."""
        return value[..., : self.target.state_space.size]

    @property
    def replay_contexts(self) -> ParticleContext:
        """Bind the same independent path under every label and the shared guide."""
        return cast(
            "ParticleContext",
            jax.tree.map(
                lambda labels, guide: jnp.concatenate((labels, guide[None])),
                self.contexts,
                self.guide,
            ),
        )

    def replayed_states(self, value: jax.Array) -> jax.Array:
        """Recover exact model states for the labels followed by the guide."""
        states = jnp.broadcast_to(
            self.state(value), (self.replay_shape[0], self.target.state_space.size)
        )
        indices = jnp.asarray(self.target.state_space.replayed_coordinates, dtype=jnp.int32)
        return states.at[:, indices].set(
            value[self.target.state_space.size : self.model_dimension].reshape(self.replay_shape)
        )

    def guide_moments(self, value: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Evaluate the exact proposal drift on the guide's own reconstructed prefix."""
        return self.target.transition_moments(self.guide, self.replayed_states(value)[-1], index)

    def initial_log_probs(self, value: jax.Array) -> jax.Array:
        """Score each label on its exact initial state."""
        return jnp.asarray(
            jax.vmap(self.target.initial_log_prob)(self.contexts, self.replayed_states(value)[:-1])
        )

    def transition_log_probs(
        self, previous: jax.Array, current: jax.Array, index: jax.Array
    ) -> jax.Array:
        """Score stochastic densities with each label's deterministic prefix held fixed."""
        return jax.vmap(self.target.transition_log_prob_with_noise, in_axes=(0, 0, 0, 0, None))(
            self.contexts,
            self.noise,
            self.replayed_states(previous)[:-1],
            self.replayed_states(current)[:-1],
            index,
        )

    def transition_means(self, previous: jax.Array, index: jax.Array) -> jax.Array:
        """Prepare each label's nonlinear mean on its exact reconstructed predecessor."""
        return jax.vmap(self.target.prepare_transition_mean, in_axes=(0, 0, None))(
            self.contexts, self.replayed_states(previous)[:-1], index
        )

    def transition_log_probs_with_means(
        self,
        previous: jax.Array,
        current: jax.Array,
        index: jax.Array,
        means: jax.Array,
    ) -> jax.Array:
        """Score a selected successor without reevaluating the fixed predecessor means."""
        return jax.vmap(self.target.transition_log_prob_with_mean, in_axes=(0, 0, 0, 0, None, 0))(
            self.contexts,
            self.noise,
            self.replayed_states(previous)[:-1],
            self.replayed_states(current)[:-1],
            index,
            means,
        )

    def initial(self, state: jax.Array) -> jax.Array:
        """Reconstruct each origin and start windows with the canonical zero statistics."""
        indices = jnp.asarray(self.target.state_space.replayed_coordinates, dtype=jnp.int32)
        replayed = jax.vmap(self.target.replay_initial, in_axes=(0, None))(
            self.replay_contexts, state
        )[:, indices]
        return jnp.concatenate(
            (
                state,
                replayed.reshape(-1),
                jnp.zeros(self.dimension - self.model_dimension, dtype=state.dtype),
            )
        )

    def advance(self, previous: jax.Array, state: jax.Array, index: jax.Array) -> jax.Array:
        """Reset completed slots and integrate the next exact response segment."""
        if self.target.state_space.requires_replay:
            indices = jnp.asarray(self.target.state_space.replayed_coordinates, dtype=jnp.int32)
            replayed = jax.vmap(self.target.replay_step, in_axes=(0, 0, None, None))(
                self.replay_contexts, self.replayed_states(previous), state, index
            )[:, indices]
            current = jnp.concatenate((state, replayed.reshape(-1)))
        else:
            current = state
        if self.windows is None:
            return current
        window = self.windows
        statistics = previous[self.model_dimension :].reshape(window.shape)
        statistics = jnp.where(window.reset[index - 1][None, ..., None], 0.0, statistics)
        statistics = (
            statistics
            + window.previous[index][None, ..., None]
            * window.moments(self.contexts, self.replayed_states(previous)[:-1])
            + window.current[index][None, ..., None]
            * window.moments(self.contexts, self.replayed_states(current)[:-1])
        )
        return jnp.concatenate((current, statistics.reshape(-1)))

    def replace_current(self, value: jax.Array, state: jax.Array, index: jax.Array) -> jax.Array:
        """Vary only the current response, holding all past window contributions fixed."""
        indices = jnp.asarray(self.target.state_space.replayed_coordinates, dtype=jnp.int32)
        replayed = value[self.target.state_space.size : self.model_dimension].reshape(
            self.replay_shape
        )
        replayed = jnp.where(self.target.independent_mask[index, indices], state[indices], replayed)
        current = jnp.concatenate((state, replayed.reshape(-1)))
        if self.windows is None:
            return current
        window = self.windows
        statistics = value[self.model_dimension :].reshape(window.shape)
        difference = window.moments(
            self.contexts, self.replayed_states(current)[:-1]
        ) - window.moments(self.contexts, self.replayed_states(value)[:-1])
        statistics = statistics + window.current[index][None, ..., None] * difference
        return jnp.concatenate((current, statistics.reshape(-1)))

    def log_probs(self, value: jax.Array, index: jax.Array) -> jax.Array:
        """Score exact point and completed-window factors under each parameter label."""
        states = self.replayed_states(value)[:-1]
        point = jax.vmap(
            lambda context, noise, state: self.target.observation_increment_with_noise(
                context,
                noise,
                state,
                index,
                None if self.log_precisions is None else self.log_precisions[index],
            )
        )(self.contexts, self.noise, states)
        if self.windows is None:
            return point
        return point + self.windows.log_probs(
            self.contexts,
            states,
            value[self.model_dimension :].reshape(self.windows.shape),
            self.target.observations,
            index,
            None if self.log_precisions is None else self.log_precisions[index],
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

    def path_log_probs(self, states: jax.Array) -> jax.Array:
        """Rescore a complete path with this runtime's exact conditional factors."""
        values = self.path(states)
        return (
            self.initial_log_probs(values[0])
            + jax.vmap(self.transition_log_probs)(
                values[:-1], values[1:], jnp.arange(1, len(states))
            ).sum(axis=0)
            + jax.vmap(self.log_probs)(values, jnp.arange(len(states))).sum(axis=0)
        )


def sequential_observations(
    target: ParticleTarget,
    contexts: ParticleContext,
    guide: ParticleContext,
    log_precisions: jax.Array | None = None,
) -> SequentialObservationRuntime:
    """Bind exact per-label emissions and their fixed window schedule once per sweep."""
    noise = jax.vmap(lambda context: target.prepare_noise_factors(context, target.observations))(
        contexts
    )
    return SequentialObservationRuntime(
        target,
        contexts,
        guide,
        noise,
        None
        if target.intervals is None
        else _window_memory(target.intervals, contexts, target.observations),
        log_precisions,
    )
