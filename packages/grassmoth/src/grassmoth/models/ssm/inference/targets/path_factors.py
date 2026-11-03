"""Exact block-local path factors cached within one fixed parameter context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import (
    prepare_grouped_noise,
    prepare_mean_density_with_noise,
)

if TYPE_CHECKING:
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.gaussian_noise import GaussianNoiseFactors
    from grassmoth.models.ssm.inference.targets.particle import (
        ParticleContext,
        ParticleNoiseFactors,
        ParticleTarget,
    )


class PathFactors(NamedTuple):
    """Exact states, responses, entering window memory and per-time density factors."""

    path: jax.Array
    responses: jax.Array
    memory: jax.Array
    transitions: jax.Array
    observations: jax.Array

    def difference(self, previous: PathFactors) -> jax.Array:
        """Cancel unchanged factors before reducing the log acceptance ratio."""
        return jnp.sum(self.transitions - previous.transitions) + jnp.sum(
            self.observations - previous.observations
        )


@dataclass(frozen=True)
class PathFactorRuntime:
    """Prepared density operators whose cache belongs to one sweep's fixed label."""

    target: ParticleTarget
    context: ParticleContext
    readings: jax.Array
    noise: ParticleNoiseFactors
    interval_noise: tuple[GaussianNoiseFactors, ...]

    @property
    def reach(self) -> int:
        """Maximum distance a changed state can influence a later density factor."""
        if self.target.coupled_recurrences:
            return self.target.times.size
        intervals = self.target.intervals
        return 1 if intervals is None else intervals.memory.max_replay

    def _responses(self, path: jax.Array, indices: jax.Array) -> jax.Array:
        """Evaluate only responses at the changed state coordinates."""
        intervals = self.target.intervals
        if intervals is None:
            return jnp.zeros((indices.size, 0), dtype=path.dtype)
        return jax.vmap(lambda state, index: intervals.response(self.context, state, index))(
            path[indices], indices
        )

    def _densities(
        self, path: jax.Array, responses: jax.Array, memory: jax.Array, indices: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Score exact transition and joint reading laws on the affected time range."""
        target, context = self.target, self.context
        initial = target.initial_log_prob(context, path[0])
        if target.times.size == 1:
            transitions = jnp.broadcast_to(initial, indices.shape)
        else:
            transitions = jax.vmap(
                lambda index: target.transition_log_prob_with_noise(
                    context,
                    self.noise,
                    path[jnp.maximum(index - 1, 0)],
                    path[index],
                    jnp.maximum(index, 1),
                )
            )(indices)
            transitions = jnp.where(indices == 0, initial, transitions)
        observations = jax.vmap(
            lambda index: target.observation_increment_with_noise(
                context, self.noise, path[index], index
            )
        )(indices)
        intervals = target.intervals
        if intervals is not None:
            observation = cast(
                "HeterogeneousObservation", target.dynamical_model(context).observation_model
            )
            groups = compile_law_groups(observation.laws, intervals.memory.channels, interval=True)

            def window(index: jax.Array) -> jax.Array:
                means, emitted = intervals.means_from_response(
                    memory[index], responses[index], index
                )
                density = prepare_mean_density_with_noise(
                    groups,
                    jnp.nan_to_num(self.readings[index]),
                    observation.measurement.manifest_cov,
                    emitted & jnp.isfinite(self.readings[index]),
                    tuple(factor.at(index) for factor in self.interval_noise),
                )
                return density(means)

            observations = observations + jax.vmap(window)(indices)
        return transitions, observations

    def initialize(self, path: jax.Array) -> PathFactors:
        """Build a cache for a fully reconstructed reference path."""
        indices = jnp.arange(path.shape[0])
        responses = self._responses(path, indices)
        intervals = self.target.intervals
        memory = (
            jnp.zeros((path.shape[0], 0), path.dtype)
            if intervals is None
            else intervals.memory_segment(
                responses, jnp.zeros(intervals.memory.shape, path.dtype), indices
            )
        )
        transitions, observations = self._densities(path, responses, memory, indices)
        return PathFactors(path, responses, memory, transitions, observations)

    def update(
        self, previous: PathFactors, path: jax.Array, start: jax.Array, length: int
    ) -> PathFactors:
        """Replace a block and only its downstream factors, retaining the exact prefix.

        A genuine recurrent dependence reaches the whole suffix. Otherwise window
        stencils bound the affected region; output-only recurrences are reconstructed
        on the supplied path but cannot alter responses or density factors outside it.
        """
        steps = path.shape[0]
        begin = jnp.maximum(start, 0)
        width = steps if self.target.coupled_recurrences else min(steps, length)
        raw = begin + jnp.arange(width)
        indices = jnp.minimum(raw, steps - 1)
        changed = raw < (steps if self.target.coupled_recurrences else start + length)
        writes = jnp.where(changed & (raw < steps), raw, steps)
        responses = previous.responses.at[writes].set(self._responses(path, indices), mode="drop")
        raw = begin + jnp.arange(min(steps, length + self.reach))
        indices = jnp.minimum(raw, steps - 1)
        writes = jnp.where((raw < steps) & (raw < start + length + self.reach), raw, steps)
        intervals = self.target.intervals
        memory = previous.memory
        if intervals is not None:
            segment = intervals.memory_segment(responses[indices], memory[begin], indices)
            memory = memory.at[writes].set(segment, mode="drop")
        transitions, observations = self._densities(path, responses, memory, indices)
        return PathFactors(
            path,
            responses,
            memory,
            previous.transitions.at[writes].set(transitions, mode="drop"),
            previous.observations.at[writes].set(observations, mode="drop"),
        )


def prepare_path_factors(
    target: ParticleTarget, context: ParticleContext, readings: jax.Array
) -> PathFactorRuntime:
    """Prepare parameter-dependent noise once before the block updates."""
    intervals = target.intervals
    interval_noise = ()
    if intervals is not None:
        observation = cast(
            "HeterogeneousObservation", target.dynamical_model(context).observation_model
        )
        channels = jnp.asarray(intervals.memory.channels, dtype=jnp.int32)
        emitted = (
            jnp.zeros_like(readings, dtype=bool)
            .at[:, channels]
            .set(jnp.asarray(intervals.memory.emission_slots) >= 0)
        )
        interval_noise = prepare_grouped_noise(
            compile_law_groups(observation.laws, intervals.memory.channels, interval=True),
            observation.measurement.manifest_cov,
            jnp.where(emitted, readings, jnp.nan),
        )
    return PathFactorRuntime(
        target, context, readings, target.prepare_noise_factors(context, readings), interval_noise
    )
