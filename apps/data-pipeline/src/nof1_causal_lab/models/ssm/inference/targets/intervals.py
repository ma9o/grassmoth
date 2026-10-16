"""Exact interval emissions and additive statistics for conditional particle trees."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    ObservationOperator,
    expected_observation_mean,
    trajectory_observation_log_probs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import dynestyx as dsx

    from nof1_causal_lab.models.ssm.compile.interval_tree import (
        CompiledIntervalTree,
        CompiledIntervalTreeLevel,
    )
    from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from nof1_causal_lab.models.ssm.execution.observation_operator import ObservationGradient
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleContext


class IntervalTreeLevelRuntime(NamedTuple):
    """Array-valued level plans carried by a compiled loop over tree depths."""

    left_slots: jax.Array
    right_slots: jax.Array
    emission_slots: jax.Array
    observations: jax.Array
    masks: jax.Array
    weights: jax.Array


def stack_interval_tree_levels(
    levels: tuple[CompiledIntervalTreeLevel, ...], node_count: int, slot_count: int
) -> IntervalTreeLevelRuntime:
    """Pad inactive nodes and slots, retaining the exact emitted-factor masks."""
    emission_count = max(level.emission_slots.shape[1] for level in levels)

    def pad(values: np.ndarray, width: int, fill: int) -> np.ndarray:
        padding = (
            (0, node_count - values.shape[0]),
            (0, width - values.shape[1]),
            *((0, 0) for _ in values.shape[2:]),
        )
        return np.pad(values, padding, constant_values=fill)

    def field(values: tuple[np.ndarray, ...], width: int, fill: int) -> jax.Array:
        return jnp.asarray(np.stack(tuple(pad(value, width, fill) for value in values)))

    return IntervalTreeLevelRuntime(
        field(tuple(level.left_slots for level in levels), slot_count, -1),
        field(tuple(level.right_slots for level in levels), slot_count, -1),
        field(tuple(level.emission_slots for level in levels), emission_count, 0),
        field(tuple(level.observations for level in levels), emission_count, 0),
        field(tuple(level.masks for level in levels), emission_count, 0),
        field(tuple(level.weights for level in levels), emission_count, 1),
    )


@dataclass(frozen=True)
class ParticleIntervalObservations:
    """One observation contract shared by path densities and tree stitching."""

    dynamical_model: Callable[[ParticleContext], dsx.DynamicalModel]
    operator: ObservationOperator
    tree: CompiledIntervalTree
    state_projection: Callable[[ParticleContext, jax.Array], jax.Array]
    power: jax.Array | None = None

    def select_channels(self, channels: tuple[int, ...]) -> ParticleIntervalObservations | None:
        """Retain complete selected factors while preserving their window schedules."""
        if not set(channels).intersection(self.operator.interval_summary_indices):
            return None
        selected = np.isin(np.arange(self.tree.leaf_weights.shape[-1]), channels)
        return replace(
            self,
            tree=replace(
                self.tree,
                leaf_weights=self.tree.leaf_weights * selected,
                levels=tuple(
                    replace(level, masks=level.masks & selected) for level in self.tree.levels
                ),
            ),
        )

    def response(self, context: ParticleContext, state: jax.Array) -> jax.Array:
        """Evaluate the authored response before temporal aggregation or noise."""
        observation = cast(
            "HeterogeneousObservation", self.dynamical_model(context).observation_model
        )
        return observation(self.state_projection(context, state), None, 0).response

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
        densities = trajectory_observation_log_probs(
            self.state_projection(context, path),
            observations,
            None,
            measurement.lambda_mat,
            measurement.manifest_means,
            measurement.manifest_cov,
            compiled.kernel,
            compiled.mean_log_prob_fn,
            self.operator.observation_support,
            gradient=gradient,
        )
        return (
            densities
            if self.power is None
            else jnp.where(self.power == 0, 0, self.power * densities)
        )

    def leaf_statistics(self, contexts: ParticleContext, particles: jax.Array) -> jax.Array:
        """Build sparse response statistics with axes time, particle, parameter, slot, channel, moment."""
        response = jax.vmap(
            lambda bank: jax.vmap(
                lambda state: jax.vmap(self.response, in_axes=(0, None))(contexts, state)
            )(bank)
        )(particles)
        moments = (response, response**2) if self.tree.statistic_count == 2 else (response,)
        statistics = jnp.stack(moments, axis=-1)[:, :, :, None, :, :]
        weights = jnp.asarray(self.tree.leaf_weights, dtype=particles.dtype)
        statistics = statistics * weights[: particles.shape[0], None, None, :, :, None]
        padding = [(0, weights.shape[0] - particles.shape[0]), *[(0, 0)] * 5]
        return jnp.pad(statistics, padding)

    def emission_log_probs(
        self,
        contexts: ParticleContext,
        statistics: jax.Array,
        level: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime,
        node: jax.Array,
    ) -> jax.Array:
        """Score emitted statistics for aligned particles, preserving parameter labels."""
        slots = jnp.asarray(level.emission_slots)[node]
        selected = jnp.take(statistics, slots, axis=-3)
        return self._emission_log_probs(contexts, selected, level, node)

    def pairwise_emission_log_probs(
        self,
        contexts: ParticleContext,
        left: jax.Array,
        right: jax.Array,
        level: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime,
        node: jax.Array,
    ) -> jax.Array:
        """Score all child pairs, expanding only slots emitted at this node."""
        slots = jnp.asarray(level.emission_slots)[node]
        left_emitted = gather_interval_statistics(left, jnp.asarray(level.left_slots)[node][slots])
        right_emitted = gather_interval_statistics(
            right, jnp.asarray(level.right_slots)[node][slots]
        )
        return jax.vmap(
            lambda stats: self._emission_log_probs(
                contexts, stats[None] + right_emitted, level, node
            )
        )(left_emitted)

    def _emission_log_probs(
        self,
        contexts: ParticleContext,
        statistics: jax.Array,
        level: CompiledIntervalTreeLevel | IntervalTreeLevelRuntime,
        node: jax.Array,
    ) -> jax.Array:
        """Evaluate already selected emissions with their joint channel laws."""
        values = jnp.asarray(level.observations, dtype=statistics.dtype)[node]
        masks = jnp.asarray(level.masks, dtype=statistics.dtype)[node]
        weights = jnp.asarray(level.weights, dtype=statistics.dtype)[node]

        def density(context: ParticleContext, stats: jax.Array) -> jax.Array:
            observation = cast(
                "HeterogeneousObservation", self.dynamical_model(context).observation_model
            )
            compiled = compile_observation_model(
                observation.laws,
                manifest_cov=observation.measurement.manifest_cov,
                observation_support=self.operator.observation_support,
            )
            mean_density = compiled.mean_log_prob_fn
            assert mean_density is not None  # This target owns interval support.
            means = expected_observation_mean(
                jnp.zeros_like(stats[..., 0]),
                stats[..., 0],
                stats[..., 1] if self.tree.statistic_count == 2 else jnp.zeros_like(stats[..., 0]),
                weights,
                self.operator.summary_operator_codes,
            )
            return jnp.sum(
                jax.vmap(
                    lambda y, mean, mask: mean_density(
                        y, mean, observation.measurement.manifest_cov, mask
                    )
                )(values, means, masks)
            )

        densities = jax.vmap(jax.vmap(density, in_axes=(0, 0)), in_axes=(None, 0))(
            contexts, statistics
        )
        return (
            densities
            if self.power is None
            else jnp.where(self.power == 0, 0, self.power * densities)
        )


def gather_interval_statistics(statistics: jax.Array, slots: jax.Array) -> jax.Array:
    """Gather a child frontier, using its zero slot for windows absent in that child."""
    zeros = jnp.zeros((*statistics.shape[:-3], 1, *statistics.shape[-2:]), dtype=statistics.dtype)
    return jnp.take(jnp.concatenate([statistics, zeros], axis=-3), slots, axis=-3)
