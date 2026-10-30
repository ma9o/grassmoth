"""Exact interval emissions and additive statistics for conditional particle trees."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import (
    compile_observation_model,
    prepare_mean_density,
)
from grassmoth.models.ssm.execution.observation_operator import (
    ObservationOperator,
    expected_observation_mean,
    trajectory_observation_log_probs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import dynestyx as dsx

    from grassmoth.models.ssm.compile.interval_tree import (
        CompiledIntervalTree,
        CompiledIntervalTreeLevel,
    )
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.observation_operator import ObservationGradient
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext


class IntervalTreeLevelRuntime(NamedTuple):
    """Array-valued level plans carried by a compiled loop over tree depths."""

    left_slots: jax.Array
    right_slots: jax.Array
    emission_slots: jax.Array
    observations: jax.Array
    masks: jax.Array
    weights: jax.Array
    emitting_nodes: jax.Array
    emission_counts: jax.Array


def stack_interval_tree_levels(
    levels: tuple[CompiledIntervalTreeLevel, ...],
    node_count: int,
    slot_count: int,
    batch_size: int,
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

    # Pack complete observed rows, retaining their joint channel factors. The
    # execution loop visits only emitting nodes in its current merge batch.
    def emitting_batches(level: CompiledIntervalTreeLevel) -> tuple[np.ndarray, ...]:
        nodes = np.flatnonzero(level.masks.any(axis=(1, 2)))
        return tuple(
            nodes[(nodes >= start) & (nodes < start + batch_size)]
            for start in range(0, node_count, batch_size)
        )

    batches = tuple(emitting_batches(level) for level in levels)
    capacity = max(len(nodes) for level in batches for nodes in level)
    emitting_nodes = jnp.asarray(
        np.stack(
            tuple(
                np.stack(tuple(np.pad(nodes, (0, capacity - len(nodes))) for nodes in level))
                for level in batches
            )
        ),
        dtype=jnp.int32,
    )
    emission_counts = jnp.asarray(
        tuple(tuple(len(nodes) for nodes in level) for level in batches), dtype=jnp.int32
    )
    return IntervalTreeLevelRuntime(
        field(tuple(level.left_slots for level in levels), slot_count, -1),
        field(tuple(level.right_slots for level in levels), slot_count, -1),
        field(tuple(level.emission_slots for level in levels), emission_count, 0),
        field(tuple(level.observations for level in levels), emission_count, 0),
        field(tuple(level.masks for level in levels), emission_count, 0),
        field(tuple(level.weights for level in levels), emission_count, 1),
        emitting_nodes,
        emission_counts,
    )


@dataclass(frozen=True)
class ParticleIntervalObservations:
    """One observation contract shared by path densities and tree stitching."""

    dynamical_model: Callable[[ParticleContext], dsx.DynamicalModel]
    operator: ObservationOperator
    tree: CompiledIntervalTree
    state_projection: Callable[[ParticleContext, jax.Array, jax.Array | int], jax.Array]
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
        densities = trajectory_observation_log_probs(
            self.state_projection(context, path, jnp.arange(path.shape[0])),
            observations,
            measurement,
            compiled,
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
            lambda bank, index: jax.vmap(
                lambda state: jax.vmap(self.response, in_axes=(0, None, None))(
                    contexts, state, index
                )
            )(bank)
        )(particles, jnp.arange(particles.shape[0]))
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
        return self._emission_log_probs(
            contexts, left_emitted[:, None] + right_emitted[None], level, node
        )

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
            groups = compile_law_groups(
                observation.laws, self.operator.interval_summary_indices, interval=True
            )

            def row_density(
                y: jax.Array, mask: jax.Array, weight: jax.Array, candidates: jax.Array
            ) -> jax.Array:
                score = prepare_mean_density(groups, y, observation.measurement.manifest_cov, mask)

                def candidate_density(moments: jax.Array) -> jax.Array:
                    means = expected_observation_mean(
                        jnp.zeros_like(moments[..., 0]),
                        moments[..., 0],
                        moments[..., 1]
                        if self.tree.statistic_count == 2
                        else jnp.zeros_like(moments[..., 0]),
                        weight,
                        self.operator.summary_operator_codes,
                    )
                    return score(means)

                return jax.vmap(candidate_density)(candidates)

            return jax.vmap(row_density, in_axes=(0, 0, 0, 1))(values, masks, weights, stats).sum(
                axis=0
            )

        # Both aligned particles and the full child-pair grid share the same
        # response-independent law terms for each parameter label and row.
        flat_statistics = statistics.reshape((-1, *statistics.shape[-4:]))
        densities = jax.vmap(density, in_axes=(0, 1), out_axes=1)(contexts, flat_statistics)
        densities = densities.reshape((*statistics.shape[:-4], statistics.shape[-4]))
        return (
            densities
            if self.power is None
            else jnp.where(self.power == 0, 0, self.power * densities)
        )


def gather_interval_statistics(statistics: jax.Array, slots: jax.Array) -> jax.Array:
    """Gather a child frontier, using its zero slot for windows absent in that child."""
    zeros = jnp.zeros((*statistics.shape[:-3], 1, *statistics.shape[-2:]), dtype=statistics.dtype)
    return jnp.take(jnp.concatenate([statistics, zeros], axis=-3), slots, axis=-3)
