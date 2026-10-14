"""Gaussian-reference replica exchange around the unchanged local MPGibbs kernel.

See Syed et al., https://arxiv.org/abs/1905.02939, for the geometric bridge,
alternating even/odd swaps, and cumulative-rejection ladder adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.sampling import GaussianReferenceTemperingSpec
    from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.kernel import (
        MarginalParticleGibbsKernel,
    )
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCState
    from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import GaussianReference
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


class ExchangeRecord(NamedTuple):
    """Per-ladder, per-edge swap evidence from one sweep."""

    accepted: jax.Array
    attempted: jax.Array
    rejection_probability: jax.Array


def exchange_permutation(
    betas: jax.Array, energies: jax.Array, parity: jax.Array, key: jax.Array
) -> tuple[jax.Array, ExchangeRecord]:
    """Exchange disjoint neighboring states using log p minus log r as the energy."""
    edge = jnp.arange(betas.size - 1)
    attempted = edge % 2 == parity
    log_ratio = (betas[:-1] - betas[1:]) * (energies[1:] - energies[:-1])
    accepted = attempted & (jnp.log(jax.random.uniform(key, log_ratio.shape)) < log_ratio)
    index = jnp.arange(betas.size)
    permutation = (
        index
        + jnp.pad(accepted.astype(jnp.int32), (0, 1))
        - jnp.pad(accepted.astype(jnp.int32), (1, 0))
    )
    return permutation, ExchangeRecord(
        accepted=accepted,
        attempted=attempted,
        rejection_probability=jnp.where(attempted, -jnp.expm1(jnp.minimum(log_ratio, 0)), 0.0),
    )


def adapt_temperature_ladder(
    betas: jax.Array, rejection_sum: jax.Array, attempts: jax.Array
) -> jax.Array:
    """Equalize the piecewise-linear cumulative rejection barrier during warmup.

    A half rejection and half acceptance pseudocount per edge keeps the estimated
    barrier strictly increasing even when every observed exchange was accepted.
    """
    rejection = (jnp.sum(rejection_sum, axis=0) + 0.5) / (jnp.sum(attempts, axis=0) + 1)
    cumulative = jnp.concatenate((jnp.zeros(1, dtype=betas.dtype), jnp.cumsum(rejection)))
    return jnp.interp(jnp.linspace(0, cumulative[-1], betas.size), cumulative, betas).astype(
        betas.dtype
    )


def _bridge_density(beta: jax.Array, exact: jax.Array, reference: jax.Array) -> jax.Array:
    return jnp.where(
        beta == 0, reference, jnp.where(beta == 1, exact, beta * exact + (1 - beta) * reference)
    )


@dataclass(frozen=True, eq=False)
class ReplicaExchangeKernel:
    """A fixed reference and exact target shared by independent temperature ladders."""

    target: ParticleTarget
    reference: GaussianReference
    kernel_for_target: Callable[[ParticleTarget], MarginalParticleGibbsKernel]
    spec: GaussianReferenceTemperingSpec

    def rebind(self, states: TrajectoryMCMCState, betas: jax.Array) -> TrajectoryMCMCState:
        """Recompute parameter contexts, replayed paths, and both density caches."""

        def one(state: TrajectoryMCMCState, beta: jax.Array) -> TrajectoryMCMCState:
            target = self.target.at_temperature(self.reference, beta)
            context = target.context(state.position, target.times)
            path = target.replay_path(context, state.latent_trajectory)
            complete, trajectory = target.log_posterior_from_context(
                state.position, context, path, target.observations
            )
            return state._replace(
                latent_context=context,
                latent_trajectory=path,
                trajectory_log_prob=trajectory,
                complete_log_posterior=complete,
            )

        return jax.vmap(one)(states, jnp.tile(betas, states.position.shape[0] // betas.size))

    def step(
        self,
        states: TrajectoryMCMCState,
        keys: jax.Array,
        betas: jax.Array,
        index: jax.Array,
    ) -> tuple[TrajectoryMCMCState, dict[str, jax.Array], ExchangeRecord]:
        """Update local replicas, independently refresh beta zero, then exchange states."""
        levels = betas.size
        chains = states.position.shape[0] // levels
        population = jax.tree.map(
            lambda value: value.reshape((chains, levels, *value.shape[1:])), states
        )
        moved, info, record = jax.vmap(
            lambda state, key: self._ladder_step(state, key, betas, index)
        )(population, keys.reshape((chains, levels, *keys.shape[1:])))

        def flatten(value: jax.Array) -> jax.Array:
            return value.reshape((chains * levels, *value.shape[2:]))

        flat_states: TrajectoryMCMCState = jax.tree.map(flatten, moved)
        flat_info: dict[str, jax.Array] = jax.tree.map(flatten, info)
        return flat_states, flat_info, record

    def _ladder_step(
        self,
        states: TrajectoryMCMCState,
        keys: jax.Array,
        betas: jax.Array,
        index: jax.Array,
    ) -> tuple[TrajectoryMCMCState, dict[str, jax.Array], ExchangeRecord]:
        def local(
            state: TrajectoryMCMCState, key: jax.Array, beta: jax.Array
        ) -> tuple[TrajectoryMCMCState, dict[str, jax.Array]]:
            return self.kernel_for_target(self.target.at_temperature(self.reference, beta)).step_fn(
                state, key
            )

        local_states, local_info = jax.vmap(local)(
            jax.tree.map(lambda value: value[:-1], states), keys[:-1], betas[:-1]
        )
        moved: TrajectoryMCMCState = jax.tree.map(
            lambda value, old: jnp.concatenate((value, old[-1:])), local_states, states
        )
        info: dict[str, jax.Array] = jax.tree.map(
            lambda value: jnp.concatenate((value, jnp.zeros_like(value[:1]))), local_info
        )
        refresh_key, swap_key = jax.random.split(keys[-1])
        position, coordinates = self.reference.sample(refresh_key)
        positions = moved.position.at[-1].set(position)
        paths = moved.latent_trajectory.at[-1].set(coordinates)
        contexts = jax.vmap(lambda p: self.target.context(p, self.target.times))(positions)
        paths = jax.vmap(self.target.replay_path)(contexts, paths)
        exact, exact_path = jax.vmap(
            lambda p, context, path: self.target.log_posterior_from_context(
                p, context, path, self.target.observations
            )
        )(positions, contexts, paths)
        reference_path = jax.vmap(self.reference.path_log_prob)(paths)
        reference = jax.vmap(self.reference.parameter_log_prob)(positions) + reference_path
        permutation, record = exchange_permutation(betas, exact - reference, index % 2, swap_key)
        return (
            moved._replace(
                position=positions[permutation],
                latent_context=jax.tree.map(lambda value: value[permutation], contexts),
                latent_trajectory=paths[permutation],
                trajectory_log_prob=_bridge_density(
                    betas, exact_path[permutation], reference_path[permutation]
                ),
                complete_log_posterior=_bridge_density(
                    betas, exact[permutation], reference[permutation]
                ),
            ),
            info,
            record,
        )
