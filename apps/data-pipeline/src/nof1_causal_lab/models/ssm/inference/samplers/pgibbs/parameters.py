"""Conditional parameter transitions for blocked Particle Gibbs.

Both kernels evaluate the exact joint density at the current latent coordinates.
ParticleTarget replays deterministic states under each proposed parameter value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from numpyro.infer.hmc import HMCState, hmc

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.artifacts.sampling import MetropolisHastingsSpec, ConditionalNutsSpec
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True)
class ParameterKernel[State]:
    """A conditional transition and projections of its private adaptation state."""

    initialize: Callable[[jax.Array, jax.Array, jax.Array], State]
    step: Callable[[State, jax.Array, jax.Array, jax.Array], tuple[State, Mapping[str, jax.Array]]]
    position: Callable[[State], jax.Array]
    step_size: Callable[[State], jax.Array]
    is_finite: Callable[[State], jax.Array]


class MetropolisState(NamedTuple):
    """Position and warmup-only dual averaging for a Gaussian random walk."""

    position: jax.Array
    iteration: jax.Array
    adaptation: DualAveragingAdaptationState
    step_size: jax.Array


def metropolis_kernel(
    target: ParticleTarget,
    spec: MetropolisHastingsSpec,
    parameter_chol: jax.Array,
    num_warmup: int,
) -> ParameterKernel[MetropolisState]:
    """Build an MH update of theta conditional on the current free latent states."""
    adapt_init, adapt_step, _ = dual_averaging_adaptation(spec.target_accept_prob)

    def initialize(position: jax.Array, _path: jax.Array, _key: jax.Array) -> MetropolisState:
        scale = jnp.asarray(spec.initial_step_size, dtype=position.dtype)
        return MetropolisState(position, jnp.asarray(0), adapt_init(scale), scale)

    def step(
        state: MetropolisState, path: jax.Array, current_density: jax.Array, key: jax.Array
    ) -> tuple[MetropolisState, Mapping[str, jax.Array]]:
        proposal_key, accept_key = jax.random.split(key)
        proposal = state.position + state.step_size * (
            parameter_chol
            @ jax.random.normal(proposal_key, state.position.shape, dtype=state.position.dtype)
        )
        proposed_density = target.log_posterior(proposal, path, target.observations, target.times)
        log_accept = jnp.minimum(0.0, proposed_density - current_density)
        accept_prob = jnp.where(jnp.isfinite(proposed_density), jnp.exp(log_accept), 0.0)
        accepted = jax.random.uniform(accept_key) < accept_prob
        position = jnp.where(accepted, proposal, state.position)

        def adapt() -> tuple[DualAveragingAdaptationState, jax.Array]:
            adaptation = adapt_step(state.adaptation, accept_prob)
            log_scale = jnp.where(
                state.iteration + 1 == num_warmup,
                adaptation.log_step_size_avg,
                adaptation.log_step_size,
            )
            return adaptation, jnp.exp(log_scale).astype(state.position.dtype)

        adaptation, scale = jax.lax.cond(
            state.iteration < num_warmup,
            adapt,
            lambda: (state.adaptation, state.step_size),
        )
        return MetropolisState(position, state.iteration + 1, adaptation, scale), {
            "parameter_accept_prob": accept_prob,
            "parameter_accepted": accepted,
            "step_size": state.step_size,
        }

    return ParameterKernel(
        initialize,
        step,
        lambda state: state.position,
        lambda state: state.step_size,
        lambda state: jnp.all(jnp.isfinite(state.position)),
    )


def nuts_kernel(
    target: ParticleTarget,
    spec: ConditionalNutsSpec,
    parameter_chol: jax.Array,
    num_warmup: int,
) -> ParameterKernel[HMCState]:
    """Build conditional NUTS, refreshing its density and gradient after every path update."""

    def potential(path: jax.Array) -> Callable[[jax.Array], jax.Array]:
        return lambda position: (
            -target.log_posterior(position, path, target.observations, target.times)
        )

    initialize_hmc, sample_hmc = hmc(potential_fn_gen=potential, algo="NUTS")

    def initialize(position: jax.Array, path: jax.Array, key: jax.Array) -> HMCState:
        return initialize_hmc(
            position,
            num_warmup=num_warmup,
            step_size=spec.initial_step_size,
            inverse_mass_matrix=jnp.sum(parameter_chol * parameter_chol, axis=1),
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=spec.target_accept_prob,
            max_tree_depth=spec.max_tree_depth,
            model_args=(path,),
            rng_key=key,
        )

    def step(
        state: HMCState, path: jax.Array, _current_density: jax.Array, key: jax.Array
    ) -> tuple[HMCState, Mapping[str, jax.Array]]:
        energy, gradient = jax.value_and_grad(potential(path))(state.z)
        refreshed = state._replace(potential_energy=energy, z_grad=gradient, rng_key=key)
        updated = sample_hmc(refreshed, model_args=(path,))
        return updated, {
            "parameter_accept_prob": jnp.asarray(updated.accept_prob),
            "parameter_accepted": jnp.any(updated.z != state.z),
            "step_size": jnp.asarray(state.adapt_state.step_size),
            "diverging": jnp.asarray(updated.diverging),
            "num_steps": jnp.asarray(updated.num_steps),
            "tree_depth": jnp.ceil(jnp.log2(updated.num_steps + 1)).astype(jnp.int32),
            "max_tree_depth_reached": jnp.asarray(updated.num_steps) >= 2**spec.max_tree_depth - 1,
        }

    return ParameterKernel(
        initialize,
        step,
        lambda state: state.z,
        lambda state: state.adapt_state.step_size,
        lambda state: jnp.isfinite(state.potential_energy) & jnp.all(jnp.isfinite(state.z_grad)),
    )


def fixed_parameter_kernel() -> ParameterKernel[jax.Array]:
    """The unique conditional transition when a model has no free parameters."""

    def step(
        position: jax.Array, _path: jax.Array, _density: jax.Array, _key: jax.Array
    ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
        information = dict[str, jax.Array]()
        return position, information

    return ParameterKernel(
        lambda position, _path, _key: position,
        step,
        lambda position: position,
        lambda position: jnp.asarray(0.0, dtype=position.dtype),
        lambda position: jnp.all(jnp.isfinite(position)),
    )
