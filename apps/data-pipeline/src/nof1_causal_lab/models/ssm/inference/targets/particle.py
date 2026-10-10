"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import (
    GaussianReference,
    GaussianReferenceBridge,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from nof1_causal_lab.models.ssm.compile.state_space import StateSpace
    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
    from nof1_causal_lab.models.ssm.inference.targets.intervals import ParticleIntervalObservations


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array]


@dataclass(frozen=True, eq=False)
class ParticleTarget:
    """Adapt a Dynestyx model to the application's particle sampling contract.

    Every transition operation uses the same declared distribution. The model's
    observation distribution owns missingness and the true emission density.
    """

    parameters: ParameterTransform
    context: Callable[[jax.Array, jax.Array], ParticleContext]
    dynamical_model: Callable[[ParticleContext], dsx.DynamicalModel]
    observations: jax.Array
    times: jax.Array
    state_space: StateSpace
    intervals: ParticleIntervalObservations | None = None
    exact_constraints: ExactStateConstraints | None = None
    bridge: GaussianReferenceBridge | None = None

    def at_temperature(self, reference: GaussianReference, beta: jax.Array) -> ParticleTarget:
        """Compose a reference bridge without changing the exact model or its support."""
        return replace(
            self,
            bridge=GaussianReferenceBridge(reference, beta),
            intervals=None if self.intervals is None else replace(self.intervals, power=beta),
        )

    @property
    def independent_mask(self) -> jax.Array:
        """Coordinates with their own density, before conditioning on exact readings."""
        mask = jnp.zeros((self.times.size, self.state_space.size), dtype=bool)
        mask = mask.at[0, jnp.asarray(self.state_space.initial, dtype=jnp.int32)].set(True)
        return mask.at[1:, jnp.asarray(self.state_space.transition, dtype=jnp.int32)].set(True)

    def replay_path(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct exact initial values, static quantities and deterministic recurrences."""
        if self.exact_constraints is not None:
            coordinates = self.exact_constraints.project(coordinates)
        if not self.state_space.requires_replay:
            return coordinates
        independent = self.independent_mask
        initial = jnp.where(independent[0], coordinates[0], self.initial_moments(context)[0])
        inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
        initial = initial.at[inputs].set(coordinates[0, inputs])

        def advance(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
            mean = self._transition_distribution(context, previous, index).mean
            current = jnp.where(independent[index], coordinates[index], mean)
            # Inputs have no density and can change at every model grid point.
            inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
            current = current.at[inputs].set(coordinates[index, inputs])
            return current, current

        _, tail = jax.lax.scan(advance, initial, jnp.arange(1, self.times.size))
        return jnp.concatenate((initial[None], tail))

    @property
    def initial_position(self):
        """Initial unconstrained parameter position supplied by the parameter target."""
        return self.parameters.initial_position

    def log_prior(self, position):
        """Evaluate the parameter prior density at the supplied unconstrained position."""

        def exact():
            return self.parameters.log_prior(position)

        return (
            exact()
            if self.bridge is None
            else self.bridge.combine(exact, self.bridge.reference.parameter_log_prob(position))
        )

    def initial_moments(self, context):
        """Return the exact model's Gaussian initial-state mean and covariance for this context."""
        distribution = self.dynamical_model(context).initial_condition
        assert isinstance(
            distribution, dist.MultivariateNormal
        )  # The exact model producer declares a Gaussian initial law.
        return jnp.asarray(distribution.mean), jnp.asarray(distribution.covariance_matrix)

    def initial_log_prob(self, context, state):
        """Evaluate initial-state density on the modeled coordinates of the supplied state."""

        def exact():
            return self._modeled_log_prob(
                self.dynamical_model(context).initial_condition, state, self.state_space.initial
            )

        return (
            exact()
            if self.bridge is None
            else self.bridge.combine(exact, self.bridge.reference.initial_log_prob(state))
        )

    def _modeled_log_prob(self, distribution, state, density_indices):
        if not density_indices:
            return jnp.asarray(0.0, dtype=state.dtype)
        indices = jnp.asarray(density_indices, dtype=jnp.int32)
        return dist.MultivariateNormal(
            distribution.mean[indices],
            covariance_matrix=distribution.covariance_matrix[jnp.ix_(indices, indices)],
        ).log_prob(state[indices])

    def _transition_distribution(self, context, previous, index):
        times = context[1]
        evolution = cast("DiscreteStateTransition", self.dynamical_model(context).state_evolution)
        return evolution(previous, None, times[jnp.maximum(index - 1, 0)], times[index])

    def transition_log_prob(self, context, previous, current, index):
        """Evaluate the current state's modeled transition density from its predecessor."""

        def exact():
            return self._modeled_log_prob(
                self._transition_distribution(context, previous, index),
                current,
                self.state_space.transition,
            )

        return (
            exact()
            if self.bridge is None
            else self.bridge.combine(
                exact, self.bridge.reference.transition_log_prob(previous, current, index)
            )
        )

    def transition_moments(
        self, context: ParticleContext, previous: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Expose declared transition moments for initialization of a frozen Gaussian reference."""
        distribution = cast(
            "dist.MultivariateNormal", self._transition_distribution(context, previous, index)
        )
        return jnp.asarray(distribution.mean), jnp.asarray(distribution.covariance_matrix)

    def aligned_transition_log_prob(self, context, previous, current, index):
        """Evaluate transition densities for aligned predecessor and successor particle pairs."""
        return jax.vmap(lambda a, b: self.transition_log_prob(context, a, b, index))(
            previous, current
        )

    def pairwise_transition_log_prob(self, context, previous, current, index):
        """Evaluate every predecessor-to-successor particle transition as a pairwise matrix."""
        return jax.vmap(
            lambda ancestor: jax.vmap(
                lambda descendant: self.transition_log_prob(context, ancestor, descendant, index)
            )(current)
        )(previous)

    def initial_path(self, context, *, exact_constraints: ExactStateConstraints | None = None):
        """Build an initialization path from successive transition means and any exact state constraints."""
        initial = jnp.asarray(self.dynamical_model(context).initial_condition.mean)
        if exact_constraints is not None:
            initial = jnp.where(
                exact_constraints.free_mask[0], initial, exact_constraints.values[0]
            )

        def step(previous, index):
            current = self._transition_distribution(context, previous, index).mean
            if exact_constraints is not None:
                current = jnp.where(
                    exact_constraints.free_mask[index], current, exact_constraints.values[index]
                )
            return current, current

        _, tail = jax.lax.scan(step, initial, jnp.arange(1, context[1].size))
        return jnp.concatenate([initial[None], tail])

    def observation_increment(self, context, state, index, observations):
        """Score point factors at one leaf; interval factors belong to enclosing tree nodes."""
        if self.intervals is not None:
            observations = jnp.where(
                self.intervals.operator.point_like_mask(jnp.dtype(bool))[None, :],
                observations,
                jnp.nan,
            )
        distribution = self.dynamical_model(context).observation_model(
            state, None, context[1][index]
        )

        def exact():
            return jnp.sum(distribution.log_prob(observations[index]))

        return exact() if self.bridge is None else self.bridge.observation(exact)

    def observation_log_probs(self, context, path, observations):
        """Evaluate the observation log likelihood at each time along a latent path."""
        if self.intervals is not None:
            return self.intervals.trajectory_log_probs(context, path, observations)
        return jax.vmap(
            lambda state, index: self.observation_increment(context, state, index, observations)
        )(path, jnp.arange(path.shape[0]))

    def path_log_prob(self, context, path, observations):
        """Sum initial, transition, and observation log densities for a complete latent path."""
        path = self.replay_path(context, path)
        transitions = jax.vmap(
            lambda previous, current, index: self.transition_log_prob(
                context, previous, current, index
            )
        )(path[:-1], path[1:], jnp.arange(1, path.shape[0]))
        return (
            self.initial_log_prob(context, path[0])
            + jnp.sum(transitions)
            + jnp.sum(self.observation_log_probs(context, path, observations))
        )

    def log_posterior_from_context(self, position, context, path, observations):
        """Return joint posterior and path log densities using an already bound model context."""
        path_density = self.path_log_prob(context, path, observations)
        return self.log_prior(position) + path_density, path_density

    def log_posterior(self, position, path, observations, times):
        """Bind the parameter context and evaluate the joint parameter-and-path posterior density."""
        return self.log_posterior_from_context(
            position, self.context(position, times), path, observations
        )[0]
