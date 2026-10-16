"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

from nof1_causal_lab.models.ssm.compile.state_space import ParallelRecurrences
from nof1_causal_lab.models.ssm.execution.affine import compose_affine_maps
from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import (
    GaussianReference,
    GaussianReferenceBridge,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from nof1_causal_lab.models.ssm.compile.observation_partition import (
        CompiledObservationPartition,
    )
    from nof1_causal_lab.models.ssm.compile.state_space import StateSpace
    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
    from nof1_causal_lab.models.ssm.inference.targets.intervals import ParticleIntervalObservations


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array, jax.Array]


def identity_state(_context: ParticleContext, state: jax.Array) -> jax.Array:
    """Leave a supplied state unchanged when evaluating the exact model factors."""
    return state


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
    observation_partition: CompiledObservationPartition
    intervals: ParticleIntervalObservations | None = None
    exact_constraints: ExactStateConstraints | None = None
    bridge: GaussianReferenceBridge | None = None
    state_projection: Callable[[ParticleContext, jax.Array], jax.Array] = identity_state

    def tree_target(self) -> ParticleTarget:
        """Keep replay-independent observations in the exact intermediate tree target."""
        if not self.state_space.requires_replay:
            return self
        channels = jnp.asarray(self.observation_partition.local, dtype=jnp.int32)
        observations = (
            jnp.full_like(self.observations, jnp.nan)
            .at[:, channels]
            .set(self.observations[:, channels])
        )
        return replace(
            self,
            observations=observations,
            state_projection=self.fixed_recurrences,
            intervals=None
            if self.intervals is None
            else self.intervals.select_channels(self.observation_partition.local),
        )

    def replay_correction(
        self, local_target: ParticleTarget, context: ParticleContext, path: jax.Array
    ) -> jax.Array:
        """Pay only the exact recurrence correction and deferred observation factors."""
        if not self.state_space.requires_replay:
            return jnp.zeros((), dtype=path.dtype)
        replayed = self.replay_path(context, path)
        correction = self.prior_path_log_prob(context, replayed) - local_target.prior_path_log_prob(
            context, path
        )
        if self.observation_partition.replayed:
            channels = jnp.asarray(self.observation_partition.replayed, dtype=jnp.int32)
            observations = (
                jnp.full_like(self.observations, jnp.nan)
                .at[:, channels]
                .set(self.observations[:, channels])
            )
            correction = correction + jnp.sum(
                self.observation_log_probs(context, replayed, observations)
            )
        return jnp.asarray(correction)

    def project_state(self, context: ParticleContext, state: jax.Array) -> jax.Array:
        """Bind global static coordinates before evaluating a local model factor."""
        indices = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        state = state.at[..., indices].set(context[2][indices])
        return self.state_projection(context, state)

    def fixed_recurrences(self, context: ParticleContext, state: jax.Array) -> jax.Array:
        """A parameter-only local reference; the root pays its exact density correction."""
        indices = jnp.asarray(self.state_space.recurrence.indices, dtype=jnp.int32)
        initial = self.initial_moments(context)[0]
        if self.exact_constraints is not None:
            initial = jnp.where(
                self.exact_constraints.free_mask[0], initial, self.exact_constraints.values[0]
            )
        return state.at[..., indices].set(initial[indices])

    def at_temperature(self, reference: GaussianReference, beta: jax.Array) -> ParticleTarget:
        """Compose a reference bridge without changing the exact model or its support."""
        return replace(
            self,
            bridge=GaussianReferenceBridge(reference, beta),
            intervals=None if self.intervals is None else replace(self.intervals, power=beta),
        )

    @property
    def state_density_mask(self) -> jax.Array:
        """State density coordinates, including one copy of every random static state."""
        mask = jnp.zeros((self.times.size, self.state_space.size), dtype=bool)
        mask = mask.at[0, jnp.asarray(self.state_space.initial, dtype=jnp.int32)].set(True)
        return mask.at[1:, jnp.asarray(self.state_space.transition, dtype=jnp.int32)].set(True)

    @property
    def independent_mask(self) -> jax.Array:
        """Free path coordinates; random static states are sampled in the global block."""
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        return self.state_density_mask.at[:, static].set(False)

    def replay_path(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct exact initial values, static quantities and deterministic recurrences."""
        if self.exact_constraints is not None:
            coordinates = self.exact_constraints.project(coordinates)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        coordinates = coordinates.at[:, static].set(context[2][static])
        if not self.state_space.requires_replay:
            return coordinates
        independent = self.independent_mask
        initial = jnp.where(independent[0], coordinates[0], self.initial_moments(context)[0])
        inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
        initial = initial.at[inputs].set(coordinates[0, inputs])
        initial = initial.at[static].set(context[2][static])
        if isinstance(self.state_space.recurrence, ParallelRecurrences):
            coordinates = coordinates.at[0].set(initial)
            deterministic = jnp.asarray(self.state_space.recurrence.indices, dtype=jnp.int32)
            if not self.state_space.recurrence.indices or self.times.size == 1:
                return coordinates

            def affine(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
                zero = previous.at[deterministic].set(0)
                offset = jnp.asarray(self._transition_distribution(context, zero, index).mean)[
                    deterministic
                ]
                # The compiler proved affine dependence. Basis evaluations recover
                # its exact coefficients, not a local approximation of nonlinear drift.
                basis = jnp.eye(deterministic.size, dtype=coordinates.dtype)
                columns = jax.vmap(
                    lambda unit: (
                        self._transition_distribution(
                            context, zero.at[deterministic].set(unit), index
                        ).mean[deterministic]
                        - offset
                    )
                )(basis)
                return jnp.asarray(columns).T, offset

            matrices, offsets = jax.vmap(affine)(coordinates[:-1], jnp.arange(1, self.times.size))
            matrices = jnp.concatenate((jnp.zeros_like(matrices[:1]), matrices))
            offsets = jnp.concatenate((initial[deterministic][None], offsets))

            _, values = jax.lax.associative_scan(compose_affine_maps, (matrices, offsets))
            return coordinates.at[:, deterministic].set(values)

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
        state = self.project_state(context, state)

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
        # The declared Gaussian embeds its factor on precisely these stochastic
        # coordinates. Reusing it avoids rebuilding the full-state covariance
        # and differentiating through a second Cholesky decomposition.
        return dist.MultivariateNormal(
            distribution.mean[indices],
            scale_tril=distribution.scale_tril[jnp.ix_(indices, indices)],
        ).log_prob(state[indices])

    def _transition_distribution(self, context, previous, index):
        times = context[1]
        evolution = cast("DiscreteStateTransition", self.dynamical_model(context).state_evolution)
        return evolution(
            self.project_state(context, previous),
            None,
            times[jnp.maximum(index - 1, 0)],
            times[index],
        )

    def transition_log_prob(self, context, previous, current, index):
        """Evaluate the current state's modeled transition density from its predecessor."""
        current = self.project_state(context, current)

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

    def pairwise_transition_log_prob(self, context, previous, current, index):
        """Score all Gaussian pairs with per-ancestor factors and matrix products.

        The declared nonlinear transition and its state-dependent covariance are
        evaluated once per ancestor. Expanding the exact Gaussian quadratic avoids
        N squared triangular solves and N-by-N-by-state residual workspaces.
        """

        def exact():
            if not self.state_space.transition:
                return jnp.zeros((previous.shape[0], current.shape[0]), dtype=current.dtype)
            indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)
            values = self.project_state(context, current)[:, indices]
            anchor = jnp.mean(values, axis=0)
            centered = values - anchor
            dimension = len(self.state_space.transition)

            def row(ancestor):
                distribution = cast(
                    "dist.MultivariateNormal",
                    self._transition_distribution(context, ancestor, index),
                )
                chol = jnp.asarray(distribution.scale_tril)[jnp.ix_(indices, indices)]
                inverse = solve_triangular(chol, jnp.eye(dimension, dtype=chol.dtype), lower=True)
                precision = inverse.T @ inverse
                mean = distribution.mean[indices] - anchor
                whitened = inverse @ mean
                linear = inverse.T @ whitened
                constant = (
                    -0.5 * jnp.sum(whitened**2)
                    - jnp.sum(jnp.log(jnp.diag(chol)))
                    - 0.5 * dimension * jnp.log(2 * jnp.pi)
                )
                return precision, linear, constant

            precision, linear, constant = jax.vmap(row)(previous)
            outer = jnp.einsum("ni,nj->nij", centered, centered).reshape(
                current.shape[0], dimension**2
            )
            quadratic = precision.reshape(previous.shape[0], dimension**2) @ outer.T
            return constant[:, None] + linear @ centered.T - 0.5 * quadratic

        bridge = self.bridge
        if bridge is None:
            return exact()
        reference = jax.vmap(
            lambda ancestor: jax.vmap(
                lambda descendant: bridge.reference.transition_log_prob(ancestor, descendant, index)
            )(current)
        )(previous)
        return bridge.combine(exact, reference)

    def initial_path(self, context, *, exact_constraints: ExactStateConstraints | None = None):
        """Build an initialization path from successive transition means and any exact state constraints."""
        initial = jnp.asarray(self.dynamical_model(context).initial_condition.mean)
        initial = self.project_state(context, initial)
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

    def observation_increment(
        self,
        context: ParticleContext,
        state: jax.Array,
        index: jax.Array | int,
        observations: jax.Array,
    ) -> jax.Array:
        """Score point factors at one leaf; interval factors belong to enclosing tree nodes."""
        if self.intervals is not None:
            observations = jnp.where(
                self.intervals.operator.point_like_mask(jnp.dtype(bool))[None, :],
                observations,
                jnp.nan,
            )
        distribution = self.dynamical_model(context).observation_model(
            self.project_state(context, state), None, context[1][index]
        )

        def exact() -> jax.Array:
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
        """Evaluate the joint path density after reconstructing constrained coordinates.

        Exact readings and deterministic coordinates are replayed before scoring
        the initial, transition, and observation factors.
        """
        path = self.replay_path(context, path)
        return self.factor_log_prob(context, path, observations)

    def factor_log_prob(self, context, path, observations):
        """Sum local factors on supplied coordinates, without a path reconstruction."""
        return self.prior_path_log_prob(context, path) + jnp.sum(
            self.observation_log_probs(context, path, observations)
        )

    def prior_path_log_prob(self, context, path):
        """Sum initial and transition factors on supplied coordinates without replay."""
        transitions = jax.vmap(
            lambda previous, current, index: self.transition_log_prob(
                context, previous, current, index
            )
        )(path[:-1], path[1:], jnp.arange(1, path.shape[0]))
        return self.initial_log_prob(context, path[0]) + jnp.sum(transitions)

    def log_posterior_from_context(
        self,
        position: jax.Array,
        context: ParticleContext,
        path: jax.Array,
        observations: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return joint posterior and path log densities using an already bound model context."""
        path_density: jax.Array = self.path_log_prob(context, path, observations)
        joint_density: jax.Array = self.log_prior(position) + path_density
        return joint_density, path_density

    def log_posterior(
        self,
        position: jax.Array,
        path: jax.Array,
        observations: jax.Array,
        times: jax.Array,
    ) -> jax.Array:
        """Bind the parameter context and evaluate the joint parameter-and-path posterior density."""
        return self.log_posterior_from_context(
            position, self.context(position, times), path, observations
        )[0]
