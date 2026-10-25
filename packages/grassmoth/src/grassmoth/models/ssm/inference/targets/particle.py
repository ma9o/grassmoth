"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

from grassmoth.models.ssm.compile.state_space import (
    AdditiveRecurrences,
    SequentialRecurrences,
)
from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.execution.gaussian_noise import GaussianNoiseFactors
from grassmoth.models.ssm.execution.innovations import GaussianPathMap
from grassmoth.models.ssm.execution.observation_model import (
    grouped_density_with_noise,
    prepare_grouped_noise,
)
from grassmoth.models.ssm.inference.targets.gaussian_reference import (
    GaussianReference,
    GaussianReferenceBridge,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from grassmoth.models.ssm.compile.observation_partition import (
        CompiledObservationPartition,
    )
    from grassmoth.models.ssm.compile.state_space import StateSpace
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.state_transition import SupportedEulerMaruyama
    from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
    from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
    from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array, jax.Array]


def identity_state(_context: ParticleContext, state: jax.Array) -> jax.Array:
    """Leave a supplied state unchanged when evaluating the exact model factors."""
    return state


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ParticleNoiseFactors:
    """Exact noise factors for one fixed parameter context and its observation grid."""

    transitions: GaussianNoiseFactors
    observations: tuple[GaussianNoiseFactors, ...]


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
    point_reading_mask: jax.Array
    intervals: ParticleIntervalObservations | None = None
    exact_constraints: ExactStateConstraints | None = None
    bridge: GaussianReferenceBridge | None = None
    state_projection: Callable[[ParticleContext, jax.Array], jax.Array] = identity_state

    def tree_target(self) -> ParticleTarget:
        """Keep replay-independent observations in the exact intermediate tree target."""
        if not self.state_space.requires_replay:
            return self
        return replace(
            self._select_observations(self.observation_partition.local),
            state_projection=self.fixed_recurrences,
        )

    def _select_observations(self, channels: tuple[int, ...]) -> ParticleTarget:
        """Bind only the selected factors and their required interval statistics."""
        selected = jnp.asarray(channels, dtype=jnp.int32)
        observations = (
            jnp.full_like(self.observations, jnp.nan)
            .at[:, selected]
            .set(self.observations[:, selected])
        )
        return replace(
            self,
            observations=observations,
            intervals=None if self.intervals is None else self.intervals.select_channels(channels),
        )

    def replay_correction(
        self, context: ParticleContext, path: jax.Array, local_prior_log_prob: jax.Array
    ) -> jax.Array:
        """Replace the carried local prior and pay deferred observation factors."""
        if not self.state_space.requires_replay:
            return jnp.zeros((), dtype=path.dtype)
        replayed = self.replay_path(context, path)
        correction = self.prior_path_log_prob(context, replayed) - local_prior_log_prob
        if self.observation_partition.replayed:
            deferred = self._select_observations(self.observation_partition.replayed)
            if deferred.intervals is None:
                channels = self.observation_partition.replayed
                observation = cast(
                    "HeterogeneousObservation", self.dynamical_model(context).observation_model
                ).select_channels(channels)
                readings = self.observations[:, jnp.asarray(channels, dtype=jnp.int32)]

                def point_density(state: jax.Array, reading: jax.Array) -> jax.Array:
                    def exact() -> jax.Array:
                        return observation(self.project_state(context, state), None, 0).log_prob(
                            reading
                        )

                    return exact() if self.bridge is None else self.bridge.observation(exact)

                densities = jax.vmap(point_density)(replayed, readings)
            else:
                densities = deferred.observation_log_probs(context, replayed, deferred.observations)
            correction = correction + jnp.sum(densities)
        return jnp.asarray(correction)

    def project_state(self, context: ParticleContext, state: jax.Array) -> jax.Array:
        """Bind global static coordinates before evaluating a local model factor."""
        indices = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        state = state.at[..., indices].set(context[2][indices])
        return self.state_projection(context, state)

    def fixed_recurrences(self, context: ParticleContext, state: jax.Array) -> jax.Array:
        """A parameter-only local reference; the root pays its exact density correction."""
        indices = jnp.asarray(
            tuple(index for stage in self.state_space.recurrence for index in stage.indices),
            dtype=jnp.int32,
        )
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

    def replay_initial(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct the origin from constrained independent coordinates and fixed values."""
        state = jnp.where(self.independent_mask[0], coordinates, self.initial_moments(context)[0])
        inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        return state.at[inputs].set(coordinates[inputs]).at[static].set(context[2][static])

    def replay_step(
        self,
        context: ParticleContext,
        previous: jax.Array,
        coordinates: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Advance exact recurrences with their declared previous-time dependencies."""
        mean = self._transition_distribution(context, previous, index).mean
        state = jnp.where(self.independent_mask[index], coordinates, mean)
        inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        return state.at[inputs].set(coordinates[inputs]).at[static].set(context[2][static])

    def replay_path(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct exact initial values, static quantities and deterministic recurrences."""
        if self.exact_constraints is not None:
            coordinates = self.exact_constraints.project(coordinates)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        coordinates = coordinates.at[:, static].set(context[2][static])
        if not self.state_space.requires_replay:
            return coordinates
        initial = self.replay_initial(context, coordinates[0])
        if any(isinstance(stage, SequentialRecurrences) for stage in self.state_space.recurrence):

            def advance(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
                current = self.replay_step(context, previous, coordinates[index], index)
                return current, current

            _, tail = jax.lax.scan(advance, initial, jnp.arange(1, self.times.size))
            return jnp.concatenate((initial[None], tail))

        coordinates = coordinates.at[0].set(initial)
        if self.times.size == 1:
            return coordinates
        evolution = cast("SupportedEulerMaruyama", self.dynamical_model(context).state_evolution)
        for stage in self.state_space.recurrence:
            deterministic = jnp.asarray(stage.indices, dtype=jnp.int32)
            if isinstance(stage, AdditiveRecurrences):

                def increment(
                    previous: jax.Array, index: jax.Array, deterministic: jax.Array = deterministic
                ) -> jax.Array:
                    state = self.project_state(context, previous)
                    start, stop = context[1][index - 1], context[1][index]
                    drift = evolution.continuous.total_drift(x=state, u=None, t=start)
                    return (stop - start) * drift[deterministic]

                increments = jax.vmap(increment)(coordinates[:-1], jnp.arange(1, self.times.size))
                values = initial[deterministic] + jnp.cumsum(increments, axis=0)
                coordinates = coordinates.at[1:, deterministic].set(values)
            else:

                def affine(
                    previous: jax.Array, index: jax.Array, deterministic: jax.Array = deterministic
                ) -> tuple[jax.Array, jax.Array]:
                    zero = previous.at[deterministic].set(0)
                    offset = jnp.asarray(self._transition_distribution(context, zero, index).mean)[
                        deterministic
                    ]
                    # The compiler proved affine dependence. Basis evaluations recover
                    # its exact coefficients, not a local approximation of nonlinear drift.
                    basis = jnp.eye(deterministic.size, dtype=previous.dtype)
                    columns = jax.vmap(
                        lambda unit: (
                            self._transition_distribution(
                                context, zero.at[deterministic].set(unit), index
                            ).mean[deterministic]
                            - offset
                        )
                    )(basis)
                    return jnp.asarray(columns).T, offset

                matrices, offsets = jax.vmap(affine)(
                    coordinates[:-1], jnp.arange(1, self.times.size)
                )
                matrices = jnp.concatenate((jnp.zeros_like(matrices[:1]), matrices))
                offsets = jnp.concatenate((initial[deterministic][None], offsets))

                _, values = jax.lax.associative_scan(compose_affine_maps, (matrices, offsets))
                coordinates = coordinates.at[:, deterministic].set(values)
        return coordinates

    def innovation_map(
        self, context: ParticleContext, innovation_mask: jax.Array
    ) -> GaussianPathMap:
        """Bind exact moments, support and pinned values for a frozen parameterization policy."""
        fixed = (
            jnp.full(self.independent_mask.shape, jnp.nan, dtype=self.observations.dtype)
            if self.exact_constraints is None
            else self.exact_constraints.values
        )
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        fixed = fixed.at[:, static].set(context[2][static])
        initial_mean, initial_covariance = self.initial_moments(context)
        # continuous_state_evolution binds the compiled diffusion as an array in
        # FullDiffusion, never a state function. Its EM covariance therefore
        # depends only on the parameters and time interval. A fixed state can
        # expose those exact covariances before the nonlinear recurrence runs.
        covariances = jax.vmap(
            lambda index: self.transition_moments(context, initial_mean, index)[1]
        )(jnp.arange(1, self.times.size))
        return GaussianPathMap(
            initial_mean,
            lambda previous, index: self.transition_moments(context, previous, index)[0],
            jnp.concatenate((initial_covariance[None], covariances)),
            self.state_density_mask,
            self.independent_mask,
            innovation_mask,
            fixed,
        )

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
        factors = GaussianNoiseFactors.from_cholesky(
            distribution.scale_tril[jnp.ix_(indices, indices)]
        )
        return factors.log_prob(state[indices] - distribution.mean[indices])

    def prepare_noise_factors(
        self, context: ParticleContext, observations: jax.Array
    ) -> ParticleNoiseFactors:
        """Factor exact noise once for this sweep's fixed parameters and masks.

        The compiled additive diffusion is an array, so its Euler-Maruyama
        covariance depends on parameters and time intervals, never on the state.
        Nonlinear state-dependent means remain in the scoring methods below.
        """
        initial = self.initial_moments(context)[0]
        indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)

        def transition(index: jax.Array) -> GaussianNoiseFactors:
            distribution = cast(
                "dist.MultivariateNormal", self._transition_distribution(context, initial, index)
            )
            return GaussianNoiseFactors.from_cholesky(
                jnp.asarray(distribution.scale_tril)[jnp.ix_(indices, indices)]
            )

        transitions = jax.vmap(transition)(jnp.arange(1, context[1].size))
        # The unused origin row also makes a traced length-zero scan well shaped.
        transitions = jax.tree.map(
            lambda value: jnp.concatenate((jnp.zeros((1, *value.shape[1:]), value.dtype), value)),
            transitions,
        )
        model = cast("HeterogeneousObservation", self.dynamical_model(context).observation_model)
        return ParticleNoiseFactors(
            transitions,
            prepare_grouped_noise(
                model.density_groups(),
                model.measurement.manifest_cov,
                self._point_observations(observations),
            ),
        )

    def transition_log_prob_with_noise(
        self,
        context: ParticleContext,
        noise: ParticleNoiseFactors,
        previous: jax.Array,
        current: jax.Array,
        index: jax.Array | int,
    ) -> jax.Array:
        """Score the exact nonlinear transition using this context's prepared noise."""
        current = self.project_state(context, current)

        def exact() -> jax.Array:
            indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)
            mean = self._transition_distribution(context, previous, index).mean
            return noise.transitions.at(index).log_prob(current[indices] - mean[indices])

        return (
            exact()
            if self.bridge is None
            else self.bridge.combine(
                exact,
                self.bridge.reference.transition_log_prob(previous, current, jnp.asarray(index)),
            )
        )

    def observation_increment_with_noise(
        self,
        context: ParticleContext,
        noise: ParticleNoiseFactors,
        state: jax.Array,
        index: jax.Array | int,
        observations: jax.Array,
    ) -> jax.Array:
        """Score exact point laws and nonlinear predictors with prepared Gaussian noise."""
        model = cast("HeterogeneousObservation", self.dynamical_model(context).observation_model)
        predictor = model.linear_predictor(self.project_state(context, state))

        def exact() -> jax.Array:
            return grouped_density_with_noise(
                model.density_groups(),
                self._point_observations(observations)[index],
                predictor,
                model.measurement.manifest_cov,
                tuple(factor.at(index) for factor in noise.observations),
            ).astype(predictor.dtype)

        return exact() if self.bridge is None else self.bridge.observation(exact)

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

    def _transition_value_and_grad(
        self,
        context: ParticleContext,
        previous: jax.Array,
        current: jax.Array,
        index: jax.Array | int,
        *,
        current_gradient: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Differentiate per-ancestor Gaussian factors before particle-pair expansion."""
        indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)
        dimension = len(self.state_space.transition)

        def factors(state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            distribution = cast(
                "dist.MultivariateNormal", self._transition_distribution(context, state, index)
            )
            chol = jnp.asarray(distribution.scale_tril)[jnp.ix_(indices, indices)]
            inverse = solve_triangular(chol, jnp.eye(dimension, dtype=chol.dtype), lower=True)
            precision = inverse.T @ inverse
            constant = -jnp.sum(jnp.log(jnp.diag(chol))) - 0.5 * dimension * jnp.log(2 * jnp.pi)
            return jnp.asarray(distribution.mean)[indices], precision, constant

        def exact() -> jax.Array:
            if not self.state_space.transition:
                return jnp.zeros((current.size + 1,), dtype=current.dtype)
            if current_gradient:
                mean, precision, constant = factors(previous)

                def density(state: jax.Array) -> jax.Array:
                    residual = self.project_state(context, state)[indices] - mean
                    return jnp.asarray(constant - 0.5 * residual @ precision @ residual)

                value, gradient = jax.value_and_grad(density)(current)
            else:

                def differentiated(state):
                    result = factors(state)
                    return result, result

                derivatives, (mean, precision, constant) = jax.jacfwd(differentiated, has_aux=True)(
                    previous
                )
                mean_jacobian, precision_jacobian, constant_gradient = derivatives
                residual = self.project_state(context, current)[indices] - mean
                score = precision @ residual
                value = constant - 0.5 * residual @ score
                gradient = (
                    mean_jacobian.T @ score
                    + constant_gradient
                    - 0.5 * jnp.einsum("i,ijk,j->k", residual, precision_jacobian, residual)
                )
            return jnp.concatenate((value[None], gradient))

        bridge = self.bridge
        if bridge is None:
            result = exact()
        else:
            value, gradient = jax.value_and_grad(
                bridge.reference.transition_log_prob, argnums=1 if current_gradient else 0
            )(previous, current, index)
            result = bridge.combine(exact, jnp.concatenate((value[None], gradient)))
        return result[0], result[1:]

    def transition_moments(
        self, context: ParticleContext, previous: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Expose the declared Gaussian law for exact maps and initialization geometry."""
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

            outer = jnp.einsum("ni,nj->nij", centered, centered).reshape(
                current.shape[0], dimension**2
            )

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
                # Keep covariance-only work inside the mapped row. JAX can then
                # share its quadratic across ancestors when covariance is shared,
                # without broadcasting precision into an N-by-D-by-D array first.
                # State-dependent covariance retains one exact quadratic per row.
                quadratic = outer @ precision.reshape(dimension**2)
                return constant + centered @ linear - 0.5 * quadratic

            return jax.vmap(row)(previous)

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
        observations = self._point_observations(observations)
        distribution = self.dynamical_model(context).observation_model(
            self.project_state(context, state), None, context[1][index]
        )

        def exact() -> jax.Array:
            return jnp.sum(distribution.log_prob(observations[index]))

        return exact() if self.bridge is None else self.bridge.observation(exact)

    def _point_observations(self, observations: jax.Array) -> jax.Array:
        return (
            observations
            if self.intervals is None
            else jnp.where(
                self.intervals.operator.point_like_mask(jnp.dtype(bool))[None, :],
                observations,
                jnp.nan,
            )
        )

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
