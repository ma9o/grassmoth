"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from grassmoth.models.ssm.compile.state_space import (
    AdditiveRecurrences,
    SequentialRecurrences,
)
from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.execution.gaussian_noise import (
    DiagonalGaussianNoiseFactors,
    GaussianNoiseFactors,
    PreparedGaussianNoise,
)
from grassmoth.models.ssm.execution.innovations import GaussianPathMap
from grassmoth.models.ssm.execution.observation_model import (
    ObservationDensityFactors,
    grouped_density_with_noise,
    prepare_grouped_density,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from grassmoth.models.ssm.compile.state_space import StateSpace
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.state_transition import SupportedEulerMaruyama
    from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
    from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
    from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array, jax.Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ParticleNoiseFactors:
    """Exact noise factors for one fixed parameter context and its observation grid."""

    transitions: PreparedGaussianNoise
    observations: ObservationDensityFactors


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
    output_coordinates: tuple[int, ...]
    point_reading_mask: jax.Array
    intervals: ParticleIntervalObservations | None = None
    exact_constraints: ExactStateConstraints | None = None

    @property
    def coupled_recurrences(self) -> bool:
        """Whether a replayed coordinate can affect later exact or proposal factors."""
        return any(
            index not in self.output_coordinates
            for stage in self.state_space.recurrence
            for index in stage.indices
        )

    def project_state(
        self, context: ParticleContext, state: jax.Array, _index: jax.Array | int | None = None
    ) -> jax.Array:
        """Bind global static coordinates before evaluating a local model factor at these times."""
        indices = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        return state.at[..., indices].set(context[2][indices])

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
        fixed = context[2]
        return state.at[inputs].set(coordinates[inputs]).at[static].set(fixed[static])

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
        fixed = context[2]
        return state.at[inputs].set(coordinates[inputs]).at[static].set(fixed[static])

    def replay_path(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct exact initial values, static quantities and deterministic recurrences."""
        if self.exact_constraints is not None:
            coordinates = self.exact_constraints.project(coordinates)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        fixed = context[2]
        coordinates = coordinates.at[:, static].set(fixed[static])
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
                    state = self.project_state(context, previous, index - 1)
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
                            jnp.asarray(
                                self._transition_distribution(
                                    context, zero.at[deterministic].set(unit), index
                                ).mean
                            )[deterministic]
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
        return self.parameters.log_prior(position)

    def initial_moments(self, context):
        """Return the exact model's Gaussian initial-state mean and covariance for this context."""
        distribution = self.dynamical_model(context).initial_condition
        assert isinstance(
            distribution, dist.MultivariateNormal
        )  # The exact model producer declares a Gaussian initial law.
        return (jnp.asarray(distribution.mean)), (jnp.asarray(distribution.covariance_matrix))

    def initial_log_prob(self, context, state):
        """Evaluate initial-state density on the modeled coordinates of the supplied state."""
        state = self.project_state(context, state, 0)
        return self._modeled_log_prob(
            self.dynamical_model(context).initial_condition, state, self.state_space.initial
        )

    def _modeled_log_prob(self, distribution, state, density_indices):
        if not density_indices:
            return jnp.asarray(0.0, dtype=state.dtype)
        if len(density_indices) == 1:
            # Exact scalar Gaussian arithmetic avoids launching batched 1x1
            # Cholesky and triangular-solve GPU kernels at every leapfrog.
            index = density_indices[0]
            variance = distribution.covariance_matrix[index, index]
            residual = state[index] - distribution.mean[index]
            return -0.5 * (residual**2 / variance + jnp.log(2 * jnp.pi * variance))
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

        def transition(index: jax.Array) -> PreparedGaussianNoise:
            distribution = cast(
                "dist.MultivariateNormal", self._transition_distribution(context, initial, index)
            )
            chol = jnp.asarray(distribution.scale_tril)[jnp.ix_(indices, indices)]
            if self.state_space.transition_covariance == "diagonal":
                return DiagonalGaussianNoiseFactors.from_scale(jnp.diag(chol))
            return GaussianNoiseFactors.from_cholesky(chol)

        transitions = jax.vmap(transition)(jnp.arange(1, context[1].size))
        # The unused origin row also makes a traced length-zero scan well shaped.
        transitions = jax.tree.map(
            lambda value: jnp.concatenate((jnp.zeros((1, *value.shape[1:]), value.dtype), value)),
            transitions,
        )
        model = cast("HeterogeneousObservation", self.dynamical_model(context).observation_model)
        return ParticleNoiseFactors(
            transitions,
            prepare_grouped_density(
                model.density_groups(),
                model.measurement.manifest_cov,
                self._point_observations(observations),
                model.structure.covariance,
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
        return self._transition_log_prob_using_mean(
            context,
            noise,
            current,
            index,
            lambda: self._transition_distribution(context, previous, index).mean[
                jnp.asarray(self.state_space.transition, dtype=jnp.int32)
            ],
        )

    def prepare_transition_mean(
        self, context: ParticleContext, previous: jax.Array, index: jax.Array
    ) -> jax.Array:
        """Evaluate the exact nonlinear mean before repeated successor selection."""
        indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)
        return jnp.asarray(self._transition_distribution(context, previous, index).mean)[indices]

    def transition_log_prob_with_mean(
        self,
        context: ParticleContext,
        noise: ParticleNoiseFactors,
        current: jax.Array,
        index: jax.Array,
        mean: jax.Array,
    ) -> jax.Array:
        """Score the same exact transition with its already-evaluated nonlinear mean."""
        return self._transition_log_prob_using_mean(context, noise, current, index, lambda: mean)

    def _transition_log_prob_using_mean(
        self,
        context: ParticleContext,
        noise: ParticleNoiseFactors,
        current: jax.Array,
        index: jax.Array | int,
        mean: Callable[[], jax.Array],
    ) -> jax.Array:
        current = self.project_state(context, current, index)
        indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)
        return noise.transitions.at(index).log_prob(current[indices] - mean())

    def observation_increment_with_noise(
        self,
        context: ParticleContext,
        noise: ParticleNoiseFactors,
        state: jax.Array,
        index: jax.Array | int,
        log_precision: jax.Array | None = None,
    ) -> jax.Array:
        """Score exact point laws and nonlinear predictors with prepared Gaussian noise."""
        model = cast("HeterogeneousObservation", self.dynamical_model(context).observation_model)
        predictor = model.linear_predictor(self.project_state(context, state, index))
        return grouped_density_with_noise(
            model,
            self._point_observations(self.observations)[index],
            predictor,
            noise.observations.at(index),
            log_precision,
        ).astype(predictor.dtype)

    def _transition_distribution(self, context, previous, index):
        times = context[1]
        evolution = cast("DiscreteStateTransition", self.dynamical_model(context).state_evolution)
        return evolution(
            self.project_state(context, previous, jnp.maximum(jnp.asarray(index) - 1, 0)),
            None,
            times[jnp.maximum(index - 1, 0)],
            times[index],
        )

    def transition_log_prob(self, context, previous, current, index):
        """Evaluate the current state's modeled transition density from its predecessor."""
        current = self.project_state(context, current, index)
        return self._modeled_log_prob(
            self._transition_distribution(context, previous, index),
            current,
            self.state_space.transition,
        )

    def transition_moments(
        self, context: ParticleContext, previous: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Expose the declared Gaussian law for exact maps and initialization geometry."""
        distribution = cast(
            "dist.MultivariateNormal", self._transition_distribution(context, previous, index)
        )
        return (jnp.asarray(distribution.mean)), (jnp.asarray(distribution.covariance_matrix))

    def initial_path(self, context, *, exact_constraints: ExactStateConstraints | None = None):
        """Build an initialization path from successive transition means and any exact state constraints."""
        initial = self.initial_moments(context)[0]
        initial = self.project_state(context, initial, 0)
        if exact_constraints is not None:
            initial = jnp.where(
                exact_constraints.free_mask[0],
                initial,
                (exact_constraints.values[0]),
            )

        def step(previous, index):
            current = self._transition_distribution(context, previous, index).mean
            if exact_constraints is not None:
                current = jnp.where(
                    exact_constraints.free_mask[index],
                    current,
                    (exact_constraints.values[index]),
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
        """Score point factors at one time; interval factors are scored over their windows."""
        observations = self._point_observations(observations)
        distribution = self.dynamical_model(context).observation_model(
            self.project_state(context, state, index), None, context[1][index]
        )
        return jnp.sum(distribution.log_prob(observations[index]))

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
