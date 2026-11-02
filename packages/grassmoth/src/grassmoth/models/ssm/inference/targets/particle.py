"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from grassmoth.models.ssm.execution.gaussian_noise import GaussianNoiseFactors
from grassmoth.models.ssm.execution.observation_model import (
    ObservationDensityFactors,
    grouped_density_with_noise,
    prepare_grouped_density,
)
from grassmoth.models.ssm.inference.targets.coordinates import (
    ScaledDeviationFrame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from grassmoth.models.ssm.compile.state_space import StateSpace
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.state_transition import SupportedEulerMaruyama
    from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
    from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
    from grassmoth.models.ssm.inference.targets.coordinates import PathChart
    from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
# The last leaf is the label's chart: a translation offset or a scaled-deviation frame.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array, jax.Array, ScaledDeviationFrame | None]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ParticleNoiseFactors:
    """Exact noise factors for one fixed parameter context and its observation grid."""

    transitions: GaussianNoiseFactors
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
    intervals: ParticleIntervalObservations | None = None
    exact_constraints: ExactStateConstraints | None = None

    def in_coordinates(self, chart: PathChart) -> ParticleTarget:
        """Compose an exact chart beneath every particle density and proposal.

        Each ensemble label gets its own physical map. The common particle bank and
        auxiliaries remain in chart coordinates. No model factor or parameter prior
        is approximated. A translation needs no Jacobian correction; scaled
        deviations pay their exact log-Jacobian in every initial and transition
        density, and their proposal moments are expressed in chart units.
        """

        def chart_context(position: jax.Array, times: jax.Array) -> ParticleContext:
            model, grid, static, _ = self.context(position, times)
            frame = chart.frame(position)
            return model, grid, static, frame

        target = replace(self, context=chart_context)
        return replace(
            target,
            intervals=None
            if self.intervals is None
            else replace(self.intervals, state_projection=target.project_state),
        )

    def decode_state(
        self, context: ParticleContext, state: jax.Array, index: jax.Array | int | None = None
    ) -> jax.Array:
        """Map chart coordinates at these times to the exact physical model coordinates."""
        frame = context[3]
        if frame is None:
            return state
        assert index is not None, "Path charts decode each state at its own time."
        return frame.decode(state, index)

    def encode_state(
        self, context: ParticleContext, state: jax.Array, index: jax.Array | int | None = None
    ) -> jax.Array:
        """Map physical model coordinates at these times to the current frozen chart."""
        frame = context[3]
        if frame is None:
            return state
        assert index is not None, "Path charts encode each state at its own time."
        return frame.encode(state, index)

    def _encode_covariance(self, context: ParticleContext, covariance: jax.Array) -> jax.Array:
        """Express a physical Gaussian covariance in chart units."""
        frame = context[3]
        if not isinstance(frame, ScaledDeviationFrame):
            return covariance
        inverse = jnp.exp(-frame.log_scale)
        return covariance * inverse[:, None] * inverse[None, :]

    def _chart_log_jacobian(
        self, context: ParticleContext, coordinates: tuple[int, ...]
    ) -> jax.Array | float:
        """Log-determinant of one time slice's chart map over the modeled coordinates."""
        frame = context[3]
        if not isinstance(frame, ScaledDeviationFrame) or not coordinates:
            return 0.0
        return jnp.sum(frame.log_scale[jnp.asarray(coordinates, dtype=jnp.int32)])

    def diffusion_scale_exponents(self, reference: jax.Array) -> jax.Array:
        """Differentiate each stochastic state's log diffusion scale at a parameter position.

        Scaled-deviation charts use these slopes as structural exponents. Any exponents
        keep the target exact, so a state-dependent or nonlinear diffusion only costs
        mixing through its local slope.
        """
        exponents = jnp.zeros((self.state_space.size, reference.size), dtype=reference.dtype)
        if reference.size == 0 or self.times.size < 2 or not self.state_space.transition:
            return exponents
        indices = jnp.asarray(self.state_space.transition, dtype=jnp.int32)

        def log_scale(position: jax.Array) -> jax.Array:
            context = self.context(position, self.times)
            covariance = self.transition_moments(
                context, self.initial_moments(context)[0], jnp.asarray(1)
            )[1]
            return 0.5 * jnp.log(jnp.diagonal(covariance)[indices])

        return exponents.at[indices].set(jax.jacfwd(log_scale)(reference))

    def project_state(
        self, context: ParticleContext, state: jax.Array, index: jax.Array | int | None = None
    ) -> jax.Array:
        """Bind global static coordinates before evaluating a local model factor at these times."""
        state = self.decode_state(context, state, index)
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
        fixed = self.encode_state(context, context[2], 0)
        return state.at[inputs].set(coordinates[inputs]).at[static].set(fixed[static])

    def replay_step(
        self,
        context: ParticleContext,
        previous: jax.Array,
        coordinates: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Advance exact recurrences with their declared previous-time dependencies."""
        mean = self.encode_state(
            context, self._transition_distribution(context, previous, index).mean, index
        )
        state = jnp.where(self.independent_mask[index], coordinates, mean)
        inputs = jnp.asarray(self.state_space.inputs, dtype=jnp.int32)
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        fixed = self.encode_state(context, context[2], index)
        return state.at[inputs].set(coordinates[inputs]).at[static].set(fixed[static])

    def replay_path(self, context: ParticleContext, coordinates: jax.Array) -> jax.Array:
        """Reconstruct exact initial values, static quantities and deterministic recurrences."""
        times = jnp.arange(coordinates.shape[0])
        if self.exact_constraints is not None:
            coordinates = self.encode_state(
                context,
                self.exact_constraints.project(self.decode_state(context, coordinates, times)),
                times,
            )
        static = jnp.asarray(self.state_space.static, dtype=jnp.int32)
        fixed = self.encode_state(context, context[2], 0)
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
                    offset = self.encode_state(
                        context, self._transition_distribution(context, zero, index).mean, index
                    )[deterministic]
                    # The compiler proved affine dependence. Basis evaluations recover
                    # its exact coefficients, not a local approximation of nonlinear drift.
                    basis = jnp.eye(deterministic.size, dtype=previous.dtype)
                    columns = jax.vmap(
                        lambda unit: (
                            self.encode_state(
                                context,
                                self._transition_distribution(
                                    context, zero.at[deterministic].set(unit), index
                                ).mean,
                                index,
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
        return self.encode_state(
            context, jnp.asarray(distribution.mean), 0
        ), self._encode_covariance(context, jnp.asarray(distribution.covariance_matrix))

    def initial_log_prob(self, context, state):
        """Evaluate initial-state density on the modeled coordinates of the supplied state."""
        state = self.project_state(context, state, 0)
        density = self._modeled_log_prob(
            self.dynamical_model(context).initial_condition, state, self.state_space.initial
        )
        return density + self._chart_log_jacobian(context, self.state_space.initial)

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
            prepare_grouped_density(
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
        density = noise.transitions.at(index).log_prob(current[indices] - mean())
        return density + self._chart_log_jacobian(context, self.state_space.transition)

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
        density = self._modeled_log_prob(
            self._transition_distribution(context, previous, index),
            current,
            self.state_space.transition,
        )
        return density + self._chart_log_jacobian(context, self.state_space.transition)

    def transition_moments(
        self, context: ParticleContext, previous: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Expose the declared Gaussian law for exact maps and initialization geometry."""
        distribution = cast(
            "dist.MultivariateNormal", self._transition_distribution(context, previous, index)
        )
        return self.encode_state(
            context, jnp.asarray(distribution.mean), index
        ), self._encode_covariance(context, jnp.asarray(distribution.covariance_matrix))

    def initial_path(self, context, *, exact_constraints: ExactStateConstraints | None = None):
        """Build an initialization path from successive transition means and any exact state constraints."""
        initial = self.initial_moments(context)[0]
        initial = self.encode_state(context, self.project_state(context, initial, 0), 0)
        if exact_constraints is not None:
            initial = jnp.where(
                exact_constraints.free_mask[0],
                initial,
                self.encode_state(context, exact_constraints.values[0], 0),
            )

        def step(previous, index):
            current = self.encode_state(
                context, self._transition_distribution(context, previous, index).mean, index
            )
            if exact_constraints is not None:
                current = jnp.where(
                    exact_constraints.free_mask[index],
                    current,
                    self.encode_state(context, exact_constraints.values[index], index),
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
