"""Local positive path curvature for initialization and path proposals.

Dynamics are affine linearizations with their exact, possibly singular noise.
Window memory joins the state as noise-free coordinates whose exact accumulators
depend only on earlier states, so every interval reading becomes a factor of one
augmented time point. Reading curvature is clipped to PSD in predictor space before
pulling it back through the predictor Jacobian. Chart mean responses also differentiate transition noise and its pin-conditioning
algebra, while keeping dynamics Jacobians and reading quadratics frozen. These approximations shape coordinates, starts
and proposals only; auxiliary Kalman moves correct against the exact target.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.covariance import symmetric_eigh
from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.execution.innovations import conditional_gaussian_factors
from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import (
    grouped_density_with_noise,
    prepare_mean_density,
)
from grassmoth.models.ssm.inference.samplers.particle.kalman import GaussianPathTerms
from grassmoth.models.ssm.inference.samplers.particle.twist import (
    GaussianTwist,
    future_information,
    symmetric,
)

if TYPE_CHECKING:
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation

from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from grassmoth.models.ssm.shapes import Array, Float


def positive_curvature(hessian: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Project a reading's negative log-density Hessian onto the PSD cone."""
    eigenvalues, eigenvectors = symmetric_eigh(symmetric(hessian))
    return symmetric((eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T)


def smoothing_mean(
    matrices: jax.Array,
    offsets: jax.Array,
    covariances: jax.Array,
    information: GaussianTwist,
) -> jax.Array:
    """Solve the Gaussian path system using I + Q J, including deterministic stages."""
    future = future_information(matrices[1:], offsets[1:], covariances[1:], information)
    precision = information.precision + future.precision
    shift = information.shift + future.shift
    identity = jnp.eye(offsets.shape[-1], dtype=offsets.dtype)
    systems = identity + covariances @ precision
    conditioned_matrices = jnp.linalg.solve(systems, matrices)
    conditioned_offsets = jnp.linalg.solve(
        systems, (offsets + jnp.einsum("tij,tj->ti", covariances, shift))[..., None]
    )[..., 0]
    _, means = jax.lax.associative_scan(
        compose_affine_maps, (conditioned_matrices, conditioned_offsets)
    )
    return jnp.asarray(means)


def point_predictors(
    target: ParticleTarget, context: ParticleContext, path: jax.Array
) -> jax.Array:
    """Evaluate authored predictors, including static parameter coordinates."""
    observation = cast(
        "HeterogeneousObservation", target.dynamical_model(context).observation_model
    )
    return jax.vmap(
        lambda state, index: observation.linear_predictor(
            target.project_state(context, state, index)
        )
    )(path, jnp.arange(path.shape[0]))


def _augmented_mean(
    target: ParticleTarget, context: ParticleContext, previous: jax.Array, index: jax.Array
) -> jax.Array:
    """Exact transition mean of the state followed by its exact window memory."""
    size = target.state_space.size
    mean = target.transition_moments(context, previous[:size], index)[0]
    if target.intervals is None:
        return mean
    memory = target.intervals.advance(
        context, previous[size:].reshape(target.intervals.memory.shape), previous[:size], index
    )
    return jnp.concatenate((mean, memory.reshape(-1)))


def _interval_predictor(
    target: ParticleTarget, context: ParticleContext, value: jax.Array, index: jax.Array
) -> jax.Array:
    """Means of the windows emitted at this time, as functions of state and memory."""
    assert target.intervals is not None
    size = target.state_space.size
    means, _ = target.intervals.expected_means(
        context, value[size:].reshape(target.intervals.memory.shape), value[:size], index
    )
    return means[jnp.asarray(target.intervals.memory.channels, dtype=jnp.int32)]


def _predictors(target: ParticleTarget, context: ParticleContext, path: jax.Array) -> jax.Array:
    """Point predictors of the state, then emitted window means of the augmented state."""
    point = point_predictors(target, context, path[:, : target.state_space.size])
    if target.intervals is None:
        return point
    windows = jax.vmap(lambda value, index: _interval_predictor(target, context, value, index))(
        path, jnp.arange(path.shape[0])
    )
    return jnp.concatenate((point, windows), axis=-1)


@dataclass(frozen=True, eq=False)
class PathCurvature:
    """One fixed Gauss--Newton geometry over the state and its window memory.

    Arrays span the augmented coordinates: the state first, then flattened memory.
    ``memory`` is the exact window memory at the linearization point.
    """

    matrices: jax.Array
    covariances: jax.Array
    regression: jax.Array
    pinned_precision: jax.Array
    fixed: jax.Array
    memory: jax.Array
    predictors: jax.Array
    reading_jacobian: jax.Array
    reading_precision: jax.Array
    reading_gradient: jax.Array

    def augment(self, values: jax.Array, *, fill: float | bool) -> jax.Array:
        """Extend per-time state values with constant entries for the memory coordinates."""
        padding = jnp.full((*values.shape[:-1], self.memory.shape[-1]), fill, dtype=values.dtype)
        return jnp.concatenate((values, padding), axis=-1)

    @property
    def conditioned_matrices(self) -> jax.Array:
        """Transition Jacobians after conditioning exact stochastic pins."""
        identity = jnp.eye(self.matrices.shape[-1], dtype=self.matrices.dtype)
        return jnp.where(self.fixed[:, :, None], 0.0, (identity - self.regression) @ self.matrices)

    @property
    def precision(self) -> jax.Array:
        """Readings and the marginal density of stochastic pins."""
        jacobian = self.reading_jacobian
        observed = jnp.swapaxes(jacobian, -1, -2) @ self.reading_precision @ jacobian
        pinned = jnp.swapaxes(self.matrices, -1, -2) @ self.pinned_precision @ self.matrices
        return observed.at[:-1].add(pinned[1:])

    def proposal_terms(
        self, target: ParticleTarget, context: ParticleContext, path: jax.Array
    ) -> GaussianPathTerms:
        """Form affine means; derivatives in context give the parameter cross solve.

        Covariances, transition Jacobians and reading quadratics belong to this
        frozen geometry, as does the memory at the linearization point. Only
        transition means and predictors vary with context.
        """
        augmented = jnp.concatenate((path, self.memory), axis=-1)
        initial = jnp.concatenate(
            (target.initial_moments(context)[0], jnp.zeros_like(self.memory[0]))
        )
        tail = jax.vmap(lambda previous, index: _augmented_mean(target, context, previous, index))(
            augmented[:-1], jnp.arange(1, path.shape[0])
        )
        offsets = jnp.concatenate((initial[None], tail)) - jnp.einsum(
            "tij,tj->ti",
            self.matrices,
            jnp.concatenate((jnp.zeros_like(augmented[:1]), augmented[:-1])),
        )
        static = jnp.asarray(target.state_space.static, dtype=jnp.int32)
        fixed_values = augmented.at[:, static].set(context[2][static])
        pinned_shift = jnp.einsum(
            "tji,tjk,tk->ti", self.matrices, self.pinned_precision, fixed_values - offsets
        )
        conditioned_offsets = offsets + jnp.einsum(
            "tij,tj->ti", self.regression, fixed_values - offsets
        )
        conditioned_offsets = jnp.where(self.fixed, fixed_values, conditioned_offsets)
        residual = self.predictors - _predictors(target, context, augmented)
        residual = residual + jnp.einsum("tij,tj->ti", self.reading_jacobian, augmented)
        observed_shift = jnp.einsum(
            "tji,tj->ti",
            self.reading_jacobian,
            self.reading_gradient + jnp.einsum("tij,tj->ti", self.reading_precision, residual),
        )
        information = GaussianTwist(self.precision, observed_shift.at[:-1].add(pinned_shift[1:]))
        return GaussianPathTerms(
            self.conditioned_matrices, conditioned_offsets, self.covariances, information
        )

    def mean(self, target: ParticleTarget, context: ParticleContext, path: jax.Array) -> jax.Array:
        """Smooth the shared Gaussian proposal terms and return the state coordinates."""
        return smoothing_mean(*self.proposal_terms(target, context, path))[:, : path.shape[-1]]


def _block_diagonal(first: jax.Array, second: jax.Array) -> jax.Array:
    """Place two square blocks on a diagonal, keeping the first block's leading axes."""
    upper = jnp.concatenate(
        (first, jnp.zeros((*first.shape[:-1], second.shape[-1]), dtype=first.dtype)), axis=-1
    )
    lower = jnp.concatenate(
        (jnp.zeros((*second.shape[:-1], first.shape[-1]), dtype=first.dtype), second), axis=-1
    )
    return jnp.concatenate((upper, lower), axis=-2)


def _path_noise(
    target: ParticleTarget,
    context: ParticleContext,
    path: jax.Array,
    memory: jax.Array,
    initial_covariance: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Condition supported transition noise, retaining stochastic pin information."""
    steps = path.shape[0]
    indices = jnp.arange(steps)
    size = path.shape[1] + memory.shape[-1]

    def pad(values: jax.Array) -> jax.Array:
        return jnp.concatenate(
            (values, jnp.zeros((*values.shape[:-1], memory.shape[-1]), dtype=values.dtype)),
            axis=-1,
        )

    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    free, density_mask = pad(free), pad(target.state_density_mask)
    tail_covariances = jax.vmap(
        lambda previous, index: target.transition_moments(context, previous, index)[1]
    )(path[:-1], indices[1:])
    covariances = jnp.concatenate((initial_covariance[None], tail_covariances))
    covariances = _block_diagonal(
        covariances, jnp.zeros((steps, memory.shape[-1], memory.shape[-1]), dtype=path.dtype)
    )
    geometry = jax.vmap(
        lambda covariance, density, active: conditional_gaussian_factors(
            covariance, density_mask=density, innovation_mask=active
        )
    )(covariances, density_mask, free)
    conditioned_covariances = geometry.chol @ jnp.swapaxes(geometry.chol, -1, -2)
    conditioned_covariances = jnp.where(
        free[:, :, None] & free[:, None, :], conditioned_covariances, 0.0
    )
    pinned = geometry.centred_mask
    pinned_covariances = jnp.where(pinned[:, :, None] & pinned[:, None, :], covariances, 0.0)
    pinned_covariances = pinned_covariances + jax.vmap(jnp.diag)((~pinned).astype(path.dtype))
    pinned_precision = (
        jnp.linalg.solve(
            pinned_covariances,
            jnp.broadcast_to(jnp.eye(size, dtype=path.dtype), covariances.shape),
        )
        * pinned[:, :, None]
        * pinned[:, None, :]
    )
    return conditioned_covariances, geometry.regression, jnp.asarray(pinned_precision), pinned


def linearize_path(
    target: ParticleTarget,
    context: ParticleContext,
    path: jax.Array,
    log_precisions: jax.Array | None = None,
) -> PathCurvature:
    """Own path curvature, exact pin conditioning and PSD reading quadratics.

    The linearization point is the supplied path with its exact window memory.
    Memory coordinates carry no noise and are never free; the readings they emit
    supply information to the earlier states that fed each window. Supplied
    precisions replace each StudentT reading with its conditional Gaussian.
    """
    steps, dimension = path.shape
    indices = jnp.arange(steps)
    intervals = target.intervals
    memory = (
        jnp.zeros((steps, 0), dtype=path.dtype)
        if intervals is None
        else intervals.memory_path(context, path).reshape(steps, -1)
    )
    size = dimension + memory.shape[-1]
    augmented = jnp.concatenate((path, memory), axis=-1)

    initial_covariance = target.initial_moments(context)[1]
    conditioned_covariances, regression, pinned_precision, pinned = _path_noise(
        target, context, path, memory, initial_covariance
    )
    matrices = jax.vmap(
        jax.jacfwd(lambda previous, index: _augmented_mean(target, context, previous, index))
    )(augmented[:-1], indices[1:])
    matrices = jnp.concatenate((jnp.zeros((1, size, size), dtype=path.dtype), matrices))
    static = (
        jnp.zeros(size, dtype=bool)
        .at[jnp.asarray(target.state_space.static + target.state_space.inputs, dtype=jnp.int32)]
        .set(True)
    )
    observation = cast(
        "HeterogeneousObservation", target.dynamical_model(context).observation_model
    )
    noise = target.prepare_noise_factors(context, target.observations).observations
    readings = target.observations
    if intervals is not None:
        readings = jnp.where(intervals.operator.point_like_mask(jnp.dtype(bool)), readings, jnp.nan)

    def point(value: jax.Array, index: jax.Array) -> tuple[jax.Array, ...]:
        def predictor(value: jax.Array) -> jax.Array:
            return observation.linear_predictor(
                target.project_state(context, value[:dimension], index)
            )

        eta = predictor(value)

        def density(eta: jax.Array) -> jax.Array:
            return grouped_density_with_noise(
                observation,
                readings[index],
                eta,
                noise.at(index),
                None if log_precisions is None else log_precisions[index],
            )

        return (
            eta,
            jnp.asarray(jax.jacfwd(predictor)(value)),
            positive_curvature(-jax.hessian(density)(eta)),
            jnp.asarray(jax.grad(density)(eta)),
        )

    predictors, jacobian, precision, gradient = jax.vmap(point)(augmented, indices)
    if intervals is not None:
        channels = jnp.asarray(intervals.memory.channels, dtype=jnp.int32)
        groups = compile_law_groups(observation.laws, intervals.memory.channels, interval=True)

        def window(value: jax.Array, index: jax.Array) -> tuple[jax.Array, ...]:
            def predictor(value: jax.Array) -> jax.Array:
                return _interval_predictor(target, context, value, index)

            means = predictor(value)
            reading = target.observations[index]
            emitted = intervals.expected_means(
                context,
                value[dimension:].reshape(intervals.memory.shape),
                value[:dimension],
                index,
            )[1]
            score = prepare_mean_density(
                groups,
                jnp.nan_to_num(reading, nan=0.0),
                observation.measurement.manifest_cov,
                emitted & jnp.isfinite(reading),
                None if log_precisions is None else log_precisions[index],
            )

            def density(means: jax.Array) -> jax.Array:
                return score(jnp.zeros_like(reading).at[channels].set(means))

            return (
                means,
                jnp.asarray(jax.jacfwd(predictor)(value)),
                positive_curvature(-jax.hessian(density)(means)),
                jnp.asarray(jax.grad(density)(means)),
            )

        windows = jax.vmap(window)(augmented, indices)
        predictors = jnp.concatenate((predictors, windows[0]), axis=-1)
        jacobian = jnp.concatenate((jacobian, windows[1]), axis=-2)
        precision = _block_diagonal(precision, windows[2])
        gradient = jnp.concatenate((gradient, windows[3]), axis=-1)
    return PathCurvature(
        matrices,
        conditioned_covariances,
        regression,
        pinned_precision,
        pinned | static[None],
        memory,
        predictors,
        jacobian,
        precision,
        gradient,
    )
