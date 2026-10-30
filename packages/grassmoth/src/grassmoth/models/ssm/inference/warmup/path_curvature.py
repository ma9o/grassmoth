"""Local positive path curvature shared by initialization and translation charts.

Dynamics are affine linearizations with their exact, possibly singular noise.
Reading curvature is clipped to PSD in predictor space before pulling it back
through the predictor Jacobian. Parameter cross terms differentiate means and
predictors, holding this geometry fixed: Gaussian covariance parameters therefore
have no mean cross term. These approximations shape coordinates and starts only.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from grassmoth.models.ssm.execution.affine import compose_affine_maps
from grassmoth.models.ssm.execution.innovations import conditional_gaussian_factors
from grassmoth.models.ssm.execution.observation_groups import compile_law_groups
from grassmoth.models.ssm.execution.observation_model import (
    grouped_density_with_noise,
    prepare_mean_density,
)
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
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric(hessian))
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


@dataclass(frozen=True, eq=False)
class PathCurvature:
    """One fixed Gauss--Newton geometry and its parameter-dependent affine means."""

    matrices: jax.Array
    covariances: jax.Array
    regression: jax.Array
    pinned_precision: jax.Array
    fixed: jax.Array
    predictors: jax.Array
    reading_jacobian: jax.Array
    reading_precision: jax.Array
    reading_gradient: jax.Array

    @property
    def conditioned_matrices(self) -> jax.Array:
        """Transition Jacobians after conditioning exact stochastic pins."""
        identity = jnp.eye(self.matrices.shape[-1], dtype=self.matrices.dtype)
        return jnp.where(self.fixed[:, :, None], 0.0, (identity - self.regression) @ self.matrices)

    @property
    def precision(self) -> jax.Array:
        """Point readings and the marginal density of stochastic pins."""
        jacobian = self.reading_jacobian
        observed = jnp.swapaxes(jacobian, -1, -2) @ self.reading_precision @ jacobian
        pinned = jnp.swapaxes(self.matrices, -1, -2) @ self.pinned_precision @ self.matrices
        return observed.at[:-1].add(pinned[1:])

    def mean(self, target: ParticleTarget, context: ParticleContext, path: jax.Array) -> jax.Array:
        """Smooth affine means; derivatives in context give the parameter cross solve.

        Covariances, transition Jacobians and reading quadratics belong to this
        frozen geometry. Only transition means and predictors vary with context.
        """
        initial = target.initial_moments(context)[0]
        tail = jax.vmap(
            lambda previous, index: target.transition_moments(context, previous, index)[0]
        )(path[:-1], jnp.arange(1, path.shape[0]))
        offsets = jnp.concatenate((initial[None], tail)) - jnp.einsum(
            "tij,tj->ti", self.matrices, jnp.concatenate((jnp.zeros_like(path[:1]), path[:-1]))
        )
        static = jnp.asarray(target.state_space.static, dtype=jnp.int32)
        fixed_values = path.at[:, static].set(context[2][static])
        pinned_shift = jnp.einsum(
            "tji,tjk,tk->ti", self.matrices, self.pinned_precision, fixed_values - offsets
        )
        conditioned_offsets = offsets + jnp.einsum(
            "tij,tj->ti", self.regression, fixed_values - offsets
        )
        conditioned_offsets = jnp.where(self.fixed, fixed_values, conditioned_offsets)
        residual = self.predictors - point_predictors(target, context, path)
        residual = residual + jnp.einsum("tij,tj->ti", self.reading_jacobian, path)
        observed_shift = jnp.einsum(
            "tji,tj->ti",
            self.reading_jacobian,
            self.reading_gradient + jnp.einsum("tij,tj->ti", self.reading_precision, residual),
        )
        information = GaussianTwist(self.precision, observed_shift.at[:-1].add(pinned_shift[1:]))
        return smoothing_mean(
            self.conditioned_matrices, conditioned_offsets, self.covariances, information
        )

    def covariance_product(self, rhs: jax.Array) -> jax.Array:
        """Apply the point-conditioned path covariance without inverting process noise."""
        return smoothing_mean(
            self.conditioned_matrices,
            jnp.zeros_like(rhs),
            self.covariances,
            GaussianTwist(self.precision, rhs),
        )


def linearize_path(
    target: ParticleTarget, context: ParticleContext, path: jax.Array
) -> PathCurvature:
    """Own path curvature, exact pin conditioning and PSD reading quadratics."""
    steps, dimension = path.shape
    indices = jnp.arange(steps)
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    initial_covariance = target.initial_moments(context)[1]
    tail_covariances = jax.vmap(
        lambda previous, index: target.transition_moments(context, previous, index)[1]
    )(path[:-1], indices[1:])
    covariances = jnp.concatenate((initial_covariance[None], tail_covariances))
    geometry = jax.vmap(
        lambda covariance, density, active: conditional_gaussian_factors(
            covariance, density_mask=density, innovation_mask=active
        )
    )(covariances, target.state_density_mask, free)
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
            jnp.broadcast_to(jnp.eye(dimension, dtype=path.dtype), covariances.shape),
        )
        * pinned[:, :, None]
        * pinned[:, None, :]
    )
    matrices = jax.vmap(
        jax.jacfwd(lambda previous, index: target.transition_moments(context, previous, index)[0])
    )(path[:-1], indices[1:])
    matrices = jnp.concatenate((jnp.zeros((1, dimension, dimension), dtype=path.dtype), matrices))
    static = (
        jnp.zeros(dimension, dtype=bool)
        .at[jnp.asarray(target.state_space.static + target.state_space.inputs, dtype=jnp.int32)]
        .set(True)
    )
    observation = cast(
        "HeterogeneousObservation", target.dynamical_model(context).observation_model
    )
    noise = target.prepare_noise_factors(context, target.observations).observations
    readings = target.observations
    if target.intervals is not None:
        readings = jnp.where(
            target.intervals.operator.point_like_mask(jnp.dtype(bool)), readings, jnp.nan
        )

    def point(state: jax.Array, index: jax.Array) -> tuple[jax.Array, ...]:
        def predictor(value: jax.Array) -> jax.Array:
            return observation.linear_predictor(target.project_state(context, value, index))

        eta = predictor(state)

        def density(value: jax.Array) -> jax.Array:
            return grouped_density_with_noise(observation, readings[index], value, noise.at(index))

        return (
            eta,
            jnp.asarray(jax.jacfwd(predictor)(state)),
            positive_curvature(-jax.hessian(density)(eta)),
            jnp.asarray(jax.grad(density)(eta)),
        )

    predictors, jacobian, precision, gradient = jax.vmap(point)(path, indices)
    return PathCurvature(
        matrices,
        conditioned_covariances,
        geometry.regression,
        pinned_precision,
        pinned | static[None],
        predictors,
        jacobian,
        precision,
        gradient,
    )


def interval_means(target: ParticleTarget, context: ParticleContext, path: jax.Array) -> jax.Array:
    """Project exact authored responses through the compiled overlapping windows."""
    assert target.intervals is not None
    observation = cast(
        "HeterogeneousObservation", target.dynamical_model(context).observation_model
    )
    responses = jax.vmap(
        lambda state, index: (
            observation(target.project_state(context, state, index), None, 0).response
        )
    )(path, jnp.arange(path.shape[0]))
    return target.intervals.operator.project_response_trajectory(responses)[0]


def interval_translation(
    target: ParticleTarget,
    position: jax.Array,
    path: jax.Array,
    curvature: PathCurvature,
    point_slopes: jax.Array,
) -> jax.Array:
    """Solve interval Gauss--Newton cross terms using the point path preconditioner.

    Woodbury's identity reduces the system to I + W½ L C Lᵀ W½, where C is
    the singular point-conditioned path covariance. CG uses only products of
    the exact window operator's Jacobian L, never a dense time-by-time matrix.
    This is the block-tridiagonal path preconditioner in reading coordinates;
    deterministic stages retain their zero process noise and full propagation.
    A failed residual check produces non-finite slopes for the typed boundary.
    """
    assert target.intervals is not None
    context = target.context(position, target.times)
    observation = cast(
        "HeterogeneousObservation", target.dynamical_model(context).observation_model
    )
    groups = compile_law_groups(
        observation.laws, target.intervals.operator.interval_summary_indices, interval=True
    )
    means, project = jax.linearize(lambda value: interval_means(target, context, value), path)
    transpose = jax.linear_transpose(project, jnp.zeros_like(path))
    readings = target.observations

    def root(mean: jax.Array, reading: jax.Array) -> jax.Array:
        density = prepare_mean_density(
            groups,
            jnp.nan_to_num(reading, nan=0.0),
            observation.measurement.manifest_cov,
            ~jnp.isnan(reading),
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(positive_curvature(-jax.hessian(density)(mean)))
        return symmetric((eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))) @ eigenvectors.T)

    roots = jax.vmap(root)(means, readings)
    parameter_means = jax.jacfwd(
        lambda value: interval_means(target, target.context(value, target.times), path)
    )(position)

    def weight(value: jax.Array) -> jax.Array:
        return jnp.einsum("tij,tj->ti", roots, value)

    def correction(value: jax.Array) -> jax.Array:
        return curvature.covariance_product(transpose(weight(value))[0])

    def system(value: jax.Array) -> jax.Array:
        return value + weight(project(correction(value)))

    def column(slope: jax.Array, parameter_mean: jax.Array) -> jax.Array:
        rhs = weight(project(slope) + parameter_mean)
        solved, _ = cg(system, rhs, tol=1e-5, atol=1e-6, maxiter=256)
        error = jnp.linalg.norm(system(solved) - rhs)
        converged = error <= 5e-5 * jnp.linalg.norm(rhs) + 5e-6
        result = slope - correction(solved)
        return jnp.where(converged, result, jnp.full_like(result, jnp.nan))

    return jax.vmap(column, in_axes=(2, 2), out_axes=2)(point_slopes, parameter_means)
