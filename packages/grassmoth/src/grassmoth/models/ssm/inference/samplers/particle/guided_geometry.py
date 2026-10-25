"""Point or singular augmented-memory proposal geometry.

Only the guide linearizes; the conditional filter owns the true reading,
transition and auxiliary factors. Sampling and normalized proposal density
share the same covariance factor and affine mean map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.artifacts.sampling import TwistedParticleAGradSpec
from grassmoth.models.ssm.inference.samplers.particle.guided_information import (
    memory_information,
    point_information,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import (
    GaussianTwist,
    future_information,
    symmetric,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from grassmoth.artifacts.sampling import GuidedPathSpec
    from grassmoth.models.ssm.inference.samplers.particle.guided_information import InformationRule
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime


type GuideGeometry = Literal["point", "memory"]


def _free_covariance(covariance: jax.Array, mask: jax.Array) -> jax.Array:
    """Embed the free marginal with unit dummy coordinates of zero log-density cost."""
    return covariance * mask[:, None] * mask[None, :] + jnp.diag((~mask).astype(covariance.dtype))


class _ProposalGeometry(NamedTuple):
    """One covariance factor and affine mean map per time, shared by all particles."""

    mean_map: jax.Array
    shift: jax.Array
    chol: jax.Array
    whiten: jax.Array
    log_normalizer: jax.Array
    parent_map: jax.Array

    def center(self, prior_mean: jax.Array, mask: jax.Array) -> jax.Array:
        return (
            jnp.sum(jnp.where(mask, prior_mean, 0.0)[..., None, :] * self.mean_map, axis=-1)
            + self.shift
        )

    def initial_particles(
        self,
        center: jax.Array,
        key: jax.Array,
        reference: jax.Array,
        mask: jax.Array,
        count: int,
    ) -> jax.Array:
        """Draw the initial bank, preserving the filter's original evaluation order."""
        dimension = reference.shape[1]
        return jnp.concatenate(
            (
                reference[0, None],
                jnp.where(
                    mask[0],
                    center
                    + jnp.sum(
                        jax.random.normal(key, (count - 1, dimension), reference.dtype)[:, None, :]
                        * self.chol,
                        axis=-1,
                    ),
                    reference[0],
                ),
            )
        )

    def next_particles(
        self,
        centers: jax.Array,
        noise: jax.Array,
        reference: jax.Array,
        mask: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Draw a forward bank with the factor also used by log_density."""
        return jnp.concatenate(
            (
                reference[index, None],
                jnp.where(
                    mask[index],
                    centers[1:] + jnp.sum(noise[:, None, :] * self.chol, axis=-1),
                    reference[index],
                ),
            )
        )

    def log_density(self, difference: jax.Array, mask: jax.Array) -> jax.Array:
        whitened = jnp.sum(jnp.where(mask, difference, 0.0)[..., None, :] * self.whiten, axis=-1)
        return self.log_normalizer - 0.5 * jnp.sum(whitened**2, axis=-1)


def _proposal_geometry(
    covariance: jax.Array,
    local: GaussianTwist,
    mask: jax.Array,
    precision: jax.Array,
    shift: jax.Array,
) -> _ProposalGeometry:
    identity = jnp.eye(local.shift.size, dtype=local.shift.dtype)
    prior_precision = jnp.linalg.solve(_free_covariance(covariance, mask), identity)
    posterior_precision = prior_precision + local.precision + precision
    posterior_covariance = symmetric(jnp.linalg.solve(posterior_precision, identity))
    chol = jnp.linalg.cholesky(posterior_covariance)
    return _ProposalGeometry(
        posterior_covariance @ prior_precision,
        posterior_covariance @ (local.shift + shift),
        chol,
        jax.scipy.linalg.solve_triangular(chol, identity, lower=True),
        -0.5 * jnp.sum(mask) * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol))),
        jnp.zeros_like(covariance),
    )


def _build_twist(
    target: ParticleTarget,
    guide: ParticleContext,
    auxiliary: jax.Array,
    mask: jax.Array,
    spec: GuidedPathSpec,
    information: GaussianTwist,
) -> tuple[GaussianTwist, jax.Array]:
    """Batch all guide geometry over time before the sequential particle filter.

    Supported models have additive Gaussian diffusion. Its covariance depends on
    the label and time, not the parent particle; evaluating it at u also keeps any
    proposal approximation fixed throughout the conditional update.
    """
    steps, dimension = auxiliary.shape
    if not isinstance(spec, TwistedParticleAGradSpec) or steps == 1:
        covariances = jax.vmap(
            lambda previous, index: target.transition_moments(guide, previous, index)[1]
        )(auxiliary[:-1], jnp.arange(1, steps))
        return (
            GaussianTwist(
                jnp.zeros((steps, dimension, dimension), auxiliary.dtype), jnp.zeros_like(auxiliary)
            ),
            covariances,
        )

    def affine(previous: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        def moments(state: jax.Array):
            mean, covariance = target.transition_moments(guide, state, index)
            return mean, (mean, covariance)

        matrix, (mean, covariance) = jax.jacfwd(moments, has_aux=True)(previous)
        matrix = jnp.asarray(matrix)
        matrix = matrix * mask[index, :, None] * mask[index - 1, None, :]
        offset = jnp.where(mask[index], mean, 0.0) - matrix @ previous
        return matrix, offset, _free_covariance(covariance, mask[index])

    matrices, offsets, covariances = jax.vmap(affine)(auxiliary[:-1], jnp.arange(1, steps))
    return future_information(matrices, offsets, covariances, information), covariances


def _memory_guide(
    target: ParticleTarget,
    guide: ParticleContext,
    memory: SequentialObservationRuntime,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    label_log_probs: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    spec: GuidedPathSpec,
    rule: InformationRule,
) -> tuple[GaussianTwist, _ProposalGeometry]:
    """Build a singular augmented guide at u, sampling only independent coordinates.

    Recurrences and window statistics inherit shared state innovations with no added
    noise. Their exact nonlinear updates remain in every correction. Long memory,
    nonlinear responses, clipped curvature and separated modes can reduce efficiency.
    """
    points = memory.path(auxiliary)
    steps, dimension = auxiliary.shape
    augmented_mask = jnp.concatenate(
        (mask, jnp.ones((steps, memory.dimension - dimension), dtype=bool)), axis=-1
    )

    information = memory_information(
        rule, memory, observe, label_log_probs, points, variance, mask, augmented_mask
    )

    def affine(previous, current, index):
        parent, innovation = jax.jacfwd(memory.advance, argnums=(0, 1))(previous, current, index)
        parent = parent * augmented_mask[index - 1][None, :]
        innovation = innovation * mask[index][None, :]
        offset = memory.advance(previous, current, index) - parent @ previous - innovation @ current

        def moments(value):
            mean, covariance = memory.guide_moments(value, index)
            return mean, (mean, covariance)

        matrix, (mean, covariance) = jax.jacfwd(moments, has_aux=True)(previous)
        matrix = matrix * augmented_mask[index - 1][None, :]
        transition = parent + innovation @ matrix
        shift = offset + innovation @ (mean - matrix @ previous)
        noise = innovation @ covariance @ innovation.T
        return parent, innovation, offset, covariance, transition, shift, noise

    parents, innovations, offsets, covariances, matrices, shifts, noises = jax.vmap(affine)(
        points[:-1], auxiliary[1:], jnp.arange(1, steps)
    )
    if isinstance(spec, TwistedParticleAGradSpec):
        twist = future_information(matrices, shifts, noises, information)
    else:
        twist = GaussianTwist(
            jnp.zeros_like(information.precision), jnp.zeros_like(information.shift)
        )
    first_innovation = jax.jacfwd(memory.initial)(auxiliary[0]) * mask[0][None, :]
    innovations = jnp.concatenate((first_innovation[None], innovations))
    parents = jnp.concatenate(
        (jnp.zeros((1, memory.dimension, memory.dimension), points.dtype), parents)
    )
    offsets = jnp.concatenate(((points[0] - first_innovation @ auxiliary[0])[None], offsets))
    covariances = jnp.concatenate((target.initial_moments(guide)[1][None], covariances))

    def geometry(
        covariance: jax.Array,
        parent: jax.Array,
        innovation: jax.Array,
        offset: jax.Array,
        local: GaussianTwist,
        future: GaussianTwist,
        free: jax.Array,
    ) -> _ProposalGeometry:
        precision = local.precision + future.precision
        shift = local.shift + future.shift
        pulled = GaussianTwist(
            symmetric(innovation.T @ precision @ innovation),
            innovation.T @ (shift - precision @ offset),
        )
        result = _proposal_geometry(
            covariance, pulled, free, jnp.zeros_like(covariance), jnp.zeros(dimension)
        )
        posterior = result.chol @ result.chol.T
        return result._replace(parent_map=-posterior @ innovation.T @ precision @ parent)

    return twist, jax.vmap(geometry)(
        covariances, parents, innovations, offsets, information, twist, mask
    )


def build_guide(
    geometry: GuideGeometry,
    rule: InformationRule,
    target: ParticleTarget,
    guide: ParticleContext,
    memory: SequentialObservationRuntime,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    label_log_probs: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    spec: GuidedPathSpec,
) -> tuple[GaussianTwist, _ProposalGeometry]:
    """Build proposal generation and its normalized density from a fixed guide."""
    if geometry == "memory":
        return _memory_guide(
            target, guide, memory, observe, label_log_probs, auxiliary, variance, mask, spec, rule
        )
    information = point_information(rule, observe, auxiliary, variance, mask)
    twist, transition_covariances = _build_twist(target, guide, auxiliary, mask, spec, information)
    proposal = jax.vmap(_proposal_geometry)(
        jnp.concatenate((target.initial_moments(guide)[1][None], transition_covariances)),
        information,
        mask,
        twist.precision,
        twist.shift,
    )
    return twist, proposal
