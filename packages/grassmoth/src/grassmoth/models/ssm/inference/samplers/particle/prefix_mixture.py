"""Prefix-weighted per-label local Gaussian proposals for twisted Particle-aGRAD.

The midpoint twist and shared auxiliary law remain unchanged. Every particle,
including the reference, pays the full normalized mixture density. Early prefix
label collapse and a poor common twist can reduce efficiency without changing
exact target weights. Sampler admission limits this proposal to point readings without replay.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    _proposal_geometry,
    _ProposalGeometry,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist
from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


def _mixture_geometry(
    target: ParticleTarget,
    contexts: ParticleContext,
    auxiliary: jax.Array,
    mask: jax.Array,
    information: GaussianTwist,
    twist: GaussianTwist,
) -> _ProposalGeometry:
    """Give each label its covariance while retaining the common local/future information."""

    def component(context: ParticleContext) -> _ProposalGeometry:
        covariances = jax.vmap(
            lambda previous, index: target.transition_moments(context, previous, index)[1]
        )(auxiliary[:-1], jnp.arange(1, auxiliary.shape[0]))
        return jax.vmap(_proposal_geometry)(
            jnp.concatenate((target.initial_moments(context)[1][None], covariances)),
            information,
            mask,
            twist.precision,
            twist.shift,
        )

    return cast(
        "_ProposalGeometry",
        jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), jax.vmap(component)(contexts)),
    )


def _mixture_particles(
    proposal: _ProposalGeometry,
    centers: jax.Array,
    label_log_probs: jax.Array,
    noise: jax.Array,
    component_noise: jax.Array,
    reference: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample free particles by component; score all particles under the full mixture."""
    components = jnp.argmax(label_log_probs[1:] + component_noise, axis=-1)
    selected_centers = centers[jnp.arange(1, centers.shape[0]), components]
    free = jnp.where(
        mask,
        selected_centers + jnp.sum(noise[:, None, :] * proposal.chol[components], axis=-1),
        reference,
    )
    particles = jnp.concatenate((reference[None], free))
    component_densities = proposal.log_density(particles[:, None, :] - centers, mask)
    density = logsumexp(label_log_probs + component_densities, axis=-1)
    mean = jnp.sum(jnp.exp(label_log_probs)[..., None] * centers, axis=1)
    return particles, density, mean
