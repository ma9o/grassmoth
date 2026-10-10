"""Auxiliary conditional importance updates on independent trajectory coordinates.

Eliminating deterministic recurrences makes the reduced target non-Markovian.
Consequently its particles are complete coordinate paths, rather than independently
stitched time leaves. Each path is replayed under each parameter label by the target.
The auxiliary Gaussian and the full proposal density are both paid explicitly,
as in https://arxiv.org/abs/2303.00301, section 3.3.1. This kernel preserves the
same marginalized parameter-ensemble target as the dSMC smoother.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _masked_normal_log_prob,
    coordinate_proposal_mask,
)


def step(ctx: SmootherContext, key: jax.Array, reference: jax.Array) -> MPGibbsLatentSmootherResult:
    """Condition one particle on the reference and resample on the reduced support."""
    auxiliary_key, mask_key, sample_key, mixture_key, select_key = jax.random.split(key, 5)
    mask = coordinate_proposal_mask(mask_key, ctx.latent_free_mask, ctx.latent_block_coords)
    variance = 0.5 * ctx.amala_delta[:, None]
    auxiliary = reference + jnp.where(
        mask, jnp.sqrt(variance) * jax.random.normal(auxiliary_key, reference.shape), 0.0
    )

    def log_target(path: jax.Array) -> jax.Array:
        return jax.scipy.special.logsumexp(ctx.trajectory_label_log_weights(path))

    gradient = jnp.where(mask, jax.grad(log_target)(auxiliary), 0.0)
    norm = jnp.linalg.norm(gradient)
    gradient = gradient * jnp.minimum(1.0, ctx.amala_grad_clip / jnp.maximum(norm, 1e-12))
    center = auxiliary + ctx.amala_kappa * variance * gradient
    noise = jax.random.normal(sample_key, (ctx.num_free_particles, *reference.shape))
    proposals = center + jnp.sqrt(variance) * noise

    def normal_log_prob(paths: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
        return jnp.sum(_masked_normal_log_prob(paths, mean, var, mask), axis=-1)

    log_weights = jnp.log(
        jnp.asarray(
            [
                ctx.paid_mix_z_weight,
                ctx.paid_mix_pilot_weight,
                1 - ctx.paid_mix_z_weight - ctx.paid_mix_pilot_weight,
            ]
        )
    )
    if ctx.pilot_moments is not None:
        pilot_mean, pilot_variance, wide_variance = ctx.pilot_moments
        component = jax.random.categorical(
            mixture_key, log_weights, shape=(ctx.num_free_particles,)
        )
        proposals = jnp.where(
            (component == 0)[:, None, None],
            proposals,
            pilot_mean
            + jnp.sqrt(jnp.where((component == 1)[:, None, None], pilot_variance, wide_variance))
            * noise,
        )

    proposals = jnp.where(mask, proposals, reference)
    particles = jnp.concatenate((reference[None], proposals))
    proposal_lp = normal_log_prob(particles, center, variance)
    if ctx.pilot_moments is not None:
        pilot_mean, pilot_variance, wide_variance = ctx.pilot_moments
        proposal_lp = jax.scipy.special.logsumexp(
            jnp.stack(
                (
                    log_weights[0] + proposal_lp,
                    log_weights[1] + normal_log_prob(particles, pilot_mean, pilot_variance),
                    log_weights[2] + normal_log_prob(particles, pilot_mean, wide_variance),
                )
            ),
            axis=0,
        )
    weights = (
        jax.vmap(log_target)(particles)
        + normal_log_prob(particles, auxiliary, variance)
        - proposal_lp
    )
    selected = jax.random.categorical(select_key, weights)
    path = particles[selected]
    return MPGibbsLatentSmootherResult(
        latent_path=path,
        final_label_log_probs=ctx.trajectory_label_log_probs(path),
        origin_path=jnp.full(reference.shape[0], selected, dtype=jnp.int32),
        proposal_mask=mask,
        diagnostics=MappingProxyType({"amala_grad_norm_mean": norm, "amala_grad_norm_max": norm}),
    )
