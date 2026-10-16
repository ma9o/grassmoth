"""Auxiliary laws and proposal densities for gradient particle samplers.

Particle-aMALA and aMALA+ use Algorithms 3 and 5 of
https://arxiv.org/abs/2401.14868: the drift is in u | x, and q(x | u) is
centered at u.

The optional fixed pilot mixture changes q only. Every component is scored
under its full mixture density, on precisely the proposed coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.samplers.particle.math import (
    _masked_normal_log_prob,
    _value_and_grad_by_param,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.samplers.particle.contract import (
        SmootherContext,
    )


def _drift(
    gradient: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    kappa: float,
    clip: float,
) -> tuple[jax.Array, jax.Array]:
    gradient = jnp.where(mask, gradient, 0.0)
    norms: jax.Array = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, clip / jnp.maximum(norms, jnp.finfo(gradient.dtype).tiny))
    return kappa * variance * gradient * scale, norms[..., 0]


def _local_gradient(
    ctx: SmootherContext,
    previous: jax.Array,
    current: jax.Array,
    following: jax.Array,
    index: jax.Array,
) -> jax.Array:
    """Use the same bounded-window drift for generation and candidate scoring."""
    prior, prior_gradient = jax.lax.cond(
        index == 0,
        lambda: ctx.initial_value_grad_by_param(current),
        lambda: ctx.transition_current_value_grad_by_param(previous, current, index),
    )
    observation, observation_gradient = _value_and_grad_by_param(
        ctx.contexts,
        current,
        lambda bound, value: ctx.obs_increment_fn(
            bound, value, index, ctx.runtime_observations
        ),
        value_dtype=ctx.traj_dtype,
        grad_dtype=ctx.latent_dtype,
    )
    logits = ctx.initial_label_log_probs + prior + observation
    gradients = prior_gradient + observation_gradient
    if ctx.dsmc_leaf_proposal == "amala_plus":
        future, future_gradient = jax.lax.cond(
            index + 1 < ctx.num_steps,
            lambda: ctx.transition_next_value_grad_by_param(
                current, following, jnp.minimum(index + 1, ctx.num_steps - 1)
            ),
            lambda: (jnp.zeros_like(prior), jnp.zeros_like(prior_gradient)),
        )
        logits = logits + future
        gradients = gradients + future_gradient
    return jnp.einsum("k,kd->d", jax.nn.softmax(logits), gradients)


def _path_local_gradient(ctx: SmootherContext, path: jax.Array) -> jax.Array:
    """Evaluate the declared local auxiliary drift at every path coordinate."""
    previous = jnp.concatenate((path[:1], path[:-1]))
    following = jnp.concatenate((path[1:], path[-1:]))
    return jax.vmap(lambda before, current, after, index: _local_gradient(
        ctx, before, current, after, index
    ))(previous, path, following, jnp.arange(ctx.num_steps))


@dataclass(frozen=True)
class AuxiliaryKernel:
    """One refreshed auxiliary trajectory, its proposal, and its target factors."""

    context: SmootherContext
    reference: jax.Array
    mask: jax.Array
    variance: jax.Array
    auxiliary: jax.Array
    gradient_norms: jax.Array

    def guide_log_density(self, current: jax.Array, index: jax.Array) -> jax.Array:
        """Positive leaf guide depending only on the refreshed auxiliary and candidate."""
        return _masked_normal_log_prob(
            self.auxiliary[index], current, self.variance[index], self.mask[index]
        )

    def local_log_density(
        self,
        previous: jax.Array,
        current: jax.Array,
        following: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Score the actual auxiliary law, including with labels, intervals and replay.

        The scientific target retains every recurrence and likelihood correction
        independently of this bounded-window proposal drift.
        """
        ctx = self.context
        if ctx.amala_kappa == 0:
            return self.guide_log_density(current, index)
        gradient = _local_gradient(ctx, previous, current, following, index)
        drift, _ = _drift(
            gradient,
            self.variance[index],
            self.mask[index],
            ctx.amala_kappa,
            ctx.amala_grad_clip,
        )
        center = current + drift
        return _masked_normal_log_prob(
            self.auxiliary[index], center, self.variance[index], self.mask[index]
        )

    def propose(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Draw independent time-slice banks and evaluate their complete proposal density."""
        ctx = self.context
        noise_key, component_key = jax.random.split(key)
        noise = jax.random.normal(
            noise_key,
            (ctx.num_steps, ctx.num_free_particles, self.reference.shape[-1]),
            dtype=ctx.latent_dtype,
        )
        values = self.auxiliary[:, None] + jnp.sqrt(self.variance[:, None]) * noise
        mixture = ctx.paid_mix
        if mixture is not None:
            spec = mixture.spec
            weights = jnp.asarray(
                (spec.local_weight, spec.pilot_weight, 1 - spec.local_weight - spec.pilot_weight),
                dtype=ctx.latent_dtype,
            )
            components = jax.random.categorical(
                component_key, jnp.log(weights), shape=(ctx.num_steps, ctx.num_free_particles)
            )
            pilot_variance = spec.pilot_var_scale * mixture.variance
            wide_variance = spec.wide_mult * mixture.variance
            pilot = mixture.mean[:, None] + jnp.sqrt(pilot_variance[:, None]) * noise
            wide = mixture.mean[:, None] + jnp.sqrt(wide_variance[:, None]) * noise
            values = jnp.where(
                (components == 0)[..., None],
                values,
                jnp.where((components == 1)[..., None], pilot, wide),
            )
        values = jnp.where(self.mask[:, None], values, self.reference[:, None])
        particles = jnp.concatenate((self.reference[:, None], values), axis=1)
        return particles, self.proposal_log_density(particles)

    def proposal_log_density(self, particles: jax.Array) -> jax.Array:
        """Evaluate the entire proposal mixture on the active coordinate subspace."""
        proposal = _masked_normal_log_prob(
            particles, self.auxiliary[:, None], self.variance[:, None], self.mask[:, None]
        )
        mixture = self.context.paid_mix
        if mixture is None:
            return proposal
        spec = mixture.spec
        weights = jnp.asarray(
            (spec.local_weight, spec.pilot_weight, 1 - spec.local_weight - spec.pilot_weight),
            dtype=self.context.latent_dtype,
        )
        pilot_density = _masked_normal_log_prob(
            particles,
            mixture.mean[:, None],
            spec.pilot_var_scale * mixture.variance[:, None],
            self.mask[:, None],
        )
        wide_density = _masked_normal_log_prob(
            particles,
            mixture.mean[:, None],
            spec.wide_mult * mixture.variance[:, None],
            self.mask[:, None],
        )
        return jax.scipy.special.logsumexp(
            jnp.stack((proposal, pilot_density, wide_density)) + jnp.log(weights)[:, None, None],
            axis=0,
        )


def build_auxiliary(
    context: SmootherContext, key: jax.Array, reference: jax.Array, mask: jax.Array
) -> AuxiliaryKernel:
    """Refresh the chosen auxiliary law before drawing any free particles."""
    variance = 0.5 * jnp.asarray(context.amala_delta, dtype=context.latent_dtype)[:, None]
    noise = jnp.sqrt(variance) * jax.random.normal(key, reference.shape, dtype=context.latent_dtype)
    if context.amala_kappa == 0:
        drift = jnp.zeros_like(reference)
        norms = jnp.zeros(reference.shape[0], dtype=reference.dtype)
    else:
        gradient = _path_local_gradient(context, reference)
        drift, norms = _drift(
            gradient, variance, mask, context.amala_kappa, context.amala_grad_clip
        )
    auxiliary = jnp.where(mask, reference + drift + noise, reference)
    return AuxiliaryKernel(context, reference, mask, variance, auxiliary, norms)
