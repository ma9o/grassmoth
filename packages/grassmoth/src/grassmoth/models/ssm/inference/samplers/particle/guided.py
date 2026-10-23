"""Exact Particle-aGRAD and twisted Particle-aGRAD under marginalized labels.

Corenflos & Finke (2024), https://arxiv.org/abs/2401.14868, sections 4.1/4.4;
Corenflos (2025), https://arxiv.org/abs/2505.04611, section 3.2.

Let L_t^k be the exact prefix density under label k, including its prior weight.
The intermediate target is sum_k L_t^k times the actual auxiliary factors through
t and psi_t(x_t). Thus each forward weight pays the ratio of these prefix sums,
the auxiliary density, psi_t/psi_(t-1), and the complete normalized proposal.
Backward selection uses prefix AND suffix label densities, divided by the prefix
sum and psi_t. Summing label mixtures independently at each time would be wrong.

The single Gaussian proposal guide is built at the ensemble's mean unconstrained
parameter, symmetrically in the labels. The auxiliary drift is an observation-only
mixture score, common to every label. Consequently the final label law does not
contain an auxiliary factor. Twisting linearizes only at u; exact nonlinear
transition means are used in proposals and true densities in every weight.
Poor Gaussian look-ahead and separated modes can slow mixing, never alter its target.

With the supported additive diffusion, proposal factorizations are batched over
time and shared across particles. Prefix label laws are normalized as they advance;
the selected backward suffix supplies the final exact densities without a path rescore.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from grassmoth.artifacts.sampling import TwistedParticleAGradSpec
from grassmoth.models.ssm.inference.samplers.particle.contract import ParticleSmootherResult
from grassmoth.models.ssm.inference.samplers.particle.math import (
    _masked_normal_log_prob,
    multinomial,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import (
    GaussianTwist,
    future_information,
    symmetric,
)
from grassmoth.models.ssm.inference.samplers.particle.weights import (
    requested_weight_statistics,
    weight_history_fields,
)
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    SequentialSmootherTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.sampling import GuidedPathSpec
    from grassmoth.models.ssm.inference.samplers.particle.contract import SmootherContext
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


def conditional_killing(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Killing with cyclic conditioning on ancestor[0] = 0.

    Karppinen, Singh & Vihola's conditional resampling, as used by the paper:
    https://github.com/AdrienCorenflos/particle_mala/blob/main/gradient_csmc/utils/resamplings.py
    The random rotation is essential; simply pinning ordinary killing is invalid.
    """
    survival_key, replacement_key, rotation_key = jax.random.split(key, 3)
    count = logits.size
    survival = jnp.exp(logits - jnp.max(logits))
    indices = jnp.where(
        jax.random.uniform(survival_key, (count,)) < survival,
        jnp.arange(count),
        multinomial(replacement_key, logits, count),
    )
    rotation_mass = ((1.0 - survival) / count).at[0].set(0.0)
    rotation_mass = rotation_mass.at[0].set(1.0 - jnp.sum(rotation_mass))
    rotation = jax.random.categorical(rotation_key, jnp.log(rotation_mass))
    return jnp.roll(indices, -rotation).at[0].set(0).astype(jnp.int32)


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

    def center(self, prior_mean: jax.Array, mask: jax.Array) -> jax.Array:
        return jnp.where(mask, prior_mean, 0.0) @ self.mean_map.T + self.shift

    def log_density(self, difference: jax.Array, mask: jax.Array) -> jax.Array:
        whitened = jnp.where(mask, difference, 0.0) @ self.whiten.T
        return self.log_normalizer - 0.5 * jnp.sum(whitened**2, axis=-1)


def _proposal_geometry(
    covariance: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    precision: jax.Array,
    shift: jax.Array,
) -> _ProposalGeometry:
    identity = jnp.eye(auxiliary.size, dtype=auxiliary.dtype)
    prior_precision = jnp.linalg.solve(_free_covariance(covariance, mask), identity)
    posterior_precision = prior_precision + jnp.diag(mask / variance) + precision
    posterior_covariance = symmetric(jnp.linalg.solve(posterior_precision, identity))
    chol = jnp.linalg.cholesky(posterior_covariance)
    return _ProposalGeometry(
        posterior_covariance @ prior_precision,
        posterior_covariance @ (jnp.where(mask, auxiliary / variance, 0.0) + shift),
        chol,
        jax.scipy.linalg.solve_triangular(chol, identity, lower=True),
        -0.5 * jnp.sum(mask) * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol))),
    )


def _build_twist(
    target: ParticleTarget,
    guide: ParticleContext,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    spec: GuidedPathSpec,
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
    return future_information(
        matrices, offsets, covariances, auxiliary, variance, mask
    ), covariances


def _guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: GuidedPathSpec,
    key: jax.Array,
    reference: jax.Array,
    *,
    record_trace: bool = False,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]:
    """Run one conditional filter and non-Markovian backward label-mixture pass."""
    mask = ctx.latent_free_mask
    steps, dimension = reference.shape
    count = spec.num_particles
    variance = ctx.amala_delta / 2
    aux_key, first_key, forward_key, backward_key = jax.random.split(key, 4)

    def observations(value: jax.Array, index: jax.Array | int) -> jax.Array:
        return jax.vmap(
            lambda bound: target.observation_increment(bound, value, index, target.observations)
        )(ctx.contexts).astype(ctx.traj_dtype)

    def observe(value: jax.Array, index: jax.Array | int) -> tuple[jax.Array, jax.Array]:
        if spec.kappa == 0:
            return observations(value, index), jnp.zeros_like(value)

        def mixture(state: jax.Array) -> tuple[jax.Array, jax.Array]:
            per_label = observations(state, index)
            return logsumexp(ctx.initial_label_log_probs + per_label), per_label

        (_, per_label), gradient = jax.value_and_grad(mixture, has_aux=True)(value)
        return jnp.asarray(per_label), jnp.asarray(gradient)

    reference_observed, gradients = jax.vmap(observe)(reference, jnp.arange(steps))
    auxiliary = jnp.where(
        mask,
        reference
        + variance[:, None] * gradients
        + (
            jnp.sqrt(variance[:, None])
            * jax.random.normal(aux_key, reference.shape, reference.dtype)
        ),
        reference,
    )
    twist, transition_covariances = _build_twist(target, guide, auxiliary, variance, mask, spec)
    initial_mean, initial_covariance = target.initial_moments(guide)
    # Proposal solves and Cholesky factors are independent of the particle ancestors.
    # Prepare them in parallel over time, outside both sequential particle scans.
    geometry = jax.vmap(_proposal_geometry)(
        jnp.concatenate((initial_covariance[None], transition_covariances)),
        auxiliary,
        variance,
        mask,
        twist.precision,
        twist.shift,
    )

    def local_factors(particles: jax.Array, index: jax.Array | int) -> tuple[jax.Array, jax.Array]:
        # The conditioned particle is the reference: reuse its observation and score.
        observed, scores = jax.vmap(observe, in_axes=(0, None))(particles[1:], index)
        observed = jnp.concatenate((reference_observed[index, None], observed))
        scores = jnp.concatenate((gradients[index, None], scores))
        density = _masked_normal_log_prob(
            auxiliary[index],
            particles + variance[index] * scores,
            variance[index],
            mask[index],
        )
        return observed, density

    def transition(previous: jax.Array, current: jax.Array, index: jax.Array) -> jax.Array:
        return jnp.asarray(
            jax.vmap(lambda bound: target.transition_log_prob(bound, previous, current, index))(
                ctx.contexts
            ),
            dtype=ctx.traj_dtype,
        )

    initial_geometry = jax.tree.map(lambda value: value[0], geometry)
    first_center = initial_geometry.center(initial_mean, mask[0])
    first = jnp.concatenate(
        (
            reference[0, None],
            jnp.where(
                mask[0],
                first_center
                + jax.random.normal(first_key, (count - 1, dimension), reference.dtype)
                @ initial_geometry.chol.T,
                reference[0],
            ),
        )
    )
    first_observed, first_auxiliary = local_factors(first, 0)
    first_scores = (
        ctx.initial_label_log_probs
        + jax.vmap(
            lambda value: jax.vmap(lambda bound: target.initial_log_prob(bound, value))(
                ctx.contexts
            )
        )(first)
        + first_observed
    )
    first_normalizer = logsumexp(first_scores, axis=-1)
    # A zero-mass prefix stays all -inf; it has no normalized label law.
    first_scores = (
        first_scores - jnp.where(jnp.isneginf(first_normalizer), 0.0, first_normalizer)[:, None]
    )
    first_twists = twist.log_density(first, 0)
    first_weights = (
        first_normalizer
        + first_auxiliary
        + first_twists
        - initial_geometry.log_density(first - first_center, mask[0])
    )

    def forward(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, _ProposalGeometry],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
        previous, weights, scores, previous_twists = carry
        index, step_key, proposal = inputs
        resample_key, proposal_key = jax.random.split(step_key)
        ancestors = conditional_killing(resample_key, weights)
        parents = previous[ancestors]
        parent_scores = scores[ancestors]
        means = jax.vmap(lambda value: target.transition_moments(guide, value, index)[0])(parents)
        centers = proposal.center(means, mask[index])
        noise = jax.random.normal(proposal_key, (count - 1, dimension), reference.dtype)
        current = jnp.concatenate(
            (
                reference[index, None],
                jnp.where(mask[index], centers[1:] + noise @ proposal.chol.T, reference[index]),
            )
        )
        observed, auxiliary_factors = local_factors(current, index)
        next_scores = (
            parent_scores
            + jax.vmap(transition, in_axes=(0, 0, None))(parents, current, index)
            + observed
        )
        # Normalized prefix label laws remove repeated prefix reductions and avoid
        # subtracting two increasingly large log densities as the time grid grows.
        increment = logsumexp(next_scores, axis=-1)
        next_scores = next_scores - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
        current_twists = twist.log_density(current, index)
        next_weights: jax.Array = (
            increment
            + auxiliary_factors
            + current_twists
            - previous_twists[ancestors]
            - proposal.log_density(current - centers, mask[index])
        )
        return (current, next_weights, next_scores, current_twists), (
            current,
            next_weights,
            next_scores,
            current_twists,
            ancestors,
            centers,
        )

    _, (tail, tail_weights, tail_scores, tail_twists, ancestors, centers) = jax.lax.scan(
        forward,
        (first, first_weights, first_scores, first_twists),
        (
            jnp.arange(1, steps),
            jax.random.split(forward_key, steps - 1),
            jax.tree.map(lambda value: value[1:], geometry),
        ),
    )
    bank = jnp.concatenate((first[None], tail))
    weights = jnp.concatenate((first_weights[None], tail_weights))
    scores = jnp.concatenate((first_scores[None], tail_scores))
    backward_weights = weights - jnp.concatenate((first_twists[None], tail_twists))
    terminal_key, earlier_key = jax.random.split(backward_key)
    last = jax.random.categorical(terminal_key, weights[-1]).astype(jnp.int32)

    def backward(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        following, suffix = carry
        index, step_key = inputs
        links = jax.vmap(lambda value: transition(value, following, index + 1))(bank[index])
        logits = backward_weights[index] + logsumexp(scores[index] + links + suffix, axis=-1)
        pick = jax.random.categorical(step_key, logits).astype(jnp.int32)
        value = bank[index, pick]
        return (value, suffix + links[pick] + observations(value, index)), pick

    (initial_selected, suffix), earlier = jax.lax.scan(
        backward,
        (bank[-1, last], observations(bank[-1, last], steps - 1)),
        (jnp.arange(steps - 2, -1, -1), jax.random.split(earlier_key, steps - 1)),
    )
    picks = jnp.concatenate((earlier[::-1], last[None]))
    path = bank[jnp.arange(steps), picks]
    # The backward suffix already contains every selected exact observation and
    # transition factor. Only the initial density remains; do not rescore the path.
    trajectory_scores = suffix + jax.vmap(
        lambda bound: target.initial_log_prob(bound, initial_selected)
    )(ctx.contexts)
    norms = jnp.linalg.norm(jnp.where(mask, gradients, 0.0), axis=-1)
    measurements = weight_history_fields(
        (requested_weight_statistics(weights, enabled=ctx.diagnostic_weights),)
    )
    result = ParticleSmootherResult(
        path,
        ctx.label_scores_from_densities(trajectory_scores),
        picks,
        mask,
        MappingProxyType(
            {
                "amala_grad_norm_mean": jnp.mean(norms).astype(ctx.traj_dtype),
                "amala_grad_norm_max": jnp.max(norms).astype(ctx.traj_dtype),
                **measurements,
            }
        ),
    )
    if record_trace:
        return TracedResult(
            result,
            SequentialSmootherTraceRecord(
                auxiliary=auxiliary,
                particles=bank,
                proposal_centers=jnp.concatenate(
                    (jnp.broadcast_to(first_center, first.shape)[None], centers)
                ),
                ancestors=ancestors,
                filter_log_weights=weights,
                terminal_selection=CategoricalTraceRecord(jax.nn.log_softmax(weights[-1]), last),
                selected_particles=picks,
            ),
        )
    return result


def guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: GuidedPathSpec,
    key: jax.Array,
    reference: jax.Array,
) -> ParticleSmootherResult:
    """Advance the guided filter without retaining its proposal trace."""
    result = _guided_step(target, ctx, guide, spec, key, reference)
    assert not isinstance(result, TracedResult)
    return result


def trace_guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: GuidedPathSpec,
    key: jax.Array,
    reference: jax.Array,
) -> TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]:
    """Record the same guided update, including its conditional filter genealogy."""
    result = _guided_step(target, ctx, guide, spec, key, reference, record_trace=True)
    assert isinstance(result, TracedResult)
    return result
