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
contain an auxiliary factor. For twisted kappa=1, the local proposal and future
twist share a Gaussian approximation of the readings times the actual auxiliary
law, in information form.
The observation score and Hessian are batched once at u, with noise factors reused.
Kappa=0 and ordinary aGRAD retain their unshifted-auxiliary guide.
Twisting linearizes only at u; exact nonlinear
transition means are used in proposals and true densities in every weight.
Poor Gaussian look-ahead and separated modes can slow mixing, never alter its target.

With the supported additive diffusion, proposal factorizations are batched over
time and shared across particles. Prefix label laws are normalized as they advance.
For point readings, the selected backward suffix supplies the final exact densities.
Interval readings replay windows crossing each splice, including their auxiliary
factors, then rescore the selected path once for its final label law.
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
from grassmoth.models.ssm.inference.targets.sequential import (
    SequentialObservationRuntime,
    sequential_observations,
)
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    SequentialSmootherTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import DTypeLike

    from grassmoth.artifacts.sampling import GuidedPathSpec
    from grassmoth.models.ssm.inference.samplers.particle.contract import SmootherContext
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


class _KillingNoise(NamedTuple):
    survival: jax.Array
    replacement: jax.Array
    rotation: jax.Array


def _killing_noise(key: jax.Array, count: int, dtype: DTypeLike) -> _KillingNoise:
    survival, replacement, rotation = jax.random.split(key, 3)
    return _KillingNoise(
        jax.random.uniform(survival, (count,), dtype=dtype),
        jax.random.uniform(replacement, (count,), dtype=dtype),
        jax.random.gumbel(rotation, (count,), dtype=dtype),
    )


def _killing(logits: jax.Array, noise: _KillingNoise) -> jax.Array:
    """Killing with cyclic conditioning on ancestor[0] = 0.

    Karppinen, Singh & Vihola's conditional resampling, as used by the paper:
    https://github.com/AdrienCorenflos/particle_mala/blob/main/gradient_csmc/utils/resamplings.py
    The random rotation is essential; simply pinning ordinary killing is invalid.
    """
    count = logits.size
    survival = jnp.exp(logits - jnp.max(logits))
    cumulative = jnp.cumsum(jax.nn.softmax(logits))
    replacements = jnp.minimum(
        jnp.sum(cumulative[None, :] <= noise.replacement[:, None], axis=-1), count - 1
    )
    indices = jnp.where(noise.survival < survival, jnp.arange(count), replacements)
    rotation_mass = ((1.0 - survival) / count).at[0].set(0.0)
    rotation_mass = rotation_mass.at[0].set(1.0 - jnp.sum(rotation_mass))
    rotation = jnp.argmax(jnp.log(rotation_mass) + noise.rotation)
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
    parent_map: jax.Array

    def center(self, prior_mean: jax.Array, mask: jax.Array) -> jax.Array:
        return (
            jnp.sum(jnp.where(mask, prior_mean, 0.0)[..., None, :] * self.mean_map, axis=-1)
            + self.shift
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


def _unshifted_information(
    auxiliary: jax.Array, variance: jax.Array, mask: jax.Array
) -> GaussianTwist:
    return GaussianTwist(
        jax.vmap(jnp.diag)(mask / variance),
        jnp.where(mask, auxiliary / variance, 0.0),
    )


def _guided_information(
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
) -> GaussianTwist:
    """Approximate readings times the actual auxiliary law, held fixed at u.

    For score g and Hessian H at u, the auxiliary mean's local affine map is
    B = I + diag(v) H, c = v * (g - H u). Its information is
    B.T diag(1/v) B and B.T ((u - c) / v), including when B vanishes. The reading factor uses
    the positive-semidefinite part of -H and retains its score at u.
    A Gaussian observation factor gives its exact product. Curvature clipping,
    nonlinear score extrapolation and separated modes can cost mixing; the
    exact particle corrections still use the true nonlinear model.
    """

    def point(value: jax.Array, v: jax.Array, free: jax.Array, index: jax.Array) -> GaussianTwist:
        def score(state: jax.Array) -> tuple[jax.Array, jax.Array]:
            gradient = observe(state, index)[1]
            return gradient, gradient

        hessian, gradient = jax.jacfwd(score, has_aux=True)(value)
        hessian = symmetric(jnp.asarray(hessian)) * free[:, None] * free[None, :]
        gradient = jnp.where(free, jnp.asarray(gradient), 0.0)
        matrix = jnp.diag(free.astype(value.dtype)) + v[:, None] * hessian
        residual = jnp.where(free, value - v * gradient + v * (hessian @ value), 0.0)
        auxiliary_precision = symmetric(matrix.T @ (matrix / v[:, None]))
        auxiliary_shift = matrix.T @ (residual / v)
        eigenvalues, eigenvectors = jnp.linalg.eigh(-hessian)
        observed_precision = symmetric(
            (eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        )
        observed_shift = gradient + observed_precision @ value
        return GaussianTwist(
            auxiliary_precision + observed_precision, auxiliary_shift + observed_shift
        )

    return jax.vmap(point)(auxiliary, variance, mask, jnp.arange(auxiliary.shape[0]))


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


def _window_guide(
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
    """Build a singular augmented guide at u, sampling only independent coordinates.

    Window statistics inherit the state's innovation, including its cross-covariance.
    Their exact nonlinear updates remain in every correction. Long windows, nonlinear
    responses, clipped curvature and separated modes can reduce proposal efficiency.
    """
    points = memory.path(auxiliary)
    steps, dimension = auxiliary.shape
    augmented_mask = jnp.concatenate(
        (mask, jnp.ones((steps, memory.dimension - dimension), dtype=bool)), axis=-1
    )

    def information_at(
        value: jax.Array, v: jax.Array, free: jax.Array, active: jax.Array, index: jax.Array
    ) -> GaussianTwist:
        if not isinstance(spec, TwistedParticleAGradSpec) or spec.kappa == 0:
            precision = jnp.zeros((memory.dimension, memory.dimension), value.dtype)
            precision = precision.at[:dimension, :dimension].set(jnp.diag(free / v))
            return GaussianTwist(
                precision,
                jnp.zeros_like(value)
                .at[:dimension]
                .set(jnp.where(free, value[:dimension] / v, 0.0)),
            )

        def reading(state):
            return logsumexp(label_log_probs + memory.log_probs(state, index))

        hessian = symmetric(jax.hessian(reading)(value)) * active[:, None] * active[None, :]
        gradient = jnp.where(active, jax.grad(reading)(value), 0.0)
        eigenvalues, eigenvectors = jnp.linalg.eigh(-hessian)
        observed = symmetric((eigenvectors * jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T)

        def auxiliary_mean(state):
            return memory.state(state) + v * observe(state, index)[1]

        mean = auxiliary_mean(value)
        matrix = jax.jacfwd(auxiliary_mean)(value) * free[:, None] * active[None, :]
        residual = jnp.where(free, memory.state(value) - mean + matrix @ value, 0.0)
        return GaussianTwist(
            symmetric(matrix.T @ (matrix / v[:, None])) + observed,
            matrix.T @ (residual / v) + gradient + observed @ value,
        )

    information = jax.vmap(information_at)(
        points, variance, mask, augmented_mask, jnp.arange(steps)
    )

    def affine(previous, current, index):
        parent, innovation = jax.jacfwd(memory.advance, argnums=(0, 1))(previous, current, index)
        parent = parent * augmented_mask[index - 1][None, :]
        innovation = innovation * mask[index][None, :]
        offset = memory.advance(previous, current, index) - parent @ previous - innovation @ current

        def moments(value):
            mean, covariance = target.transition_moments(guide, memory.state(value), index)
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


def _window_backward(
    memory: SequentialObservationRuntime,
    bank: jax.Array,
    weights: jax.Array,
    scores: jax.Array,
    last: jax.Array,
    key: jax.Array,
    auxiliary: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    observe: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    transition: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Pay all crossing windows and auxiliary factors before reusing an exact suffix.

    The compiler's window span bounds the replay. Beyond it, every label's future
    factors are independent of this prefix; their cached sums remain inside the
    label mixture. No approximation or finite-lag truncation enters these weights.
    """
    window = memory.windows
    assert window is not None
    steps, count, _ = bank.shape
    states = memory.state(bank)
    path = jnp.zeros_like(states[:, 0]).at[-1].set(states[-1, last])
    suffix = jnp.zeros((steps + 1, scores.shape[-1]), dtype=scores.dtype)

    def backward(carry, inputs):
        selected, suffix_scores = carry
        index, gumbels = inputs
        stop = window.replay_stops[index]

        def replay(carry, offset):
            previous, label_sum, auxiliary_sum = carry
            time = jnp.minimum(index + 1 + offset, steps - 1)

            def active():
                current = jax.vmap(memory.advance, in_axes=(0, None, None))(
                    previous, selected[time], time
                )
                observed, gradients = jax.vmap(observe, in_axes=(0, None))(current, time)
                density = _masked_normal_log_prob(
                    auxiliary[time],
                    selected[time] + variance[time] * gradients,
                    variance[time],
                    mask[time],
                )
                factors = observed + jax.vmap(transition, in_axes=(0, None, None))(
                    memory.state(previous), selected[time], time
                )
                return (current, label_sum + factors, auxiliary_sum + density), factors

            return jax.lax.cond(
                index + 1 + offset <= stop,
                active,
                lambda: (carry, jnp.zeros_like(label_sum)),
            )

        (_, label_sum, auxiliary_sum), factors = jax.lax.scan(
            replay,
            (bank[index], jnp.zeros_like(scores[index]), jnp.zeros_like(weights[index])),
            jnp.arange(window.max_replay),
        )
        logits = (
            weights[index]
            + auxiliary_sum
            + logsumexp(scores[index] + label_sum + suffix_scores[stop + 1], axis=-1)
        )
        pick = jnp.argmax(logits + gumbels).astype(jnp.int32)

        def replace_suffix(carry, offset):
            total, stored = carry
            time = jnp.minimum(index + 1 + offset, steps - 1)

            def active():
                updated = total + factors[offset, pick]
                return updated, stored.at[time].set(updated)

            return jax.lax.cond(index + 1 + offset <= stop, active, lambda: carry), None

        (_, suffix_scores), _ = jax.lax.scan(
            replace_suffix,
            (suffix_scores[stop + 1], suffix_scores),
            jnp.arange(window.max_replay - 1, -1, -1),
        )
        return (selected.at[index].set(states[index, pick]), suffix_scores), pick

    _, picks = jax.lax.scan(
        backward,
        (path, suffix),
        (
            jnp.arange(steps - 2, -1, -1),
            jax.vmap(lambda k: jax.random.gumbel(k, (count,), dtype=states.dtype))(
                jax.random.split(key, steps - 1)
            ),
        ),
    )
    return jnp.concatenate((picks[::-1], last[None]))


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
    variance = jnp.broadcast_to(ctx.amala_delta.reshape((steps, -1)), reference.shape) / 2
    aux_key, first_key, forward_key, backward_key = jax.random.split(key, 4)
    memory = sequential_observations(target, ctx.contexts)
    reference_memory = memory.path(reference)

    def observe(value: jax.Array, index: jax.Array | int) -> tuple[jax.Array, jax.Array]:
        if spec.kappa == 0:
            return memory.log_probs(value, jnp.asarray(index)).astype(
                ctx.traj_dtype
            ), jnp.zeros_like(memory.state(value))

        def mixture(state: jax.Array) -> tuple[jax.Array, jax.Array]:
            current = memory.replace_current(value, state, jnp.asarray(index))
            per_label = memory.log_probs(current, jnp.asarray(index)).astype(ctx.traj_dtype)
            return logsumexp(ctx.initial_label_log_probs + per_label), per_label

        (_, per_label), gradient = jax.value_and_grad(mixture, has_aux=True)(memory.state(value))
        return jnp.asarray(per_label), jnp.asarray(gradient)

    reference_observed, gradients = jax.vmap(observe)(reference_memory, jnp.arange(steps))
    auxiliary = jnp.where(
        mask,
        reference
        + variance * gradients
        + (jnp.sqrt(variance) * jax.random.normal(aux_key, reference.shape, reference.dtype)),
        reference,
    )
    if memory.windows is not None:
        twist, geometry = _window_guide(
            target,
            guide,
            memory,
            observe,
            ctx.initial_label_log_probs,
            auxiliary,
            variance,
            mask,
            spec,
        )
    else:
        if isinstance(spec, TwistedParticleAGradSpec) and spec.kappa == 1:
            information = _guided_information(observe, auxiliary, variance, mask)
        else:
            information = _unshifted_information(auxiliary, variance, mask)
        twist, transition_covariances = _build_twist(
            target, guide, auxiliary, mask, spec, information
        )
        geometry = jax.vmap(_proposal_geometry)(
            jnp.concatenate((target.initial_moments(guide)[1][None], transition_covariances)),
            information,
            mask,
            twist.precision,
            twist.shift,
        )
    initial_mean = target.initial_moments(guide)[0]

    def local_factors(particles: jax.Array, index: jax.Array | int) -> tuple[jax.Array, jax.Array]:
        # The conditioned particle is the reference: reuse its observation and score.
        observed, scores = jax.vmap(observe, in_axes=(0, None))(particles[1:], index)
        observed = jnp.concatenate((reference_observed[index, None], observed))
        scores = jnp.concatenate((gradients[index, None], scores))
        density = _masked_normal_log_prob(
            auxiliary[index],
            memory.state(particles) + variance[index] * scores,
            variance[index],
            mask[index],
        )
        return observed, density

    def transition(previous: jax.Array, current: jax.Array, index: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda bound, factors: target.transition_log_prob_with_noise(
                bound, factors, previous, current, index
            )
        )(ctx.contexts, memory.noise).astype(ctx.traj_dtype)

    initial_geometry = jax.tree.map(lambda value: value[0], geometry)
    first_center = initial_geometry.center(initial_mean, mask[0])
    first = jnp.concatenate(
        (
            reference[0, None],
            jnp.where(
                mask[0],
                first_center
                + jnp.sum(
                    jax.random.normal(first_key, (count - 1, dimension), reference.dtype)[
                        :, None, :
                    ]
                    * initial_geometry.chol,
                    axis=-1,
                ),
                reference[0],
            ),
        )
    )
    first_memory = jax.vmap(memory.initial)(first)
    first_observed, first_auxiliary = local_factors(first_memory, 0)
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
    first_twists = twist.log_density(first_memory, 0)
    first_weights = (
        first_normalizer
        + first_auxiliary
        + first_twists
        - initial_geometry.log_density(first - first_center, mask[0])
    )

    def forward(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, tuple[_KillingNoise, jax.Array], _ProposalGeometry],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
        previous, weights, scores, previous_twists = carry
        index, draws, proposal = inputs
        killing_noise, noise = draws
        ancestors = _killing(weights, killing_noise)
        parents = previous[ancestors]
        parent_scores = scores[ancestors]
        means = jax.vmap(
            lambda value: target.transition_moments(guide, memory.state(value), index)[0]
        )(parents)
        centers = proposal.center(means, mask[index]) + parents @ proposal.parent_map.T
        current = jnp.concatenate(
            (
                reference[index, None],
                jnp.where(
                    mask[index],
                    centers[1:] + jnp.sum(noise[:, None, :] * proposal.chol, axis=-1),
                    reference[index],
                ),
            )
        )
        current_memory = jax.vmap(memory.advance, in_axes=(0, 0, None))(parents, current, index)
        observed, auxiliary_factors = local_factors(current_memory, index)
        next_scores = (
            parent_scores
            + jax.vmap(transition, in_axes=(0, 0, None))(memory.state(parents), current, index)
            + observed
        )
        # Normalized prefix label laws remove repeated prefix reductions and avoid
        # subtracting two increasingly large log densities as the time grid grows.
        increment = logsumexp(next_scores, axis=-1)
        next_scores = next_scores - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
        current_twists = twist.log_density(current_memory, index)
        next_weights: jax.Array = (
            increment
            + auxiliary_factors
            + current_twists
            - previous_twists[ancestors]
            - proposal.log_density(current - centers, mask[index])
        )
        return (current_memory, next_weights, next_scores, current_twists), (
            current_memory,
            next_weights,
            next_scores,
            current_twists,
            ancestors,
            centers,
            observed,
        )

    def forward_noise(step_key: jax.Array) -> tuple[_KillingNoise, jax.Array]:
        resample_key, proposal_key = jax.random.split(step_key)
        return (
            _killing_noise(resample_key, count, reference.dtype),
            jax.random.normal(proposal_key, (count - 1, dimension), reference.dtype),
        )

    _, (tail, tail_weights, tail_scores, tail_twists, ancestors, centers, tail_observed) = (
        jax.lax.scan(
            forward,
            (first_memory, first_weights, first_scores, first_twists),
            (
                jnp.arange(1, steps),
                jax.vmap(forward_noise)(jax.random.split(forward_key, steps - 1)),
                jax.tree.map(lambda value: value[1:], geometry),
            ),
        )
    )
    observation_bank = jnp.concatenate((first_observed[None], tail_observed))
    memory_bank = jnp.concatenate((first_memory[None], tail))
    bank = memory.state(memory_bank)
    weights = jnp.concatenate((first_weights[None], tail_weights))
    scores = jnp.concatenate((first_scores[None], tail_scores))
    backward_weights = weights - jnp.concatenate((first_twists[None], tail_twists))
    terminal_key, earlier_key = jax.random.split(backward_key)
    last = jax.random.categorical(terminal_key, weights[-1]).astype(jnp.int32)

    if memory.windows is not None:
        picks = _window_backward(
            memory,
            memory_bank,
            backward_weights,
            scores,
            last,
            earlier_key,
            auxiliary,
            variance,
            mask,
            observe,
            transition,
        )
        path = bank[jnp.arange(steps), picks]
        trajectory_scores = jax.vmap(
            lambda bound: target.path_log_prob(bound, path, target.observations)
        )(ctx.contexts)
    else:

        def backward(
            carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            following, suffix = carry
            index, gumbels = inputs
            links = jax.vmap(lambda value: transition(value, following, index + 1))(bank[index])
            logits = backward_weights[index] + logsumexp(scores[index] + links + suffix, axis=-1)
            pick = jnp.argmax(logits + gumbels).astype(jnp.int32)
            value = bank[index, pick]
            return (value, suffix + links[pick] + observation_bank[index, pick]), pick

        (initial_selected, suffix), earlier = jax.lax.scan(
            backward,
            (bank[-1, last], observation_bank[-1, last]),
            (
                jnp.arange(steps - 2, -1, -1),
                jax.vmap(lambda k: jax.random.gumbel(k, (count,), dtype=reference.dtype))(
                    jax.random.split(earlier_key, steps - 1)
                ),
            ),
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
