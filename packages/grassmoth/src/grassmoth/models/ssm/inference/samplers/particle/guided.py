"""Exact auxiliary and marginal gradient particle updates under marginalized labels.

Corenflos & Finke (2024), https://arxiv.org/abs/2401.14868, sections 4.1/4.4;
Corenflos (2025), https://arxiv.org/abs/2505.04611, section 3.2.

Let L_t^k be the exact prefix density under label k, including its prior weight.
The intermediate target is sum_k L_t^k times the actual auxiliary factors through
t and psi_t(x_t). Thus each forward weight pays the ratio of these prefix sums,
the auxiliary density, psi_t/psi_(t-1), and the complete normalized proposal.
Backward selection uses prefix AND suffix label densities, divided by the prefix
sum and psi_t. Summing label mixtures independently at each time would be wrong.

The single Gaussian proposal guide is built at the mean unconstrained parameter
over labels with finite corrected prior, symmetrically in the labels. The auxiliary drift is an observation-only
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
Models with deterministic recurrence stages instead trace the terminal particle's
ancestors. This keeps exact replay linear in time, at the cost of possible early-time
genealogical degeneracy; near-uniform weights do not guarantee movement.

Particle-mGRAD retains the untwisted local proposal and integrates its auxiliary
out of the bank weights. The observation-only auxiliary drift is common to all
labels and depends only on the current point. Thus the non-Markovian mixture
prefix/suffix accounting remains exact, and backward selection uses the true
target without auxiliary factors (CF24 Appendix D.2). Point support is admitted
before execution; intervals and recurrence are not part of this construction.
"""

from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import DTypeLike

from grassmoth.artifacts.sampling import GuidedPathSpec
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    GuideGeometry,
    _ProposalGeometry,
    build_guide,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_information import InformationRule
from grassmoth.models.ssm.inference.samplers.particle.guided_selection import (
    PathSelector,
    select_path,
)
from grassmoth.models.ssm.inference.samplers.particle.marginal import (
    MarginalGeometry,
    marginal_geometry,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    _masked_normal_log_prob,
)
from grassmoth.models.ssm.inference.samplers.particle.weights import (
    requested_weight_statistics,
    weight_history_fields,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from grassmoth.models.ssm.inference.targets.sequential import (
    sequential_observations,
)
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    SequentialSmootherTraceRecord,
    TracedResult,
)
from grassmoth.models.ssm.shapes import Array, Float, Int, PRNGKeyArray


class _KillingNoise(NamedTuple):
    survival: jax.Array
    replacement: jax.Array
    rotation: jax.Array


def _killing_noise(key: PRNGKeyArray, count: int, dtype: DTypeLike) -> _KillingNoise:
    survival, replacement, rotation = jax.random.split(key, 3)
    return _KillingNoise(
        jax.random.uniform(survival, (count,), dtype=dtype),
        jax.random.uniform(replacement, (count,), dtype=dtype),
        jax.random.gumbel(rotation, (count,), dtype=dtype),
    )


def _killing(logits: Float[Array, " P"], noise: _KillingNoise) -> Int[Array, " P"]:
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


class _GuidedComponents(NamedTuple):
    information: InformationRule
    geometry: GuideGeometry
    selector: PathSelector


def _guided_components(target: ParticleTarget, spec: GuidedPathSpec) -> _GuidedComponents:
    """Resolve immutable choices from the method and compiled support before tracing scans."""
    return _GuidedComponents(
        "actual_law"
        if spec.method == "twisted_particle_agrad" and spec.kappa == 1
        else "unshifted",
        "memory" if target.intervals is not None or target.state_space.requires_replay else "point",
        "ancestral"
        if target.state_space.recurrence
        else ("window_replay" if target.intervals is not None else "backward"),
    )


def _guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: GuidedPathSpec,
    key: PRNGKeyArray,
    reference: Float[Array, "T D"],
    *,
    record_trace: bool = False,
    log_precisions: Float[Array, "T M"] | None = None,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]:
    """Run an exact conditional filter, selecting paths by their compiled memory."""
    mask = ctx.latent_free_mask
    steps, dimension = reference.shape
    count = spec.num_particles
    variance = jnp.broadcast_to(ctx.amala_delta.reshape((steps, -1)), reference.shape) / 2
    aux_key, first_key, forward_key, backward_key = jax.random.split(key, 4)
    memory = sequential_observations(target, ctx.contexts, guide, log_precisions)
    reference_memory = memory.path(reference)

    def observe(
        value: Float[Array, " R"], index: Int[Array, ""] | int
    ) -> tuple[Float[Array, " K"], Float[Array, " D"]]:
        if spec.kappa == 0:
            return memory.log_probs(value, jnp.asarray(index)).astype(
                ctx.traj_dtype
            ), jnp.zeros_like(memory.state(value))

        def mixture(state: Float[Array, " D"]) -> tuple[Float[Array, ""], Float[Array, " K"]]:
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
    components = _guided_components(target, spec)
    twist, geometry = build_guide(
        components.geometry,
        components.information,
        memory,
        observe,
        ctx.initial_label_log_probs,
        auxiliary,
        variance,
        mask,
        spec,
    )
    initial_mean = target.initial_moments(guide)[0]
    marginal = (
        jax.vmap(marginal_geometry, in_axes=(0, 0, 0, None))(geometry.chol, variance, mask, count)
        if spec.method == "particle_mgrad"
        else None
    )

    def local_factors(
        particles: Float[Array, "P R"], index: Int[Array, ""] | int
    ) -> tuple[Float[Array, "P K"], Float[Array, "P D"]]:
        # The conditioned particle is the reference: reuse its observation and score.
        observed, scores = jax.vmap(observe, in_axes=(0, None))(particles[1:], index)
        observed = jnp.concatenate((reference_observed[index, None], observed))
        scores = jnp.concatenate((gradients[index, None], scores))
        return observed, scores

    def weight_correction(
        particles: Float[Array, "P D"],
        centers: Float[Array, "P D"],
        scores: Float[Array, "P D"],
        index: Int[Array, ""] | int,
    ) -> Float[Array, " P"]:
        proposal: _ProposalGeometry = jax.tree.map(lambda value: value[index], geometry)
        if marginal is not None:
            factors: MarginalGeometry = jax.tree.map(lambda value: value[index], marginal)
            return factors.log_weights(
                particles, centers - proposal.shift, scores, variance[index], mask[index]
            )
        density = _masked_normal_log_prob(
            auxiliary[index],
            particles + variance[index] * scores,
            variance[index],
            mask[index],
        )
        return density - proposal.log_density(particles - centers, mask[index])

    def transition(
        previous: Float[Array, " R"], current: Float[Array, " R"], index: Int[Array, ""]
    ) -> Float[Array, " K"]:
        return memory.transition_log_probs(previous, current, index).astype(ctx.traj_dtype)

    initial_geometry = jax.tree.map(lambda value: value[0], geometry)
    first_center = initial_geometry.center(initial_mean, mask[0])
    first = initial_geometry.initial_particles(first_center, first_key, reference, mask, count)
    first_memory = jax.vmap(memory.initial)(first)
    first_observed, first_gradients = local_factors(first_memory, 0)
    first_scores = (
        ctx.initial_label_log_probs
        + jax.vmap(memory.initial_log_probs)(first_memory)
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
        + weight_correction(first, jnp.broadcast_to(first_center, first.shape), first_gradients, 0)
        + first_twists
    )

    def forward(
        carry: tuple[
            Float[Array, "P R"], Float[Array, " P"], Float[Array, "P K"], Float[Array, " P"]
        ],
        inputs: tuple[
            Int[Array, ""], tuple[_KillingNoise, Float[Array, "P-1 D"]], _ProposalGeometry
        ],
    ) -> tuple[
        tuple[Float[Array, "P R"], Float[Array, " P"], Float[Array, "P K"], Float[Array, " P"]],
        tuple[
            Float[Array, "P R"],
            Float[Array, " P"],
            Float[Array, "P K"],
            Float[Array, " P"],
            Int[Array, " P"],
            Float[Array, "P D"],
            Float[Array, "P K"],
        ],
    ]:
        previous, weights, scores, previous_twists = carry
        index, draws, proposal = inputs
        killing_noise, noise = draws
        ancestors = _killing(weights, killing_noise)
        parents = previous[ancestors]
        parent_scores = scores[ancestors]
        means = jax.vmap(lambda value: memory.guide_moments(value, index)[0])(parents)
        centers = proposal.center(means, mask[index]) + parents @ proposal.parent_map.T
        current = proposal.next_particles(centers, noise, reference, mask, index)
        current_memory = jax.vmap(memory.advance, in_axes=(0, 0, None))(parents, current, index)
        observed, current_gradients = local_factors(current_memory, index)
        next_scores = (
            parent_scores
            + jax.vmap(transition, in_axes=(0, 0, None))(parents, current_memory, index)
            + observed
        )
        # Normalized prefix label laws remove repeated prefix reductions and avoid
        # subtracting two increasingly large log densities as the time grid grows.
        increment = logsumexp(next_scores, axis=-1)
        next_scores = next_scores - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
        current_twists = twist.log_density(current_memory, index)
        next_weights: jax.Array = (
            increment
            + weight_correction(current, centers, current_gradients, index)
            + current_twists
            - previous_twists[ancestors]
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

    def forward_noise(step_key: PRNGKeyArray) -> tuple[_KillingNoise, Float[Array, "N D"]]:
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
            unroll=spec.scan_unroll,
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

    path, picks, trajectory_scores = select_path(
        components.selector,
        memory,
        memory_bank,
        bank,
        backward_weights,
        scores,
        observation_bank,
        ancestors,
        last,
        earlier_key,
        auxiliary,
        variance,
        mask,
        observe,
        transition,
        reference.dtype,
        spec.scan_unroll,
    )
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
    key: PRNGKeyArray,
    reference: Float[Array, "T D"],
    *,
    log_precisions: Float[Array, "T M"] | None = None,
) -> ParticleSmootherResult:
    """Advance the guided filter without retaining its proposal trace."""
    result = _guided_step(target, ctx, guide, spec, key, reference, log_precisions=log_precisions)
    assert not isinstance(result, TracedResult)
    return result


def trace_guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: GuidedPathSpec,
    key: PRNGKeyArray,
    reference: Float[Array, "T D"],
    *,
    log_precisions: Float[Array, "T M"] | None = None,
) -> TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]:
    """Record the same guided update, including its conditional filter genealogy."""
    result = _guided_step(
        target, ctx, guide, spec, key, reference, record_trace=True, log_precisions=log_precisions
    )
    assert isinstance(result, TracedResult)
    return result
