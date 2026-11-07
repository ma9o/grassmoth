"""Exact auxiliary and marginal gradient particle updates under marginalized labels.

Corenflos & Finke (2024), https://arxiv.org/abs/2401.14868, sections 4.1/4.4;
Corenflos (2025), https://arxiv.org/abs/2505.04611, section 3.2.

Let L_t^k be the exact prefix density under label k, including its prior weight.
The intermediate target is sum_k L_t^k times the actual auxiliary factors through
t and psi_t(x_t). Thus each forward weight pays the ratio of these prefix sums,
the auxiliary density, psi_t/psi_(t-1), and the complete normalized proposal.
Backward selection uses prefix AND suffix label densities, divided by the prefix
sum and psi_t. Summing label mixtures independently at each time would be wrong.

Twisted Particle-aGRAD's parameter-mean proposal builds its single Gaussian guide
at the mean unconstrained parameter over labels with finite corrected prior,
symmetrically in the labels. The auxiliary drift is an observation-only
mixture score, common to every label. Consequently the final label law does not
contain an auxiliary factor. For twisted kappa=1, the local proposal and future
twist share a Gaussian approximation of the readings times the actual auxiliary
law, in information form.
The observation score and Hessian are batched once at u, with noise factors reused.
Kappa=0 and Particle-mGRAD retain their unshifted-auxiliary guide.
Twisting linearizes only at u; exact nonlinear
transition means are used in proposals and true densities in every weight.
The optional prefix mixture uses each label's exact transition mean and covariance
for local proposals, retaining the common twist and auxiliary law. It samples by
normalized prefix label weights and pays the full unfloored mixture density for
every particle, including the reference.
Poor Gaussian look-ahead, early label collapse and separated modes can slow mixing,
never alter its target.

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
labels. Thus the non-Markovian mixture prefix/suffix accounting remains exact, and
backward selection uses the true target without auxiliary factors (CF24 Appendix
D.2). Fit preflight limits it to the fused tail's shapes, so its only other route
is the unfused bank for dense diffusion.
"""

from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.artifacts.sampling import (
    GuidedMethodSpec,
    ParameterMixtureProposalSpec,
    PathUpdateSpec,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    GuideGeometry,
    _build_twist,
    _DiagonalProposalGeometry,
    _point_geometry_factory,
    _ProposalGeometry,
    build_guide,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_information import (
    InformationRule,
    point_information,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_random import (
    MIXTURE_DRAW_FOLD_IN,
    KillingNoise,
    TailRandomKeys,
    killing_draws,
    particle_normal,
    particle_uniform,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_selection import (
    PathSelector,
    select_path,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_tail import (
    diagonal_mixture_tail,
)
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import (
    MarginalMixtureBank,
    marginal_mixture_bank,
    mixture_bank_from_labels,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    TRITON_NUM_WARPS,
    _masked_normal_log_prob,
    categorical_from_running,
    categorical_running_mass,
    gather_particles,
    label_logsumexp,
)
from grassmoth.models.ssm.inference.samplers.particle.prefix_mixture import (
    _mixture_geometry,
    _mixture_particles,
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


def _killing_noise(key: PRNGKeyArray, count: int, dtype: DTypeLike) -> KillingNoise:
    return killing_draws(key, count, dtype)


def _killing_reference(logits: Float[Array, " P"], noise: KillingNoise) -> Int[Array, " P"]:
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


def _killing_fused(
    logits: Float[Array, " P"], noise: KillingNoise, *, interpret: bool = False
) -> Int[Array, " P"]:
    """Fuse conditional resampling's reductions in a single particle block."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    count = logits.size
    block = pl.next_power_of_2(count)

    def kernel(logits_ref, survival_ref, replacement_ref, rotation_ref, output_ref, cumulative_ref):
        p = jnp.arange(block)
        values = plt.load(logits_ref.at[p], mask=p < count, other=-jnp.inf)
        maximum = jnp.max(values)
        survival = jnp.exp(values - maximum)
        cumulative = jnp.cumsum(survival / jnp.sum(survival))
        rotation_mass = jnp.where((p > 0) & (p < count), (1.0 - survival) / count, 0.0)
        rotation_mass = jnp.where(p == 0, 1.0 - jnp.sum(rotation_mass), rotation_mass)
        rotation_noise = plt.load(rotation_ref.at[p], mask=p < count, other=-jnp.inf)
        rotation_logits = jnp.log(rotation_mass) + rotation_noise
        rotation = jnp.min(jnp.where(rotation_logits == jnp.max(rotation_logits), p, block))
        source = (p + rotation) % count
        chosen_logits = plt.load(logits_ref.at[source])
        survival_noise = plt.load(survival_ref.at[source])
        replacement_noise = plt.load(replacement_ref.at[source])
        plt.store(cumulative_ref.at[p], cumulative, mask=p < count)
        if not interpret:
            plt.debug_barrier()

        def search_step(_, bounds):
            lower, upper = bounds
            middle = (lower + upper) // 2
            value = plt.load(cumulative_ref.at[jnp.minimum(middle, count - 1)])
            right = (middle < count) & (value <= replacement_noise)
            return jnp.where(right, middle + 1, lower), jnp.where(right, upper, middle)

        lower, _ = jax.lax.fori_loop(
            0,
            count.bit_length(),
            search_step,
            (jnp.zeros((block,), dtype=jnp.int32), jnp.full((block,), count, dtype=jnp.int32)),
        )
        replacements = jnp.minimum(lower, count - 1)
        indices = jnp.where(survival_noise < jnp.exp(chosen_logits - maximum), source, replacements)
        indices = jnp.where(p == 0, 0, indices).astype(jnp.int32)
        plt.store(output_ref.at[p], indices, mask=p < count)

    result, _ = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(logits.shape, jnp.int32),
            jax.ShapeDtypeStruct(logits.shape, logits.dtype),
        ),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=TRITON_NUM_WARPS),
        name="conditional_killing",
    )(logits, noise.survival, noise.replacement, noise.rotation)
    return cast("jax.Array", result)


def _killing(logits: Float[Array, " P"], noise: KillingNoise) -> Int[Array, " P"]:
    return cast(
        "jax.Array",
        jax.lax.platform_dependent(logits, noise, cpu=_killing_reference, cuda=_killing_fused),
    )


class _GuidedComponents(NamedTuple):
    information: InformationRule
    geometry: GuideGeometry
    selector: PathSelector


def _guided_components(target: ParticleTarget, spec: GuidedMethodSpec) -> _GuidedComponents:
    """Resolve immutable choices from the method and compiled support before tracing scans."""
    return _GuidedComponents(
        "actual_law" if spec.kind == "twisted_particle_agrad" and spec.kappa == 1 else "unshifted",
        "memory" if target.intervals is not None or target.state_space.requires_replay else "point",
        "ancestral"
        if target.state_space.recurrence
        else ("backward" if target.intervals is None else "window_replay"),
    )


def _guided_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    guide: ParticleContext,
    spec: PathUpdateSpec,
    key: PRNGKeyArray,
    reference: Float[Array, "T D"],
    *,
    record_trace: bool = False,
    log_precisions: Float[Array, "T M"] | None = None,
) -> ParticleSmootherResult | TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]:
    """Run an exact conditional filter, selecting paths by their compiled memory."""
    mask = ctx.latent_free_mask
    steps, dimension = reference.shape
    method = cast("GuidedMethodSpec", spec.method)
    count = method.num_particles
    marginal_mixture = method.kind == "particle_mgrad"
    variance = jnp.broadcast_to(ctx.amala_delta.reshape((steps, -1)), reference.shape) / 2
    aux_key, first_key, forward_key, backward_key = jax.random.split(key, 4)
    memory = sequential_observations(target, ctx.contexts, guide, log_precisions)
    reference_memory = memory.path(reference)

    def observe(
        value: Float[Array, " R"], index: Int[Array, ""] | int
    ) -> tuple[Float[Array, " K"], Float[Array, " D"]]:
        if method.kappa == 0:
            return memory.log_probs(value, jnp.asarray(index)).astype(
                ctx.traj_dtype
            ), jnp.zeros_like(memory.state(value))

        def mixture(state: Float[Array, " D"]) -> tuple[Float[Array, ""], Float[Array, " K"]]:
            current = memory.replace_current(value, state, jnp.asarray(index))
            per_label = memory.log_probs(current, jnp.asarray(index)).astype(ctx.traj_dtype)
            return label_logsumexp(ctx.initial_label_log_probs + per_label), per_label

        (_, per_label), gradient = jax.value_and_grad(mixture, has_aux=True)(memory.state(value))
        return jnp.asarray(per_label), jnp.asarray(gradient)

    reference_observed, gradients = jax.vmap(observe)(reference_memory, jnp.arange(steps))
    auxiliary = jnp.where(
        mask,
        reference
        + variance * gradients
        + (jnp.sqrt(variance) * particle_normal(aux_key, reference.shape, reference.dtype)),
        reference,
    )
    components = _guided_components(target, method)
    if isinstance(spec.proposal, ParameterMixtureProposalSpec):
        information = point_information(components.information, observe, auxiliary, variance, mask)
        twist, _ = _build_twist(memory, auxiliary, mask, method, information)
        geometry = _mixture_geometry(
            target,
            ctx.contexts,
            auxiliary,
            mask,
            information,
            twist,
            _point_geometry_factory(target, components.information, method),
        )
    else:
        twist, geometry = build_guide(
            components.geometry,
            components.information,
            memory,
            observe,
            ctx.initial_label_log_probs,
            auxiliary,
            variance,
            mask,
            method,
        )
    fused_tail = marginal_mixture and isinstance(geometry, _DiagonalProposalGeometry)
    prepare_parent_law = fused_tail

    def raw_observe(value: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        def density(state: jax.Array) -> tuple[jax.Array, jax.Array]:
            observed = memory.log_probs(memory.replace_current(value, state, index), index).astype(
                ctx.traj_dtype
            )
            return observed, observed

        if method.kappa == 0:
            observed = memory.log_probs(value, index).astype(ctx.traj_dtype)
            return observed, jnp.zeros(
                (ctx.num_parameter_particles, dimension), dtype=ctx.traj_dtype
            )
        if dimension <= ctx.num_parameter_particles:
            derivatives, observed = jax.jacfwd(density, has_aux=True)(memory.state(value))
            return cast("jax.Array", observed), cast("jax.Array", derivatives)

        def component(context, noise, state):
            return target.observation_increment_with_noise(
                context,
                noise,
                state,
                index,
                None if log_precisions is None else log_precisions[index],
            ).astype(ctx.traj_dtype)

        observed, derivatives = jax.vmap(
            jax.value_and_grad(component, argnums=2), in_axes=(0, 0, None)
        )(ctx.contexts, memory.noise, memory.state(value))
        return cast("jax.Array", observed), cast("jax.Array", derivatives)

    initial_mean = target.initial_moments(guide)[0]

    def local_factors(
        particles: Float[Array, "P R"], index: Int[Array, ""] | int
    ) -> tuple[Float[Array, "P K"], Float[Array, "P D"]]:
        # The conditioned particle is the reference: reuse its observation and score.
        observed, scores = jax.vmap(observe, in_axes=(0, None))(particles, index)
        conditioned = jnp.arange(count)[:, None] == 0
        observed = jnp.where(conditioned, reference_observed[index], observed)
        scores = jnp.where(conditioned, gradients[index], scores)
        return observed, scores

    def weight_correction(
        particles: Float[Array, "P D"],
        centers: Float[Array, "P D"],
        scores: Float[Array, "P D"],
        index: Int[Array, ""] | int,
        mixture_density: Float[Array, " P"] | None = None,
        mixture_bank: MarginalMixtureBank | None = None,
    ) -> Float[Array, " P"]:
        if mixture_bank is not None:
            return mixture_bank.log_weights(scores, variance[index], mask[index])
        proposal: _ProposalGeometry = jax.tree.map(lambda value: value[index], geometry)
        density = _masked_normal_log_prob(
            auxiliary[index],
            particles + variance[index] * scores,
            variance[index],
            mask[index],
        )
        return density - (
            mixture_density
            if mixture_density is not None
            else proposal.log_density(particles - centers, mask[index])
        )

    def transition(
        previous: Float[Array, " R"], current: Float[Array, " R"], index: Int[Array, ""]
    ) -> Float[Array, " K"]:
        return memory.transition_log_probs(previous, current, index).astype(ctx.traj_dtype)

    initial_geometry = jax.tree.map(lambda value: value[0], geometry)
    first_mixture_bank = None
    if isinstance(spec.proposal, ParameterMixtureProposalSpec):
        initial_means = jax.vmap(lambda context: target.initial_moments(context)[0])(ctx.contexts)
        initial_centers = initial_geometry.center(initial_means, mask[0])
        mixture_arguments = (
            initial_geometry,
            jnp.broadcast_to(initial_centers, (count, *initial_centers.shape)),
            jnp.broadcast_to(ctx.initial_label_log_probs, (count, ctx.num_parameter_particles)),
        )
        first_noise = (
            particle_normal(first_key, (count - 1, dimension), reference.dtype),
            particle_uniform(
                jax.random.fold_in(first_key, MIXTURE_DRAW_FOLD_IN),
                (count if marginal_mixture else count - 1,),
                dtype=reference.dtype,
            ),
            reference[0],
            mask[0],
        )
        if marginal_mixture:
            first_mixture_bank = marginal_mixture_bank(
                *mixture_arguments, ctx.initial_label_log_probs, *first_noise
            )
            first, first_centers = first_mixture_bank.particles, first_mixture_bank.centers
            first_density = None
        else:
            first, first_density, first_centers = _mixture_particles(
                *mixture_arguments, *first_noise
            )
        first_center = first_centers[0]
    else:
        first_center = initial_geometry.center(initial_mean, mask[0])
        first = initial_geometry.initial_particles(first_center, first_key, reference, mask, count)
        first_density = None
    first_memory = jax.vmap(memory.initial)(first)
    first_observed, first_gradients = local_factors(first_memory, 0)
    first_scores = (
        ctx.initial_label_log_probs
        + jax.vmap(memory.initial_log_probs)(first_memory)
        + first_observed
    )
    first_normalizer = label_logsumexp(first_scores)
    # A zero-mass prefix stays all -inf; it has no normalized label law.
    first_scores = (
        first_scores - jnp.where(jnp.isneginf(first_normalizer), 0.0, first_normalizer)[:, None]
    )
    first_twists = (
        twist.log_density(first_memory, 0) if method.kind == "twisted_particle_agrad" else None
    )
    first_weights = (
        first_normalizer
        + weight_correction(
            first,
            jnp.broadcast_to(first_center, first.shape),
            first_gradients,
            0,
            first_density,
            first_mixture_bank,
        )
        + (first_twists if first_twists is not None else 0.0)
    )

    def forward(
        carry: tuple[
            Float[Array, "P R"],
            Float[Array, " P"],
            Float[Array, "P K"],
            Float[Array, " P"] | None,
            Int[Array, " P"],
            Int[Array, " P"],
            Float[Array, " P"],
        ],
        index: Int[Array, ""],
    ) -> tuple[
        tuple[
            Float[Array, "P R"],
            Float[Array, " P"],
            Float[Array, "P K"],
            Float[Array, " P"] | None,
            Int[Array, " P"],
            Int[Array, " P"],
            Float[Array, " P"],
        ],
        tuple[
            Float[Array, "P R"],
            Float[Array, " P"],
            Float[Array, "P K"],
            Float[Array, " P"] | None,
            Int[Array, " P"],
            Float[Array, "P D"],
            Float[Array, "P K 1"],
        ],
    ]:
        (
            previous,
            weights,
            scores,
            previous_twists,
            prepared_ancestors,
            prepared_labels,
            prepared_correction,
        ) = carry
        proposal = jax.tree.map(lambda value: value[index], geometry)
        noise_index = jnp.minimum(index - 1, steps - 2)
        killing_noise, noise, component_noise = jax.tree.map(
            lambda value: value[noise_index], all_noise
        )
        ancestors = prepared_ancestors if fused_tail else _killing(weights, killing_noise)
        parents = gather_particles(previous, ancestors)
        parent_scores = scores if prepare_parent_law else gather_particles(scores, ancestors)
        current_mixture_bank = None
        if isinstance(spec.proposal, ParameterMixtureProposalSpec):
            means = jax.vmap(
                lambda parent: jax.vmap(
                    lambda context: target.transition_moments(context, parent, index)[0]
                )(ctx.contexts)
            )(parents)
            component_centers = proposal.center(means, mask[index])
            if marginal_mixture:
                if fused_tail:
                    labels = prepared_labels
                    current_mixture_bank = mixture_bank_from_labels(
                        proposal,
                        component_centers,
                        parent_scores,
                        ctx.initial_label_log_probs,
                        noise,
                        labels,
                        reference[index],
                        mask[index],
                    )
                else:
                    current_mixture_bank = marginal_mixture_bank(
                        proposal,
                        component_centers,
                        parent_scores,
                        ctx.initial_label_log_probs,
                        noise,
                        component_noise,
                        reference[index],
                        mask[index],
                    )
                if prepare_parent_law:
                    current_mixture_bank = current_mixture_bank._replace(
                        label_correction=prepared_correction
                    )
                current, centers = current_mixture_bank.particles, current_mixture_bank.centers
                proposal_density = None
            else:
                current, proposal_density, centers = _mixture_particles(
                    proposal,
                    component_centers,
                    parent_scores,
                    noise,
                    component_noise,
                    reference[index],
                    mask[index],
                )
        else:
            means = jax.vmap(lambda value: memory.guide_moments(value, index)[0])(parents)
            centers = proposal.center(means, mask[index]) + proposal.parent_shift(parents)
            current = proposal.next_particles(centers, noise, reference, mask, index)
            proposal_density = None
        current_memory = jax.vmap(memory.advance, in_axes=(0, 0, None))(parents, current, index)
        if (
            isinstance(spec.proposal, ParameterMixtureProposalSpec)
            and not target.state_space.requires_replay
        ):
            # The proposal already evaluated each label's exact nonlinear drift.
            # Replayed states can differ between labels, so they use the scorer
            # bound to their reconstructed predecessor instead.
            transition_scores = jax.vmap(
                memory.transition_log_probs_with_means, in_axes=(0, None, 0)
            )(
                current_memory,
                index,
                means[..., jnp.asarray(target.state_space.transition, dtype=jnp.int32)],
            ).astype(ctx.traj_dtype)
        else:
            transition_scores = jax.vmap(transition, in_axes=(0, 0, None))(
                parents, current_memory, index
            )
        if fused_tail:
            observed, raw_gradients = jax.vmap(raw_observe, in_axes=(0, None))(
                current_memory, index
            )
            assert current_mixture_bank is not None
            bank = current_mixture_bank
            (
                next_weights,
                next_scores,
                observed,
                next_ancestors,
                _,
                next_labels,
                next_parent_scores,
                next_correction,
            ) = diagonal_mixture_tail(
                current,
                bank.prior_centers,
                bank.geometry.chol,
                bank.geometry.whiten,
                bank.label_correction,
                observed,
                raw_gradients,
                parent_scores,
                transition_scores,
                ctx.initial_label_log_probs,
                reference_observed,
                gradients,
                variance,
                mask,
                tail_noise,
                index,
                reference_running,
                all_noise[2],
                compute_gradient=method.kappa != 0,
            )
            current_twists = None
        else:
            observed, current_gradients = local_factors(current_memory, index)
            next_scores = parent_scores + transition_scores + observed
            # Normalized prefix label laws remove repeated prefix reductions and avoid
            # subtracting two increasingly large log densities as the time grid grows.
            increment = label_logsumexp(next_scores)
            next_scores = next_scores - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
            current_twists = (
                twist.log_density(current_memory, index)
                if method.kind == "twisted_particle_agrad"
                else None
            )
            next_weights: jax.Array = (
                increment
                + weight_correction(
                    current,
                    centers,
                    current_gradients,
                    index,
                    proposal_density,
                    current_mixture_bank,
                )
                + (current_twists if current_twists is not None else 0.0)
                - (
                    gather_particles(previous_twists, ancestors)
                    if previous_twists is not None
                    else 0.0
                )
            )
            next_ancestors = ancestors
            next_labels = prepared_labels
            next_parent_scores = next_scores
            next_correction = prepared_correction
        return (
            current_memory,
            next_weights,
            next_parent_scores if prepare_parent_law else next_scores,
            current_twists,
            next_ancestors,
            next_labels,
            next_correction,
        ), (
            current_memory,
            next_weights,
            next_scores,
            current_twists,
            ancestors,
            cast("Float[Array, 'P D']", centers),
            observed[..., None],
        )

    def forward_noise(
        step_key: PRNGKeyArray,
    ) -> tuple[KillingNoise, Float[Array, "N D"], Float[Array, " component_draws"]]:
        resample_key, proposal_key = jax.random.split(step_key)
        return (
            _killing_noise(resample_key, count, reference.dtype),
            particle_normal(proposal_key, (count - 1, dimension), reference.dtype),
            particle_uniform(
                jax.random.fold_in(proposal_key, MIXTURE_DRAW_FOLD_IN),
                (count if marginal_mixture else count - 1,),
                dtype=reference.dtype,
            )
            if isinstance(spec.proposal, ParameterMixtureProposalSpec)
            else jnp.empty((0,), dtype=reference.dtype),
        )

    forward_steps = steps - 1
    scan_steps = (
        ((forward_steps + method.scan_unroll - 1) // method.scan_unroll) * method.scan_unroll
        if components.selector == "backward"
        else forward_steps
    )
    step_keys = jax.random.split(forward_key, max(1, forward_steps))
    if prepare_parent_law:
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(step_keys)
        subkeys = jax.vmap(
            lambda resample, proposal: jnp.concatenate(
                (
                    jax.random.split(resample, 3),
                    jax.random.fold_in(proposal, MIXTURE_DRAW_FOLD_IN)[None],
                )
            )
        )(split_keys[:, 0], split_keys[:, 1])
        reference_draws = jax.vmap(forward_noise)(step_keys)
        tail_noise = TailRandomKeys(
            jax.random.key_data(subkeys), reference_draws[0], reference_draws[2]
        )
        first_draws = forward_noise(step_keys[0])
        all_noise = (
            KillingNoise(*(jnp.zeros((1, count), dtype=reference.dtype) for _ in range(3))),
            jax.vmap(lambda key: particle_normal(key, (count - 1, dimension), reference.dtype))(
                split_keys[:, 1]
            ),
            jnp.empty((1, 0), dtype=reference.dtype),
        )
    else:
        all_noise = jax.vmap(forward_noise)(step_keys)
        first_draws = jax.tree.map(lambda value: value[0], all_noise)
        tail_noise = all_noise[0]
    initial_ancestors = (
        _killing(first_weights, first_draws[0]) if fused_tail and steps > 1 else jnp.arange(count)
    )
    reference_running = categorical_running_mass(ctx.initial_label_log_probs)
    first_running = (
        categorical_running_mass(first_scores)
        if fused_tail
        else jnp.empty((count, 0), dtype=reference.dtype)
    )
    first_labels = (
        categorical_from_running(
            jnp.where(
                jnp.arange(count)[:, None] == 0,
                reference_running,
                gather_particles(first_running, initial_ancestors),
            ),
            first_draws[2],
        )
        if fused_tail
        else jnp.zeros((count,), dtype=jnp.int32)
    )
    first_parent_scores = (
        gather_particles(first_scores, initial_ancestors) if prepare_parent_law else first_scores
    )
    first_correction = (
        ctx.initial_label_log_probs[first_labels]
        - first_parent_scores[jnp.arange(count), first_labels]
        if prepare_parent_law
        else jnp.zeros((count,), dtype=reference.dtype)
    )
    _, (tail, tail_weights, tail_scores, tail_twists, ancestors, centers, tail_observed) = (
        jax.lax.scan(
            forward,
            (
                first_memory,
                first_weights,
                first_parent_scores,
                first_twists,
                initial_ancestors,
                first_labels,
                first_correction,
            ),
            jnp.minimum(jnp.arange(1, scan_steps + 1), steps - 1),
            unroll=method.scan_unroll,
        )
    )
    # A distinct history shape prevents CSE from turning the two fills into a fill plus copy.
    tail_observed = tail_observed[..., 0]
    memory_bank = jnp.concatenate((first_memory[None], tail[:forward_steps]))
    bank = memory.state(memory_bank)
    weights = jnp.concatenate((first_weights[None], tail_weights[:forward_steps]))
    if first_twists is not None:
        assert tail_twists is not None
        backward_weights = weights - jnp.concatenate(
            (first_twists[None], tail_twists[:forward_steps])
        )
    else:
        backward_weights = weights
    ancestors = ancestors[:forward_steps]
    terminal_key, earlier_key = jax.random.split(backward_key)
    last = jax.random.categorical(terminal_key, weights[-1]).astype(jnp.int32)

    path, picks, trajectory_scores = select_path(
        components.selector,
        memory,
        memory_bank,
        bank,
        backward_weights,
        first_scores,
        tail_scores,
        first_observed,
        tail_observed,
        ancestors,
        last,
        earlier_key,
        auxiliary,
        variance,
        mask,
        observe,
        transition,
        reference.dtype,
        method.scan_unroll,
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
                    (jnp.broadcast_to(first_center, first.shape)[None], centers[:forward_steps])
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
    spec: PathUpdateSpec,
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
    spec: PathUpdateSpec,
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
