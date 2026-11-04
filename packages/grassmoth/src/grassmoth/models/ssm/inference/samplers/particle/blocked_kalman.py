"""Label-conditional auxiliary Kalman updates of time blocks.

Draw a label from its exact conditional before using Markov factorization. At
fixed label and u, nonadjacent blocks are independent given their neighbours.
The Gaussian is linearized at the label's exact replay of u, never at the
changing path. Each block pays its true incoming, internal and outgoing
transition and emission densities, plus the auxiliary and normalized proposal
ratio. Sequential colours preserve the conditional law; integrating the sampled
label preserves the mixture. Exact transition and observation factors are cached
across colours and refreshed only for accepted blocks and their outgoing
transitions. Auxiliary and normalized proposal densities remain explicit in each
Metropolis ratio. Random half-block shifts prevent permanent boundaries.

StudentT readings condition on the sweep's transient precisions: the proposal and
the cached factors use each reading's conditional Gaussian.

Window memory carries a block's influence past its neighbours. Colours separate
blocks by the full window support, and cached exact factors cover each affected
region. Deterministic recurrences that reach a density or later reading require
sequential updates through the whole suffix. Proven output-only recurrences are
reconstructed without serializing independent blocks. The block Gaussian
conditions on the entering memory but sees only the next transition beyond it,
so later memory-borne readings cost acceptance.
Nonlinear curvature, multimodality and long-range dependence can still reduce
movement or mixing.
"""

from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.models.ssm.inference.samplers.particle.contract import (
    ParticleSmootherResult,
    SmootherContext,
)
from grassmoth.models.ssm.inference.samplers.particle.kalman import (
    GaussianPathProposal,
    GaussianPathTerms,
    gaussian_path_proposal,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.targets.path_factors import PathFactors, prepare_path_factors
from grassmoth.models.ssm.inference.tracing import AuxiliaryKalmanTraceRecord, TracedResult
from grassmoth.models.ssm.inference.warmup.path_curvature import linearize_path


def blocked_gaussian_proposal(
    terms: GaussianPathTerms,
    free: jax.Array,
    reference: jax.Array,
    *,
    length: int,
    starts: jax.Array,
) -> GaussianPathProposal:
    """Factor the Gaussian conditional of each padded block at fixed neighbours."""
    steps, dimension = reference.shape

    def block(start: jax.Array) -> GaussianPathProposal:
        indices = start + jnp.arange(length)
        valid = (indices >= 0) & (indices < steps)
        selected = jnp.clip(indices, 0, steps - 1)
        mask = free[selected] & valid[:, None]
        matrices = jnp.where(valid[:, None, None], terms.matrices[selected], 0.0)
        offsets = jnp.where(valid[:, None], terms.offsets[selected], 0.0)
        covariances = jnp.where(valid[:, None, None], terms.covariances[selected], 0.0)
        precision = jnp.where(valid[:, None, None], terms.information.precision[selected], 0.0)
        information = jnp.where(valid[:, None], terms.information.shift[selected], 0.0)
        first = jnp.clip(-start, 0, length - 1)
        left = reference[jnp.clip(start - 1, 0, steps - 1)]
        offsets = offsets.at[first].add(jnp.where(start > 0, matrices[first] @ left, 0.0))
        matrices = matrices.at[first].set(0.0)
        end = start + length
        right_index = jnp.clip(end, 0, steps - 1)
        right_free = free[right_index]
        covariance = terms.covariances[right_index] + jnp.diag(
            (~right_free).astype(reference.dtype)
        )
        rhs = jnp.concatenate(
            (
                terms.matrices[right_index],
                (reference[right_index] - terms.offsets[right_index])[:, None],
            ),
            axis=-1,
        )
        solved = jnp.linalg.solve(covariance, rhs) * right_free[:, None]
        right_matrix = terms.matrices[right_index]
        last = jnp.clip(steps - 1 - start, 0, length - 1)
        has_right = (end > 0) & (end < steps)
        precision = precision.at[last].add(
            jnp.where(has_right, right_matrix.T @ solved[:, :dimension], 0.0)
        )
        information = information.at[last].add(
            jnp.where(has_right, right_matrix.T @ solved[:, dimension], 0.0)
        )
        return gaussian_path_proposal(
            GaussianPathTerms(
                matrices, offsets, covariances, GaussianTwist(precision, information)
            ),
            mask,
        )

    return jax.vmap(block)(starts)


class GaussianBlockGeometry(NamedTuple):
    """One fixed block law and its linear response to neighbouring states."""

    proposal: GaussianPathProposal
    left_gain: jax.Array
    right_gain: jax.Array
    left_reference: jax.Array
    right_reference: jax.Array
    left_index: jax.Array
    right_index: jax.Array
    first: jax.Array

    def condition(self, path: jax.Array) -> GaussianPathProposal:
        """Update conditional means without repeating Gaussian elimination."""
        offsets = self.proposal.offsets + jnp.einsum(
            "tij,j->ti", self.right_gain, path[self.right_index] - self.right_reference
        )
        offsets = offsets.at[self.first].add(
            self.left_gain @ (path[self.left_index] - self.left_reference)
        )
        return self.proposal._replace(offsets=offsets)


def prepare_block_geometry(
    terms: GaussianPathTerms,
    free: jax.Array,
    reference: jax.Array,
    *,
    length: int,
    starts: jax.Array,
) -> GaussianBlockGeometry:
    """Factor every block once and transport its endpoint information.

    Tilting a Gaussian Markov law by endpoint linear information changes only
    its conditional offsets. The right tilt propagates through transposed
    conditional matrices; the incoming tilt affects only the first offset.
    All geometry is fixed by the auxiliary point, parameters and free mask.
    """
    proposals = blocked_gaussian_proposal(terms, free, reference, length=length, starts=starts)
    steps, dimension = reference.shape
    identity = jnp.eye(dimension, dtype=reference.dtype)

    def prepare(proposal: GaussianPathProposal, start: jax.Array) -> GaussianBlockGeometry:
        first = jnp.clip(-start, 0, length - 1)
        last = jnp.clip(steps - 1 - start, 0, length - 1)
        first_index = jnp.clip(start, 0, steps - 1)
        left_index = jnp.clip(start - 1, 0, steps - 1)
        right_index = jnp.clip(start + length, 0, steps - 1)
        covariance = proposal.cholesky @ jnp.swapaxes(proposal.cholesky, -1, -2)
        covariance = jnp.where(
            proposal.free[:, :, None] & proposal.free[:, None, :], covariance, 0.0
        )
        incoming = terms.covariances[first_index] + jnp.diag(
            (~free[first_index]).astype(reference.dtype)
        )
        left_information = jnp.linalg.solve(incoming, terms.matrices[first_index])
        left_information = left_information * free[first_index, :, None]
        left_gain = jnp.where(start > 0, covariance[first] @ left_information, 0.0)
        outgoing = terms.covariances[right_index] + jnp.diag(
            (~free[right_index]).astype(reference.dtype)
        )
        right_information = terms.matrices[right_index].T @ (
            jnp.linalg.solve(outgoing, identity) * free[right_index, :, None]
        )
        transport = jnp.concatenate((jnp.swapaxes(proposal.matrices[1:], -1, -2), identity[None]))
        transport = jnp.where((jnp.arange(length) >= last)[:, None, None], identity, transport)
        suffix = jax.lax.associative_scan(lambda left, right: right @ left, transport[::-1])[::-1]
        right_gain = jnp.where(
            (start + length > 0) & (start + length < steps),
            covariance @ suffix @ right_information,
            0.0,
        )
        return GaussianBlockGeometry(
            proposal,
            left_gain,
            right_gain,
            reference[left_index],
            reference[right_index],
            left_index,
            right_index,
            first,
        )

    return jax.vmap(prepare)(proposals, starts)


def blocked_kalman_step(
    target: ParticleTarget,
    ctx: SmootherContext,
    key: jax.Array,
    reference: jax.Array,
    *,
    block_length: int,
    log_precisions: jax.Array | None = None,
) -> TracedResult[ParticleSmootherResult, AuxiliaryKalmanTraceRecord]:
    """Update all block groups under one exact conditional label and one auxiliary."""
    label_key, shift_key, auxiliary_key, colour_key = jax.random.split(key, 4)
    label = jax.random.categorical(label_key, ctx.score_labels(reference).log_probs)
    context = jax.tree.map(lambda value: value[label], ctx.contexts)
    free = ctx.latent_free_mask
    variance = jnp.broadcast_to(ctx.amala_delta.reshape((ctx.num_steps, -1)), reference.shape) / 2
    auxiliary = reference + jnp.where(
        free,
        jnp.sqrt(variance) * jax.random.normal(auxiliary_key, reference.shape, reference.dtype),
        0.0,
    )
    point = target.replay_path(context, auxiliary)
    curvature = linearize_path(target, context, point, log_precisions)
    raw = curvature.proposal_terms(target, context, point)
    augmented_free = curvature.augment(free, fill=False)
    augmented_variance = curvature.augment(variance, fill=1.0)
    terms = raw._replace(
        information=GaussianTwist(
            raw.information.precision + jax.vmap(jnp.diag)(augmented_free / augmented_variance),
            raw.information.shift
            + jnp.where(
                augmented_free,
                curvature.augment(auxiliary, fill=0.0) / augmented_variance,
                0.0,
            ),
        )
    )
    length = min(block_length, ctx.num_steps)
    shift = (
        jax.random.bernoulli(shift_key).astype(jnp.int32) * (length // 2)
        if length < ctx.num_steps
        else jnp.asarray(0, dtype=jnp.int32)
    )
    blocks = (ctx.num_steps + length - 1) // length + 1
    time = jnp.arange(ctx.num_steps)
    block_ids, within = (time + shift) // length, (time + shift) % length
    indices = jnp.arange(blocks)[:, None] * length - shift + jnp.arange(length)[None]
    valid = (indices >= 0) & (indices < ctx.num_steps)
    gather = jnp.clip(indices, 0, ctx.num_steps - 1)
    nonempty = jnp.any(free[gather] & valid[:, :, None], axis=(1, 2))
    starts = jnp.arange(blocks) * length - shift
    memory = bool(target.state_space.recurrence) or target.intervals is not None

    if memory:
        factors = prepare_path_factors(target, context, ctx.runtime_observations, log_precisions)
        initial = factors.initialize(target.replay_path(context, reference))

        def augmented(cache: PathFactors) -> jax.Array:
            return jnp.concatenate((cache.path, cache.memory.reshape(ctx.num_steps, -1)), axis=-1)

        geometry = prepare_block_geometry(
            terms, augmented_free, augmented(initial), length=length, starts=starts
        )
        # Only recurrences that reach a density or a later reading couple all blocks.
        # Separate same-coloured blocks by the entire window support, including the
        # outgoing transition. Proposal derivatives have the same structural support.
        parallel = not target.coupled_recurrences
        reach = 1 if target.intervals is None else target.intervals.memory.max_replay
        colours = min(blocks, 1 + (reach + length - 1) // length)
        groups = (
            np.arange(colours)[:, None]
            + colours * np.arange((blocks + colours - 1) // colours)[None]
            if parallel
            else np.arange(blocks)[:, None]
        )

        def update(
            carry: PathFactors, item: tuple[jax.Array, jax.Array]
        ) -> tuple[PathFactors, tuple[jax.Array, jax.Array, jax.Array]]:
            path = carry.path
            members, random_key = item
            sampling_key, acceptance_key = jax.random.split(random_key)
            present = members < blocks
            safe = jnp.minimum(members, blocks - 1)
            neighbours = augmented(carry)
            laws = jax.vmap(
                lambda block: jax.tree.map(lambda value: value[block], geometry).condition(
                    neighbours
                )
            )(safe)
            draws, log_q_new = jax.vmap(
                lambda law, draw_key: law.sample_with_log_density(draw_key)
            )(laws, jax.random.split(sampling_key, members.shape[0]))
            inside = (
                (block_ids[None, :] == safe[:, None])[..., None]
                & free[None]
                & present[:, None, None]
            )
            candidates = jax.vmap(
                lambda draw, mask: target.replay_path(
                    context, jnp.where(mask, draw[within, : reference.shape[-1]], path)
                )
            )(draws, inside)
            auxiliary_ratio = -0.5 * jnp.sum(
                jnp.where(
                    inside,
                    ((candidates - auxiliary) ** 2 - (path - auxiliary) ** 2) / variance,
                    0.0,
                ),
                axis=(-2, -1),
            )
            times = starts[safe][:, None] + jnp.arange(length)[None]
            valid = (times >= 0) & (times < ctx.num_steps)
            old_blocks = jnp.where(
                valid[..., None], neighbours[jnp.clip(times, 0, ctx.num_steps - 1)], 0.0
            )
            log_q_old = jax.vmap(lambda law, value: law.log_density(value))(laws, old_blocks)
            candidate_factors = jax.vmap(
                lambda candidate, start: factors.update(carry, candidate, start, length)
            )(candidates, starts[safe])
            density_changes = jax.vmap(lambda candidate: candidate.difference(carry))(
                candidate_factors
            )
            log_acceptance = jnp.minimum(
                0.0,
                density_changes + auxiliary_ratio + log_q_old - log_q_new,
            )
            accepted = (
                present
                & nonempty[safe]
                & (
                    jnp.log(
                        jax.random.uniform(acceptance_key, members.shape, dtype=reference.dtype)
                    )
                    < log_acceptance
                )
            )
            # Same-group blocks touch disjoint factors, so accepted blocks combine freely.
            chosen = jnp.any(inside & accepted[:, None, None], axis=0)
            proposed = jnp.sum(jnp.where(inside, candidates, 0.0), axis=0)
            if parallel:
                updated = target.replay_path(context, jnp.where(chosen, proposed, path))
                affected = (
                    accepted[:, None]
                    & (time[None] >= jnp.maximum(starts[safe, None], 0))
                    & (time[None] < starts[safe, None] + length + factors.reach)
                )
                owners = jnp.argmax(affected, axis=0)
                changed = jnp.any(affected, axis=0)

                def merge(old: jax.Array, new: jax.Array) -> jax.Array:
                    selected = jax.vmap(lambda index, owner: new[owner, index])(time, owners)
                    return jnp.where(
                        changed.reshape((ctx.num_steps, *((1,) * (old.ndim - 1)))), selected, old
                    )

                updated_factors = jax.tree.map(merge, carry, candidate_factors)._replace(
                    path=updated
                )
            else:
                updated_factors = jax.tree.map(
                    lambda old, new: jnp.where(accepted[0], new[0], old), carry, candidate_factors
                )
            block_acceptance = (
                jnp.zeros(blocks, reference.dtype)
                .at[members]
                .set(jnp.where(present, log_acceptance, 0.0), mode="drop")
            )
            block_accepted = jnp.zeros(blocks, dtype=bool).at[members].set(accepted, mode="drop")
            return cast("PathFactors", updated_factors), (
                jnp.where(jnp.any(inside, axis=0), proposed, path),
                block_acceptance,
                block_accepted,
            )

        final_factors, (proposals, log_acceptance, accepted) = jax.lax.scan(
            update,
            initial,
            (jnp.asarray(groups), jax.random.split(colour_key, len(groups))),
        )
        return _blocked_result(
            ctx,
            reference,
            final_factors.path,
            free,
            block_ids,
            nonempty,
            auxiliary,
            proposals,
            log_acceptance,
            accepted,
        )

    geometry = prepare_block_geometry(
        terms, augmented_free, reference, length=length, starts=starts
    )
    noise = target.prepare_noise_factors(context, ctx.runtime_observations)

    def densities(path: jax.Array, selected: jax.Array) -> tuple[jax.Array, jax.Array]:
        initial = target.initial_log_prob(context, path[0])
        if ctx.num_steps == 1:
            transitions = jnp.broadcast_to(initial, selected.shape)
        else:
            transitions = jax.vmap(
                lambda index: target.transition_log_prob_with_noise(
                    context,
                    noise,
                    path[jnp.maximum(index - 1, 0)],
                    path[index],
                    jnp.maximum(index, 1),
                )
            )(selected)
            transitions = jnp.where(selected == 0, initial, transitions)
        observations = jax.vmap(
            lambda index: target.observation_increment_with_noise(
                context,
                noise,
                path[index],
                index,
                None if log_precisions is None else log_precisions[index],
            )
        )(selected)
        return transitions, observations

    initial_densities = densities(reference, time)

    def colour(carry: tuple[jax.Array, jax.Array, jax.Array], item: tuple[jax.Array, jax.Array]):
        path, old_transition, old_observation = carry
        parity, random_key = item
        sampling_key, acceptance_key = jax.random.split(random_key)
        selected_blocks = 2 * jnp.arange((blocks + 1) // 2) + parity
        selected_geometry = jax.tree.map(
            lambda value: value[jnp.minimum(selected_blocks, blocks - 1)], geometry
        )
        proposal = jax.vmap(lambda block: block.condition(path))(selected_geometry)
        draws, log_q_new = jax.vmap(lambda law, draw_key: law.sample_with_log_density(draw_key))(
            proposal,
            jax.random.split(sampling_key, blocks)[jnp.minimum(selected_blocks, blocks - 1)],
        )
        active = (jnp.arange(blocks) % 2 == parity) & nonempty
        proposed = jnp.where(active[block_ids, None], draws[block_ids // 2, within], path)
        selected_indices = selected_blocks[:, None] * length - shift + jnp.arange(length + 1)
        selected_valid = (selected_indices >= 0) & (selected_indices < ctx.num_steps)
        selected_inside = selected_valid & (jnp.arange(length + 1) < length)
        selected_times = jnp.clip(selected_indices, 0, ctx.num_steps - 1)
        new_transition, new_observation = (
            value.reshape(selected_times.shape)
            for value in densities(proposed, selected_times.ravel())
        )
        ratios = jnp.sum(
            jnp.where(selected_valid, new_transition - old_transition[selected_times], 0.0), axis=-1
        ) + jnp.sum(
            jnp.where(selected_inside, new_observation - old_observation[selected_times], 0.0),
            axis=-1,
        )
        auxiliary_ratio = -0.5 * jnp.sum(
            jnp.where(
                free[selected_times],
                (
                    (proposed[selected_times] - auxiliary[selected_times]) ** 2
                    - (path[selected_times] - auxiliary[selected_times]) ** 2
                )
                / variance[selected_times],
                0.0,
            ),
            axis=-1,
        )
        ratios = ratios + jnp.sum(jnp.where(selected_inside, auxiliary_ratio, 0.0), axis=-1)
        old_blocks = jnp.where(
            selected_valid[:, :length, None], path[selected_times[:, :length]], 0.0
        )
        log_q_old = jax.vmap(lambda law, value: law.log_density(value))(proposal, old_blocks)
        log_acceptance = (
            jnp.zeros(blocks, path.dtype)
            .at[selected_blocks]
            .set(jnp.minimum(0.0, ratios + log_q_old - log_q_new), mode="drop")
        )
        accepted = active & (
            jnp.log(jax.random.uniform(acceptance_key, (blocks,), dtype=path.dtype))
            < log_acceptance
        )
        updated = jnp.where(accepted[block_ids, None], proposed, path)
        selected_accepted = (selected_blocks < blocks) & accepted[
            jnp.minimum(selected_blocks, blocks - 1)
        ]
        transition_indices = jnp.where(
            selected_valid & selected_accepted[:, None], selected_indices, ctx.num_steps
        )
        observation_indices = jnp.where(
            selected_inside & selected_accepted[:, None], selected_indices, ctx.num_steps
        )
        transitions = old_transition.at[transition_indices.ravel()].set(
            new_transition.ravel(), mode="drop"
        )
        observations = old_observation.at[observation_indices.ravel()].set(
            new_observation.ravel(), mode="drop"
        )
        return (updated, transitions, observations), (proposed, log_acceptance, accepted)

    (path, _, _), (proposals, log_acceptance, accepted) = jax.lax.scan(
        colour, (reference, *initial_densities), (jnp.arange(2), jax.random.split(colour_key, 2))
    )
    return _blocked_result(
        ctx,
        reference,
        path,
        free,
        block_ids,
        nonempty,
        auxiliary,
        proposals,
        log_acceptance,
        accepted,
    )


def _blocked_result(
    ctx: SmootherContext,
    reference: jax.Array,
    path: jax.Array,
    free: jax.Array,
    block_ids: jax.Array,
    nonempty: jax.Array,
    auxiliary: jax.Array,
    proposals: jax.Array,
    log_acceptance: jax.Array,
    accepted: jax.Array,
) -> TracedResult[ParticleSmootherResult, AuxiliaryKalmanTraceRecord]:
    """Report the updated path with one pass of proposals and decisions per block group."""
    moved = jnp.any(accepted, axis=0)[block_ids]
    result = ParticleSmootherResult(
        path,
        ctx.score_labels(path),
        moved.astype(jnp.int32),
        free,
        MappingProxyType(
            {
                "amala_grad_norm_mean": jnp.asarray(0.0, reference.dtype),
                "amala_grad_norm_max": jnp.asarray(0.0, reference.dtype),
                "path_block_acceptance": jnp.sum(accepted) / jnp.maximum(jnp.sum(nonempty), 1),
                "path_update_rms": jnp.sqrt(
                    jnp.sum(jnp.where(free, (path - reference) ** 2, 0.0))
                    / jnp.maximum(jnp.sum(free), 1)
                ),
            }
        ),
    )
    return TracedResult(
        result, AuxiliaryKalmanTraceRecord(auxiliary, proposals, log_acceptance, accepted)
    )
