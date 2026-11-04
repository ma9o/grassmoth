"""Self-contained exact teaching kernels for cubic SV and a Gaussian oracle.

CPU/float64 is configured by the caller before importing JAX. The same particle
and auxiliary-Kalman code handles one parameter and a finite parameter ensemble.
Particle-mGRAD integrates the shared auxiliary out of the other-particle bank
density (Corenflos & Finke, 2401.14868, Appendix D.2). The global parameter label
is summed using prefix/suffix component densities (Corenflos, 2505.04611, §3).
N always counts the reference particle as well as the N-1 proposed particles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, assert_never

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.infer.hmc import HMCState, hmc

from notebooks.support import sv_model as model

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike


class SVTarget(NamedTuple):
    parameters: jax.Array
    readings: jax.Array
    chart: model.Chart
    log_label_weights: jax.Array


class GaussianTarget(NamedTuple):
    parameters: jax.Array
    readings: jax.Array
    chart: model.Chart
    log_label_weights: jax.Array
    reading_variance: jax.Array


type PathTarget = SVTarget | GaussianTarget
type PathCoordinates = model.PhysicalCoordinates | model.Chart


def sv_target(
    parameters: jax.Array,
    r2: jax.Array,
    coordinates: PathCoordinates = model.PHYSICAL,
) -> SVTarget:
    """Parameters are (M,3); label weights include the transformed parameter prior."""
    chart = (
        model.identity_chart(jnp.mean(parameters, axis=0), r2.shape[0])
        if isinstance(coordinates, model.PhysicalCoordinates)
        else coordinates
    )
    return SVTarget(parameters, r2, chart, jax.vmap(model.log_prior)(parameters))


def gaussian_target(
    parameters: jax.Array,
    readings: jax.Array,
    coordinates: PathCoordinates = model.PHYSICAL,
    *,
    reading_variance: ArrayLike = 0.25,
) -> GaussianTarget:
    """Linear drift, Gaussian observations; otherwise the same mixture semantics."""
    base = sv_target(parameters, readings, coordinates)
    return GaussianTarget(*base, jnp.asarray(reading_variance))


def _parameters_target(
    target: PathTarget, parameters: jax.Array, log_weights: jax.Array
) -> PathTarget:
    match target:
        case SVTarget():
            return SVTarget(parameters, target.readings, target.chart, log_weights)
        case GaussianTarget():
            return GaussianTarget(
                parameters, target.readings, target.chart, log_weights, target.reading_variance
            )
        case _:
            assert_never(target)


class PathFrames(NamedTuple):
    scale: jax.Array
    offset: jax.Array


def _frames(target: PathTarget) -> PathFrames:
    difference = target.parameters - target.chart.reference
    scale = jnp.exp(target.chart.exponents @ difference.T)
    offset = target.chart.center[:, None] + target.chart.slopes @ difference.T
    return PathFrames(scale, offset - scale * target.chart.center[:, None])


def _transition_moments(
    target: PathTarget, frames: PathFrames, index: jax.Array, previous: jax.Array
) -> tuple[jax.Array, jax.Array]:
    physical_previous = previous[..., None] * frames.scale[index - 1] + frames.offset[index - 1]
    cubic = model.CUBIC if isinstance(target, SVTarget) else 0.0
    mean = (
        1.0 - jnp.exp(target.parameters[:, 1])
    ) * physical_previous - cubic * physical_previous**3
    transformed_mean = (mean - frames.offset[index]) / frames.scale[index]
    transformed_variance = jnp.exp(2.0 * target.parameters[:, 2]) / frames.scale[index] ** 2
    return (
        jnp.where(index == 0, -frames.offset[index] / frames.scale[index], transformed_mean),
        jnp.where(index == 0, 1.0 / frames.scale[index] ** 2, transformed_variance),
    )


def _reading_parts(
    target: PathTarget, frames: PathFrames, index: jax.Array, current: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    physical = current[..., None] * frames.scale[index] + frames.offset[index]
    predictor = target.parameters[:, 0] + physical
    if isinstance(target, SVTarget):
        scaled_reading = target.readings[index] * jnp.exp(-predictor)
        density = -0.5 * (
            model.LOG_TWO_PI + jnp.log(target.readings[index]) + predictor + scaled_reading
        )
        score = 0.5 * (scaled_reading - 1.0) * frames.scale[index]
        curvature = 0.5 * scaled_reading * frames.scale[index] ** 2
        return density, score, curvature
    return (
        model.normal_logpdf(target.readings[index], predictor, target.reading_variance),
        (target.readings[index] - predictor) / target.reading_variance * frames.scale[index],
        jnp.broadcast_to(frames.scale[index] ** 2 / target.reading_variance, predictor.shape),
    )


def _increment(
    target: PathTarget,
    frames: PathFrames,
    index: jax.Array,
    previous: jax.Array,
    current: jax.Array,
) -> jax.Array:
    mean, variance = _transition_moments(target, frames, index, previous)
    density = (
        model.normal_logpdf(current[..., None], mean, variance)
        + _reading_parts(target, frames, index, current)[0]
    )
    # The target is on finite real states. Overflowed proposals have zero mass;
    # inf-inf in their transition/reading arithmetic must not become a weight.
    physical = current[..., None] * frames.scale[index] + frames.offset[index]
    return jnp.where(jnp.isfinite(physical), density, -jnp.inf)


def prefix_log_densities(target: PathTarget, path: jax.Array) -> jax.Array:
    """Complete component prefix log densities, including label weights: (T,M)."""
    frames = _frames(target)
    previous = jnp.concatenate((jnp.zeros_like(path[:1]), path[:-1]))
    factors = jax.vmap(lambda index, old, new: _increment(target, frames, index, old, new))(
        jnp.arange(path.shape[0]), previous, path
    )
    return target.log_label_weights + jnp.cumsum(factors, axis=0)


def component_log_densities(target: PathTarget, path: jax.Array) -> jax.Array:
    return prefix_log_densities(target, path)[-1]


def path_log_density(target: PathTarget, path: jax.Array) -> jax.Array:
    return logsumexp(component_log_densities(target, path))


def label_probabilities(target: PathTarget, path: jax.Array) -> jax.Array:
    return jax.nn.softmax(component_log_densities(target, path))


def decoded_paths(target: PathTarget, stored: jax.Array) -> jax.Array:
    """One physical interpretation per parameter candidate: (M,T)."""
    return jax.vmap(lambda eta: model.decode(target.chart, eta, stored))(target.parameters)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NoPathAuxiliary:
    """Bootstrap proposals do not introduce a Gaussian path auxiliary."""


class GradientAuxiliary(NamedTuple):
    values: jax.Array
    nudge: jax.Array


class BankTrace(NamedTuple):
    particles: jax.Array
    ancestors: jax.Array
    log_weights: jax.Array
    probabilities: jax.Array
    target_increments: jax.Array
    proposal_corrections: jax.Array
    auxiliary: NoPathAuxiliary | GradientAuxiliary
    proposal_means: jax.Array
    proposal_variances: jax.Array
    backward_probabilities: jax.Array
    selected_indices: jax.Array
    path: jax.Array


def marginal_bank_log_densities(
    particles: jax.Array,
    prior_means: jax.Array,
    gradients: jax.Array,
    auxiliary_variance: jax.Array,
    auxiliary_map: jax.Array,
    proposal_variance: jax.Array,
) -> jax.Array:
    """Normalized log q^{-n}: integrate u from every other particle's joint law.

    u | candidate n ~ N(x_n + E g_n, E);
    x_i | u ~ N((1-H)m_i + H u, D), i != n.
    The integrated covariance is D I + H^2 E 11.T. Mean/contrast decomposition
    computes all N bank densities in O(N), without a dense covariance matrix.
    """
    count = particles.shape[0]
    others = count - 1
    residuals = particles - (1.0 - auxiliary_map) * prior_means
    average = jnp.mean(residuals)
    centered = residuals - average
    other_average = (count * average - residuals) / others
    scatter = jnp.sum(centered**2) - count / others * centered**2
    mean_variance = proposal_variance + others * auxiliary_map**2 * auxiliary_variance
    auxiliary_mean = particles + auxiliary_variance * gradients
    quadratic = (
        scatter / proposal_variance
        + others * (other_average - auxiliary_map * auxiliary_mean) ** 2 / mean_variance
    )
    return -0.5 * (
        others * model.LOG_TWO_PI
        + (others - 1) * jnp.log(proposal_variance)
        + jnp.log(mean_variance)
        + quadratic
    )


def _guide_gradient(
    guide: PathTarget, frames: PathFrames, index: jax.Array, current: jax.Array
) -> jax.Array:
    """A shared reading score depending only on this state, never its prefix.

    At one parameter this is standard Particle-mGRAD. For a mixture, this is an
    explicit guide choice: the exact target still appears in the bank weights.
    Prefix-independent drift keeps future bank laws unchanged during backward
    prefix swaps, so only the exact mixture suffix enters backward probabilities.
    """
    return _reading_parts(guide, frames, index, current)[1][..., 0]


def _backward_sample(
    key: jax.Array,
    target: PathTarget,
    frames: PathFrames,
    particles: jax.Array,
    log_weights: jax.Array,
    prefixes: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    days = particles.shape[0]
    keys = jax.random.split(key, days)
    terminal_probabilities = jax.nn.softmax(log_weights[-1])
    terminal_index = jax.random.categorical(keys[-1], log_weights[-1])
    terminal = particles[-1, terminal_index]

    def _step(carry: tuple[jax.Array, jax.Array], index: jax.Array):
        following, suffix = carry
        continuation = _increment(target, frames, index + 1, particles[index], following)
        future = continuation + suffix
        joined = jnp.where(jnp.isneginf(prefixes[index]), -jnp.inf, prefixes[index] + future)
        scores = log_weights[index] + logsumexp(joined, axis=-1)
        scores = scores - logsumexp(prefixes[index], axis=-1)
        # A zero-weight prefix cannot be selected, even when its conditional
        # suffix is the indeterminate ratio 0/0 in floating-point arithmetic.
        scores = jnp.where(jnp.isneginf(log_weights[index]), -jnp.inf, scores)
        probabilities = jax.nn.softmax(scores)
        selected = jax.random.categorical(keys[index], scores)
        value = particles[index, selected]
        return (value, future[selected]), (value, selected, probabilities)

    _, (head, indices, probabilities) = jax.lax.scan(
        _step,
        (terminal, jnp.zeros_like(target.log_label_weights)),
        jnp.arange(days - 1),
        reverse=True,
    )
    return (
        jnp.concatenate((head, terminal[None])),
        jnp.concatenate((indices, terminal_index[None])),
        jnp.concatenate((probabilities, terminal_probabilities[None]), axis=0),
    )


def _multinomial_parents(key: jax.Array, log_weights: jax.Array, count: int) -> jax.Array:
    """IID categorical parents without an N-by-N bank of Gumbel random numbers."""
    cumulative = jnp.cumsum(jax.nn.softmax(log_weights)).at[-1].set(1.0)
    uniforms = jax.random.uniform(key, (count,), dtype=log_weights.dtype)
    return jnp.searchsorted(cumulative, uniforms, side="right", method="scan")


def _particle_trace(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    *,
    particles: int,
    delta: ArrayLike,
    marginal: bool,
) -> BankTrace:
    if particles < 2:
        raise ValueError("Conditional SMC needs at least two total particles.")
    days = reference.shape[0]
    frames = _frames(target)
    guide = _parameters_target(
        target, jnp.mean(target.parameters, axis=0, keepdims=True), jnp.zeros(1)
    )
    guide_frames = _frames(guide)
    forward_key, backward_key = jax.random.split(key)
    keys = jax.random.split(forward_key, days)
    variances = jnp.broadcast_to(jnp.asarray(delta) / 2.0, reference.shape)

    def _step(carry: tuple[jax.Array, jax.Array, jax.Array], index: jax.Array):
        previous_bank, previous_weights, previous_prefixes = carry
        ancestor_key, auxiliary_key, proposal_key, label_key = jax.random.split(keys[index], 4)
        ancestors = _multinomial_parents(ancestor_key, previous_weights, particles)
        ancestors = ancestors.at[0].set(0)
        previous = previous_bank[ancestors]
        parent_prefixes = previous_prefixes[ancestors]
        means, transition_variances = _transition_moments(target, frames, index, previous)
        guide_means, guide_variances = _transition_moments(guide, guide_frames, index, previous)
        guide_means, guide_variance = guide_means[:, 0], guide_variances[0]
        auxiliary_variance = variances[index]
        auxiliary_map = guide_variance / (guide_variance + auxiliary_variance)
        proposal_variance = auxiliary_variance * auxiliary_map
        if marginal:
            gradient = _guide_gradient(guide, guide_frames, index, reference[index])
            nudge = auxiliary_variance * gradient
            auxiliary = (
                reference[index]
                + nudge
                + jnp.sqrt(auxiliary_variance)
                * jax.random.normal(auxiliary_key, dtype=reference.dtype)
            )
            proposal_means = (1.0 - auxiliary_map) * guide_means + auxiliary_map * auxiliary
            proposal_variances = jnp.full_like(proposal_means, proposal_variance)
        else:
            labels = jax.random.categorical(label_key, parent_prefixes, axis=-1)
            proposal_means = means[jnp.arange(particles), labels]
            proposal_variances = transition_variances[labels]
            auxiliary, nudge = jnp.zeros((), dtype=reference.dtype), jnp.zeros(())

        bank = proposal_means + jnp.sqrt(proposal_variances) * jax.random.normal(
            proposal_key, (particles,), dtype=reference.dtype
        )
        bank = bank.at[0].set(reference[index])
        factors = _increment(target, frames, index, previous, bank)
        prefixes = jnp.where(jnp.isneginf(parent_prefixes), -jnp.inf, parent_prefixes + factors)
        target_increment = logsumexp(prefixes, axis=-1) - logsumexp(parent_prefixes, axis=-1)
        if marginal:
            gradients = _guide_gradient(guide, guide_frames, index, bank)
            correction = marginal_bank_log_densities(
                bank, guide_means, gradients, variances[index], auxiliary_map, proposal_variance
            )
        else:
            log_proposal = logsumexp(
                parent_prefixes + model.normal_logpdf(bank[:, None], means, transition_variances),
                axis=-1,
            ) - logsumexp(parent_prefixes, axis=-1)
            correction = -log_proposal
        log_weights = target_increment + correction
        log_weights = jnp.where(jnp.isneginf(logsumexp(prefixes, axis=-1)), -jnp.inf, log_weights)
        saved_ancestors = jnp.where(index == 0, -jnp.ones_like(ancestors), ancestors)
        return (bank, log_weights, prefixes), (
            bank,
            saved_ancestors,
            log_weights,
            prefixes,
            target_increment,
            correction,
            auxiliary,
            nudge,
            proposal_means,
            proposal_variances,
        )

    initial = (
        jnp.zeros(particles, dtype=reference.dtype),
        jnp.zeros(particles, dtype=reference.dtype),
        jnp.broadcast_to(target.log_label_weights, (particles, target.parameters.shape[0])),
    )
    _, values = jax.lax.scan(_step, initial, jnp.arange(days))
    (
        bank,
        ancestors,
        weights,
        prefixes,
        increments,
        corrections,
        auxiliary,
        nudge,
        means,
        variance,
    ) = values
    path, indices, backward_probabilities = _backward_sample(
        backward_key, target, frames, bank, weights, prefixes
    )
    auxiliary_record = GradientAuxiliary(auxiliary, nudge) if marginal else NoPathAuxiliary()
    return BankTrace(
        bank,
        ancestors,
        weights,
        jax.nn.softmax(weights, axis=-1),
        increments,
        corrections,
        auxiliary_record,
        means,
        variance,
        backward_probabilities,
        indices,
        path,
    )


def csmc_trace(
    key: jax.Array, target: PathTarget, reference: jax.Array, *, particles: int = 32
) -> BankTrace:
    """Bootstrap conditional filter with exact non-Markovian backward sampling."""
    return _particle_trace(key, target, reference, particles=particles, delta=1.0, marginal=False)


def mgrad_trace(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    *,
    particles: int = 32,
    delta: ArrayLike = 0.2,
) -> BankTrace:
    """Particle-mGRAD with a shared nonlinear Gaussian guide at the mean eta."""
    return _particle_trace(key, target, reference, particles=particles, delta=delta, marginal=True)


class MHLedger(NamedTuple):
    current_target: jax.Array
    proposed_target: jax.Array
    current_auxiliary: jax.Array
    proposed_auxiliary: jax.Array
    current_proposal: jax.Array
    proposed_proposal: jax.Array

    @property
    def log_ratio(self) -> jax.Array:
        return (
            self.proposed_target
            - self.current_target
            + self.proposed_auxiliary
            - self.current_auxiliary
            + self.current_proposal
            - self.proposed_proposal
        )


class KalmanTrace(NamedTuple):
    reference: jax.Array
    auxiliary: jax.Array
    proposal: model.GaussianPath
    proposed: jax.Array
    proposal_mean: jax.Array
    proposal_sd: jax.Array
    ledger: MHLedger
    acceptance_probability: jax.Array
    accepted: jax.Array
    path: jax.Array


def _gaussian_precision(distribution: model.GaussianPath) -> tuple[jax.Array, jax.Array, jax.Array]:
    diagonal = distribution.pivots.at[1:].add(
        distribution.multipliers**2 * distribution.pivots[:-1]
    )
    off = distribution.multipliers * distribution.pivots[:-1]
    information = (diagonal * distribution.mean).at[:-1].add(off * distribution.mean[1:])
    information = information.at[1:].add(off * distribution.mean[:-1])
    return diagonal, off, information


def _guide_precision(
    target: PathTarget, auxiliary: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    eta = jnp.mean(target.parameters, axis=0)
    physical = model.decode(target.chart, eta, auxiliary)
    if isinstance(target, SVTarget):
        diagonal, off, information = model.surrogate_precision(
            eta, model.freeze_surrogate(eta, physical, target.readings)
        )
    else:
        parameters = model.from_eta(eta)
        twin = model.linear_gaussian_twin(
            auxiliary.shape[0],
            decay=parameters.decay,
            vol=parameters.vol,
            level=parameters.level,
            reading_sd=jnp.sqrt(target.reading_variance),
        )
        diagonal, off, information = _gaussian_precision(
            model.linear_gaussian_posterior(twin, target.readings)
        )
    difference = eta - target.chart.reference
    scale = jnp.exp(target.chart.exponents @ difference)
    offset = target.chart.center + target.chart.slopes @ difference - scale * target.chart.center
    precision_offset = (diagonal * offset).at[:-1].add(off * offset[1:])
    precision_offset = precision_offset.at[1:].add(off * offset[:-1])
    return (
        diagonal * scale**2,
        off * scale[:-1] * scale[1:],
        scale * (information - precision_offset),
    )


def _aux_kalman_block(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    delta: ArrayLike,
    start: int,
    stop: int,
) -> KalmanTrace:
    auxiliary_key, proposal_key, acceptance_key = jax.random.split(key, 3)
    variance = jnp.broadcast_to(jnp.asarray(delta) / 2.0, reference.shape)[start:stop]
    auxiliary = reference.at[start:stop].add(
        jnp.sqrt(variance)
        * jax.random.normal(auxiliary_key, (stop - start,), dtype=reference.dtype)
    )
    diagonal, off, information = _guide_precision(target, auxiliary)
    block_diagonal = diagonal[start:stop] + 1.0 / variance
    block_information = information[start:stop] + auxiliary[start:stop] / variance
    if start > 0:
        block_information = block_information.at[0].add(-off[start - 1] * reference[start - 1])
    if stop < reference.shape[0]:
        block_information = block_information.at[-1].add(-off[stop - 1] * reference[stop])
    proposal = model.gaussian_from_precision(
        block_diagonal, off[start : stop - 1], block_information
    )
    proposed = reference.at[start:stop].set(model.gaussian_sample(proposal_key, proposal))
    ledger = MHLedger(
        path_log_density(target, reference),
        path_log_density(target, proposed),
        jnp.sum(model.normal_logpdf(auxiliary[start:stop], reference[start:stop], variance)),
        jnp.sum(model.normal_logpdf(auxiliary[start:stop], proposed[start:stop], variance)),
        model.gaussian_logpdf(proposal, reference[start:stop]),
        model.gaussian_logpdf(proposal, proposed[start:stop]),
    )
    acceptance_probability = jnp.exp(jnp.minimum(0.0, ledger.log_ratio))
    accepted = jnp.log(jax.random.uniform(acceptance_key)) < ledger.log_ratio
    path = jnp.where(accepted, proposed, reference)
    proposal_mean = reference.at[start:stop].set(proposal.mean)
    proposal_sd = (
        jnp.zeros_like(reference)
        .at[start:stop]
        .set(jnp.sqrt(model.gaussian_marginals(proposal).variance))
    )
    return KalmanTrace(
        reference,
        auxiliary,
        proposal,
        proposed,
        proposal_mean,
        proposal_sd,
        ledger,
        acceptance_probability,
        accepted,
        path,
    )


def aux_kalman_trace(
    key: jax.Array, target: PathTarget, reference: jax.Array, *, delta: ArrayLike = 0.5
) -> KalmanTrace:
    """One Gaussian at the ensemble mean, frozen at u; exact whole-mixture MH."""
    return _aux_kalman_block(key, target, reference, delta, 0, reference.shape[0])


class KalmanColorTrace(NamedTuple):
    reference: jax.Array
    auxiliary: jax.Array
    proposal: model.GaussianPath
    active: jax.Array
    proposed: jax.Array
    ledger: MHLedger
    valid_blocks: jax.Array
    acceptance_probabilities: jax.Array
    accepted: jax.Array
    path: jax.Array


def _kalman_color(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    delta: ArrayLike,
    block_ids: jax.Array,
    block_count: int,
    color: int,
) -> KalmanColorTrace:
    """Independent nonadjacent block MH moves, using one tridiagonal solve."""
    auxiliary_key, proposal_key, acceptance_key = jax.random.split(key, 3)
    active = block_ids % 2 == color
    variance = jnp.broadcast_to(jnp.asarray(delta) / 2.0, reference.shape)
    auxiliary = reference + jnp.where(
        active, jnp.sqrt(variance) * jax.random.normal(auxiliary_key, reference.shape), 0.0
    )
    diagonal, off, information = _guide_precision(target, auxiliary)
    diagonal = jnp.where(active, diagonal + 1.0 / variance, 1.0)
    information = jnp.where(active, information + auxiliary / variance, 0.0)
    information = information.at[1:].add(
        jnp.where(active[1:] & ~active[:-1], -off * reference[:-1], 0.0)
    )
    information = information.at[:-1].add(
        jnp.where(active[:-1] & ~active[1:], -off * reference[1:], 0.0)
    )
    off = jnp.where(active[:-1] & active[1:], off, 0.0)
    proposal = model.gaussian_from_precision(diagonal, off, information)
    proposed = jnp.where(active, model.gaussian_sample(proposal_key, proposal), reference)
    frames = _frames(target)
    indices = jnp.arange(reference.shape[0])

    def _sum(values: jax.Array) -> jax.Array:
        return jax.ops.segment_sum(jnp.where(active, values, 0.0), block_ids, block_count)

    def _target_terms(path: jax.Array) -> jax.Array:
        previous = jnp.concatenate((jnp.zeros_like(path[:1]), path[:-1]))
        means, variances = jax.vmap(lambda t, old: _transition_moments(target, frames, t, old))(
            indices, previous
        )
        transition = model.normal_logpdf(path[:, None], means, variances)[:, 0]
        reading = jax.vmap(lambda t, value: _reading_parts(target, frames, t, value)[0])(
            indices, path
        )[:, 0]
        # Assign an outgoing boundary factor to the block containing its parent.
        terms = reading + jnp.where(active, transition, 0.0)
        return terms.at[:-1].add(jnp.where(active[:-1] & ~active[1:], transition[1:], 0.0))

    def _proposal_terms(path: jax.Array) -> jax.Array:
        residual = path - proposal.mean
        transformed = residual.at[:-1].add(proposal.multipliers * residual[1:])
        return 0.5 * (
            jnp.log(proposal.pivots) - model.LOG_TWO_PI - proposal.pivots * transformed**2
        )

    ledger = MHLedger(
        _sum(_target_terms(reference)),
        _sum(_target_terms(proposed)),
        _sum(model.normal_logpdf(auxiliary, reference, variance)),
        _sum(model.normal_logpdf(auxiliary, proposed, variance)),
        _sum(_proposal_terms(reference)),
        _sum(_proposal_terms(proposed)),
    )
    valid = jax.ops.segment_sum(active.astype(jnp.int32), block_ids, block_count) > 0
    probabilities = jnp.where(valid, jnp.exp(jnp.minimum(0.0, ledger.log_ratio)), 0.0)
    accepted = valid & (
        jnp.log(jax.random.uniform(acceptance_key, (block_count,))) < ledger.log_ratio
    )
    path = jnp.where(active & accepted[block_ids], proposed, reference)
    return KalmanColorTrace(
        reference,
        auxiliary,
        proposal,
        active,
        proposed,
        ledger,
        valid,
        probabilities,
        accepted,
        path,
    )


class BlockedKalmanTrace(NamedTuple):
    shift: jax.Array
    block_ids: jax.Array
    colors: tuple[KalmanColorTrace, KalmanColorTrace]
    acceptance_probability: jax.Array
    path: jax.Array


def blocked_kalman_trace(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    *,
    delta: ArrayLike = 0.5,
    block_length: int = 128,
) -> BlockedKalmanTrace:
    """Temporary conditional label, random half-block shift, two colored MH passes.

    Each color is one linear-cost tridiagonal solve with independent accept/reject
    decisions for nonadjacent blocks. Each block conditions on held neighbors.
    A joint MPGibbs caller still selects its final label from the updated mixture.
    """
    if block_length < 1:
        raise ValueError("Block length must be positive.")
    label_key, shift_key, first_key, second_key = jax.random.split(key, 4)
    probabilities = label_probabilities(target, reference)
    label = jax.random.categorical(label_key, jnp.log(probabilities))
    component = _parameters_target(
        target, target.parameters[label][None], target.log_label_weights[label][None]
    )
    shift = jax.random.bernoulli(shift_key).astype(jnp.int32) * (block_length // 2)
    block_ids = (jnp.arange(reference.shape[0]) + shift) // block_length
    block_count = (reference.shape[0] + block_length - 1) // block_length + 1
    first = _kalman_color(first_key, component, reference, delta, block_ids, block_count, 0)
    second = _kalman_color(second_key, component, first.path, delta, block_ids, block_count, 1)
    count = jnp.sum(first.valid_blocks) + jnp.sum(second.valid_blocks)
    acceptance = (
        jnp.sum(first.acceptance_probabilities) + jnp.sum(second.acceptance_probabilities)
    ) / count
    return BlockedKalmanTrace(shift, block_ids, (first, second), acceptance, second.path)


class LeapfrogTrace(NamedTuple):
    positions: jax.Array
    momenta: jax.Array
    energies: jax.Array
    acceptance_probability: jax.Array
    accepted: jax.Array
    position: jax.Array


def _velocity(inverse_mass: jax.Array, momentum: jax.Array) -> jax.Array:
    return inverse_mass @ momentum if inverse_mass.ndim == 2 else inverse_mass * momentum


def hmc_trace(
    key: jax.Array,
    log_density: Callable[[jax.Array], jax.Array],
    position: jax.Array,
    *,
    step_size: ArrayLike = 0.05,
    steps: int = 10,
    inverse_mass: ArrayLike = 1.0,
) -> LeapfrogTrace:
    """Fixed-length HMC, not NUTS; momentum covariance is M, argument is M^-1."""
    momentum_key, acceptance_key = jax.random.split(key)
    inverse = jnp.asarray(inverse_mass)
    noise = jax.random.normal(momentum_key, position.shape, dtype=position.dtype)
    if inverse.ndim == 2:
        momentum = jax.scipy.linalg.solve_triangular(
            jnp.linalg.cholesky(inverse).T, noise, lower=False
        )
    else:
        momentum = noise / jnp.sqrt(inverse)
    score = jax.grad(log_density)

    def _energy(point: jax.Array, impulse: jax.Array) -> jax.Array:
        return -log_density(point) + 0.5 * jnp.dot(impulse, _velocity(inverse, impulse))

    def _step(carry: tuple[jax.Array, jax.Array], _: None):
        point, impulse = carry
        impulse_half = impulse + 0.5 * step_size * score(point)
        proposed = point + step_size * _velocity(inverse, impulse_half)
        impulse_new = impulse_half + 0.5 * step_size * score(proposed)
        return (proposed, impulse_new), (proposed, impulse_new, _energy(proposed, impulse_new))

    _, (positions, momenta, energies) = jax.lax.scan(
        _step, (position, momentum), None, length=steps
    )
    initial_energy = _energy(position, momentum)
    ratio = initial_energy - energies[-1]
    accepted = jnp.log(jax.random.uniform(acceptance_key)) < ratio
    return LeapfrogTrace(
        jnp.concatenate((position[None], positions)),
        jnp.concatenate((momentum[None], momenta)),
        jnp.concatenate((initial_energy[None], energies)),
        jnp.exp(jnp.minimum(0.0, ratio)),
        accepted,
        jnp.where(accepted, positions[-1], position),
    )


def leapfrog_trace(
    key: jax.Array,
    target: PathTarget,
    reference: jax.Array,
    *,
    step_size: ArrayLike = 0.05,
    steps: int = 10,
    inverse_mass: ArrayLike = 1.0,
) -> LeapfrogTrace:
    return hmc_trace(
        key,
        lambda path: path_log_density(target, path),
        reference,
        step_size=step_size,
        steps=steps,
        inverse_mass=inverse_mass,
    )


class OneDayAnatomy(NamedTuple):
    grid: jax.Array
    transition_density: jax.Array
    auxiliary_density: jax.Array
    proposal_density: jax.Array
    reading_density: jax.Array
    target_density: jax.Array
    auxiliary: jax.Array
    nudge: jax.Array
    proposal_mean: jax.Array
    proposal_variance: jax.Array


def one_day_anatomy(
    key: jax.Array,
    eta: jax.Array,
    previous: jax.Array,
    reference: jax.Array,
    squared_reading: jax.Array,
    grid: jax.Array,
    *,
    delta: ArrayLike = 0.2,
    gradient: bool = True,
) -> OneDayAnatomy:
    """One noninitial SV transition; target normalized on the displayed grid."""
    variance = jnp.asarray(delta) / 2.0
    nudge = (
        variance * model.reading_score(eta, reference, squared_reading)
        if gradient
        else jnp.zeros(())
    )
    auxiliary = reference + nudge + jnp.sqrt(variance) * jax.random.normal(key)
    transition_mean = model.transition_mean(eta, previous)
    transition_variance = jnp.exp(2.0 * eta[2])
    mapping = transition_variance / (transition_variance + variance)
    proposal_mean = (1.0 - mapping) * transition_mean + mapping * auxiliary
    proposal_variance = variance * mapping
    log_transition = model.normal_logpdf(grid, transition_mean, transition_variance)
    log_reading = model.reading_logpdf(eta, grid, squared_reading)
    log_target = log_transition + log_reading
    density = jnp.exp(log_target - jnp.max(log_target))
    return OneDayAnatomy(
        grid,
        jnp.exp(log_transition),
        jnp.exp(model.normal_logpdf(grid, auxiliary, variance)),
        jnp.exp(model.normal_logpdf(grid, proposal_mean, proposal_variance)),
        jnp.exp(log_reading),
        density / jnp.trapezoid(density, grid),
        auxiliary,
        nudge,
        proposal_mean,
        proposal_variance,
    )


@dataclass(frozen=True)
class BootstrapSpec:
    particles: int = 32


@dataclass(frozen=True)
class MGradSpec:
    particles: int = 32
    delta: float = 0.2


@dataclass(frozen=True)
class AuxKalmanSpec:
    delta: float = 0.5


@dataclass(frozen=True)
class BlockedKalmanSpec:
    delta: float = 0.5
    block_length: int = 128


@dataclass(frozen=True)
class PathHMCSpec:
    step_size: float = 0.05
    steps: int = 10


type SmootherSpec = BootstrapSpec | MGradSpec | AuxKalmanSpec | BlockedKalmanSpec | PathHMCSpec
type SmootherTrace = BankTrace | KalmanTrace | BlockedKalmanTrace | LeapfrogTrace


def smoother_trace(
    spec: SmootherSpec, key: jax.Array, target: PathTarget, reference: jax.Array
) -> SmootherTrace:
    match spec:
        case BootstrapSpec():
            return csmc_trace(key, target, reference, particles=spec.particles)
        case MGradSpec():
            return mgrad_trace(key, target, reference, particles=spec.particles, delta=spec.delta)
        case AuxKalmanSpec():
            return aux_kalman_trace(key, target, reference, delta=spec.delta)
        case BlockedKalmanSpec():
            return blocked_kalman_trace(
                key, target, reference, delta=spec.delta, block_length=spec.block_length
            )
        case PathHMCSpec():
            return leapfrog_trace(
                key, target, reference, step_size=spec.step_size, steps=spec.steps
            )
        case _:
            assert_never(spec)


def _trace_path(trace: SmootherTrace) -> jax.Array:
    return trace.position if isinstance(trace, LeapfrogTrace) else trace.path


class PathRun(NamedTuple):
    stored_paths: jax.Array
    change_rate: jax.Array


def run_path_chain(
    spec: SmootherSpec,
    key: jax.Array,
    target: PathTarget,
    start: jax.Array,
    *,
    sweeps: int,
) -> PathRun:
    """Pure fixed-target chain in the target's stored coordinates."""

    def _step(path: jax.Array, draw_key: jax.Array):
        updated = _trace_path(smoother_trace(spec, draw_key, target, path))
        return updated, (updated, updated != path)

    _, (paths, changes) = jax.lax.scan(_step, start, jax.random.split(key, sweeps))
    return PathRun(paths, jnp.mean(changes, axis=0))


class BootstrapFilterResult(NamedTuple):
    log_likelihood: jax.Array
    particles: jax.Array
    probabilities: jax.Array


def bootstrap_filter(
    key: jax.Array, eta: jax.Array, r2: jax.Array, *, particles: int = 128
) -> BootstrapFilterResult:
    """Unconditional bootstrap PF: exp(log_likelihood) is the unbiased estimator."""

    def _step(carry: tuple[jax.Array, jax.Array], values: tuple[jax.Array, jax.Array]):
        old, log_weights = carry
        index, time_key = values
        ancestor_key, propagation_key = jax.random.split(time_key)
        ancestors = _multinomial_parents(ancestor_key, log_weights, particles)
        mean = jnp.where(
            index == 0, jnp.zeros_like(old), model.transition_mean(eta, old[ancestors])
        )
        deviation = jnp.where(index == 0, 1.0, jnp.exp(eta[2]))
        current = mean + deviation * jax.random.normal(propagation_key, (particles,))
        weights = model.reading_logpdf(eta, current, r2[index])
        increment = logsumexp(weights) - jnp.log(particles)
        return (current, weights), (increment, current, jax.nn.softmax(weights))

    _, (increments, bank, probabilities) = jax.lax.scan(
        _step,
        (jnp.zeros(particles), jnp.zeros(particles)),
        (jnp.arange(r2.shape[0]), jax.random.split(key, r2.shape[0])),
    )
    return BootstrapFilterResult(jnp.sum(increments), bank, probabilities)


class MPGibbsTrace(NamedTuple):
    parameter_auxiliary: jax.Array
    candidates: jax.Array
    candidate_physical_paths_before: jax.Array
    candidate_physical_paths_after: jax.Array
    label_probabilities_before: jax.Array
    label_probabilities_after: jax.Array
    final_label_probabilities: jax.Array
    selected_label: jax.Array
    path_move: SmootherTrace
    eta: jax.Array
    stored_path: jax.Array
    physical_path: jax.Array


def ensemble_trace(
    key: jax.Array,
    eta: jax.Array,
    stored: jax.Array,
    make_target: Callable[[jax.Array], PathTarget],
    smoother: SmootherSpec,
    parameter_cholesky: jax.Array,
) -> MPGibbsTrace:
    """Two symmetric half proposals, mixture path transition, final label draw.

    parameter_cholesky parameterizes the total candidate displacement covariance;
    each half has covariance chol @ chol.T / 2. The label correction cancels.
    """
    auxiliary_key, candidate_key, path_key, selection_key = jax.random.split(key, 4)
    auxiliary = eta + parameter_cholesky @ jax.random.normal(auxiliary_key, eta.shape) / jnp.sqrt(
        2.0
    )
    candidate = auxiliary + parameter_cholesky @ jax.random.normal(
        candidate_key, eta.shape
    ) / jnp.sqrt(2.0)
    candidates = jnp.stack((eta, candidate))
    target = make_target(candidates)
    path_move = smoother_trace(smoother, path_key, target, stored)
    updated = _trace_path(path_move)
    probabilities = label_probabilities(target, updated)
    selected = jax.random.categorical(selection_key, component_log_densities(target, updated))
    return MPGibbsTrace(
        auxiliary,
        candidates,
        decoded_paths(target, stored),
        decoded_paths(target, updated),
        jax.nn.softmax(prefix_log_densities(target, stored), axis=-1),
        jax.nn.softmax(prefix_log_densities(target, updated), axis=-1),
        probabilities,
        selected,
        path_move,
        candidates[selected],
        updated,
        decoded_paths(target, updated)[selected],
    )


def mpgibbs_trace(
    key: jax.Array,
    eta: jax.Array,
    stored: jax.Array,
    r2: jax.Array,
    coordinates: PathCoordinates,
    smoother: SmootherSpec,
    parameter_cholesky: jax.Array,
) -> MPGibbsTrace:
    return ensemble_trace(
        key,
        eta,
        stored,
        lambda parameters: sv_target(parameters, r2, coordinates),
        smoother,
        parameter_cholesky,
    )


@dataclass(frozen=True)
class NUTSSettings:
    step_size: float = 0.2
    dense_mass: bool = False
    target_accept: float = 0.85
    max_tree_depth: int = 9


DEFAULT_NUTS_SETTINGS = NUTSSettings()


@jax.custom_jvp
def gaussian_kinetic_energy(inverse_mass: jax.Array, momentum: jax.Array) -> jax.Array:
    """Gaussian kinetic energy for a positive diagonal or symmetric SPD metric.

    A dense dot product can overflow to negative infinity because its individual
    cross terms have both signs. The mathematical quadratic form is nonnegative:
    that overflow means infinite energy, hence zero Hamiltonian proposal weight.
    Finite arithmetic and NumPyro's analytic momentum gradient are preserved.
    """
    velocity = inverse_mass @ momentum if inverse_mass.ndim == 2 else inverse_mass * momentum
    energy = 0.5 * jnp.dot(velocity, momentum)
    return jnp.where(jnp.isneginf(energy), jnp.inf, energy)


@gaussian_kinetic_energy.defjvp
def _kinetic_energy_jvp(primals, tangents):
    inverse_mass, momentum = primals
    mass_tangent, momentum_tangent = tangents
    if inverse_mass.ndim == 2:
        velocity = inverse_mass @ momentum
        velocity_tangent = mass_tangent @ momentum
    else:
        velocity = inverse_mass * momentum
        velocity_tangent = mass_tangent * momentum
    tangent = 0.5 * jnp.dot(velocity_tangent, momentum) + jnp.dot(velocity, momentum_tangent)
    return gaussian_kinetic_energy(inverse_mass, momentum), tangent


class NUTSTuning(NamedTuple):
    step_size: jax.Array
    inverse_mass: jax.Array


class NUTSStatistics(NamedTuple):
    acceptance_probabilities: jax.Array
    leapfrog_steps: jax.Array
    divergences: jax.Array


class NUTSRun(NamedTuple):
    positions: jax.Array
    statistics: NUTSStatistics
    tuning: NUTSTuning


def nuts_chain(
    key: jax.Array,
    log_density: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    warmup: int,
    draws: int,
    settings: NUTSSettings = DEFAULT_NUTS_SETTINGS,
) -> NUTSRun:
    """Established NumPyro NUTS on an explicitly supplied exact density."""
    initialize, transition = hmc(
        potential_fn=lambda value: -log_density(value),
        kinetic_fn=gaussian_kinetic_energy,
        algo="NUTS",
    )
    state = initialize(
        initial,
        num_warmup=warmup,
        step_size=settings.step_size,
        dense_mass=settings.dense_mass,
        target_accept_prob=settings.target_accept,
        max_tree_depth=settings.max_tree_depth,
        rng_key=key,
    )

    def _warmup(current: HMCState, _: None):
        return transition(current), None

    state, _ = jax.lax.scan(_warmup, state, None, length=warmup)

    def _sample(current: HMCState, _: None):
        updated = transition(current)
        return updated, (updated.z, updated.accept_prob, updated.num_steps, updated.diverging)

    state, (positions, acceptance, steps, divergences) = jax.lax.scan(
        _sample, state, None, length=draws
    )
    return NUTSRun(
        positions,
        NUTSStatistics(acceptance, steps, divergences),
        NUTSTuning(state.adapt_state.step_size, state.adapt_state.inverse_mass_matrix),
    )


class ParameterTrace(NamedTuple):
    current: jax.Array
    proposed: jax.Array
    forward_mean: jax.Array
    reverse_mean: jax.Array
    log_ratio: jax.Array
    acceptance_probability: jax.Array
    accepted: jax.Array
    position: jax.Array


def parameter_mh_trace(
    key: jax.Array,
    log_density: Callable[[jax.Array], jax.Array],
    current: jax.Array,
    cholesky: jax.Array,
    *,
    langevin: bool = False,
) -> ParameterTrace:
    """RW or MALA; cholesky gives proposal covariance (including step size)."""
    proposal_key, acceptance_key = jax.random.split(key)
    covariance = cholesky @ cholesky.T
    forward_mean = (
        current + 0.5 * covariance @ jax.grad(log_density)(current) if langevin else current
    )
    proposed = forward_mean + cholesky @ jax.random.normal(proposal_key, current.shape)
    reverse_mean = (
        proposed + 0.5 * covariance @ jax.grad(log_density)(proposed) if langevin else proposed
    )
    forward = jax.scipy.linalg.solve_triangular(cholesky, proposed - forward_mean, lower=True)
    reverse = jax.scipy.linalg.solve_triangular(cholesky, current - reverse_mean, lower=True)
    ratio = (
        log_density(proposed)
        - log_density(current)
        + 0.5 * (jnp.dot(forward, forward) - jnp.dot(reverse, reverse))
    )
    accepted = jnp.log(jax.random.uniform(acceptance_key)) < ratio
    return ParameterTrace(
        current,
        proposed,
        forward_mean,
        reverse_mean,
        ratio,
        jnp.exp(jnp.minimum(0.0, ratio)),
        accepted,
        jnp.where(accepted, proposed, current),
    )


def parameter_mh_chain(
    key: jax.Array,
    log_density: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    cholesky: jax.Array,
    *,
    sweeps: int,
    langevin: bool = False,
) -> jax.Array:
    def _step(current: jax.Array, draw_key: jax.Array):
        updated = parameter_mh_trace(
            draw_key, log_density, current, cholesky, langevin=langevin
        ).position
        return updated, updated

    return jax.lax.scan(_step, initial, jax.random.split(key, sweeps))[1]


@dataclass(frozen=True)
class JointNUTSSpec:
    coordinates: model.Coordinates = model.PHYSICAL
    settings: NUTSSettings = NUTSSettings()


@dataclass(frozen=True)
class NUTSSmootherSpec:
    coordinates: PathCoordinates
    smoother: SmootherSpec
    settings: NUTSSettings = NUTSSettings(dense_mass=True)


@dataclass(frozen=True, eq=False)
class MPGibbsSpec:
    coordinates: PathCoordinates
    smoother: SmootherSpec
    parameter_cholesky: jax.Array


type MethodSpec = JointNUTSSpec | NUTSSmootherSpec | MPGibbsSpec


class EnsembleStatistics(NamedTuple):
    candidate_probabilities: jax.Array
    selected_labels: jax.Array


class EnsembleTuning(NamedTuple):
    parameter_cholesky: jax.Array


class MethodRun(NamedTuple):
    parameters: jax.Array
    paths: jax.Array
    substep_summaries: jax.Array
    statistics: NUTSStatistics | EnsembleStatistics
    tuning: NUTSTuning | EnsembleTuning


def path_summary(eta: jax.Array, path: jax.Array) -> jax.Array:
    """[mu, log decay, log vol, mean h, RMS nonlinear residual, middle h]."""
    return jnp.concatenate(
        (
            eta,
            jnp.stack(
                (
                    jnp.mean(path),
                    jnp.sqrt(jnp.mean(model.residuals(eta, path) ** 2)),
                    path[path.shape[0] // 2],
                )
            ),
        )
    )


def _joint_nuts_run(
    spec: JointNUTSSpec,
    key: jax.Array,
    eta: jax.Array,
    path: jax.Array,
    r2: jax.Array,
    warmup: int,
    draws: int,
) -> MethodRun:
    stored = model.stored_path(spec.coordinates, eta, path)

    def _density(position: jax.Array) -> jax.Array:
        return model.held_log_density(spec.coordinates, position[:3], position[3:], r2)

    run = nuts_chain(
        key,
        _density,
        jnp.concatenate((eta, stored)),
        warmup=warmup,
        draws=draws,
        settings=spec.settings,
    )
    parameters = run.positions[:, :3]
    paths = jax.vmap(
        lambda position: model.physical_path(spec.coordinates, position[:3], position[3:])
    )(run.positions)
    summaries = jax.vmap(path_summary)(parameters, paths)
    # One posterior transition, with its actual retained endpoint. The preceding
    # endpoint is available from the previous draw; there is no separate path move.
    return MethodRun(parameters, paths, summaries[:, None, :], run.statistics, run.tuning)


def _nuts_smoother_run(
    spec: NUTSSmootherSpec,
    key: jax.Array,
    eta: jax.Array,
    path: jax.Array,
    r2: jax.Array,
    warmup: int,
    draws: int,
) -> MethodRun:
    nuts_key, path_key = jax.random.split(key)
    stored = model.stored_path(spec.coordinates, eta, path)

    def _potential(held: jax.Array) -> Callable[[jax.Array], jax.Array]:
        return lambda parameters: -model.held_log_density(spec.coordinates, parameters, held, r2)

    initialize, transition = hmc(
        potential_fn_gen=_potential, kinetic_fn=gaussian_kinetic_energy, algo="NUTS"
    )
    state = initialize(
        eta,
        num_warmup=warmup,
        step_size=spec.settings.step_size,
        dense_mass=spec.settings.dense_mass,
        target_accept_prob=spec.settings.target_accept,
        max_tree_depth=spec.settings.max_tree_depth,
        rng_key=nuts_key,
        model_args=(stored,),
    )

    def _step(carry: tuple[HMCState, jax.Array], draw_key: jax.Array):
        current, held = carry
        # A new held path changes the conditional. NumPyro's cached energy and
        # gradient must be refreshed before the next Hamiltonian trajectory.
        energy, gradient = jax.value_and_grad(_potential(held))(current.z)
        current = current._replace(potential_energy=energy, z_grad=gradient)
        updated = transition(current, model_args=(held,))
        before_path = model.physical_path(spec.coordinates, updated.z, held)
        target = sv_target(updated.z[None], r2, spec.coordinates)
        result = smoother_trace(spec.smoother, draw_key, target, held)
        new_held = _trace_path(result)
        physical = model.physical_path(spec.coordinates, updated.z, new_held)
        summaries = jnp.stack(
            (
                path_summary(current.z, model.physical_path(spec.coordinates, current.z, held)),
                path_summary(updated.z, before_path),
                path_summary(updated.z, physical),
            )
        )
        return (updated, new_held), (
            updated.z,
            physical,
            summaries,
            updated.accept_prob,
            updated.num_steps,
            updated.diverging,
        )

    (state, _), values = jax.lax.scan(
        _step, (state, stored), jax.random.split(path_key, warmup + draws)
    )
    parameters, paths, summaries, acceptance, steps, divergences = jax.tree.map(
        lambda value: value[warmup:], values
    )
    return MethodRun(
        parameters,
        paths,
        summaries,
        NUTSStatistics(acceptance, steps, divergences),
        NUTSTuning(state.adapt_state.step_size, state.adapt_state.inverse_mass_matrix),
    )


def _mpgibbs_run(
    spec: MPGibbsSpec,
    key: jax.Array,
    eta: jax.Array,
    path: jax.Array,
    r2: jax.Array,
    warmup: int,
    draws: int,
) -> MethodRun:
    stored = model.stored_path(spec.coordinates, eta, path)

    def _step(carry: tuple[jax.Array, jax.Array], draw_key: jax.Array):
        parameters, held = carry
        trace = mpgibbs_trace(
            draw_key, parameters, held, r2, spec.coordinates, spec.smoother, spec.parameter_cholesky
        )
        summaries = jnp.stack(
            (
                path_summary(parameters, trace.candidate_physical_paths_before[0]),
                path_summary(parameters, trace.candidate_physical_paths_after[0]),
                path_summary(trace.eta, trace.physical_path),
            )
        )
        return (trace.eta, trace.stored_path), (
            trace.eta,
            trace.physical_path,
            summaries,
            trace.final_label_probabilities[1],
            trace.selected_label,
        )

    _, values = jax.lax.scan(_step, (eta, stored), jax.random.split(key, warmup + draws))
    parameters, paths, summaries, probabilities, labels = jax.tree.map(
        lambda value: value[warmup:], values
    )
    return MethodRun(
        parameters,
        paths,
        summaries,
        EnsembleStatistics(probabilities, labels),
        EnsembleTuning(spec.parameter_cholesky),
    )


def run_method(
    spec: MethodSpec,
    r2: jax.Array,
    key: jax.Array,
    eta_start: jax.Array,
    path_start: jax.Array,
    *,
    warmup: int,
    draws: int,
) -> MethodRun:
    """One chain, pure computation; vmap for chains, time/synchronize in the caller.

    Coordinates and smoother tuning are fixed. NumPyro adapts its metric and
    step only during warmup. MPGibbs discards warmup with fixed supplied tuning;
    a caller may estimate that tuning using a separate pilot before this run.
    """
    match spec:
        case JointNUTSSpec():
            return _joint_nuts_run(spec, key, eta_start, path_start, r2, warmup, draws)
        case NUTSSmootherSpec():
            return _nuts_smoother_run(spec, key, eta_start, path_start, r2, warmup, draws)
        case MPGibbsSpec():
            return _mpgibbs_run(spec, key, eta_start, path_start, r2, warmup, draws)
        case _:
            assert_never(spec)
