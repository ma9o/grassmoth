"""Exact daily cubic SV densities, coordinate maps, and Gaussian arithmetic.

Launch callers with JAX_PLATFORMS=cpu and JAX_ENABLE_X64=true. This pure module
does not configure the runtime, read files, or import grassmoth internals.
Functions describe one history: parameters (3,), paths (T,), chart slopes (T, 3).
Batch independent histories explicitly with jax.vmap. Squared readings are > 0;
there is no likelihood floor, clipped exponent, or linearized target.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import TYPE_CHECKING, NamedTuple, assert_never

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.typing import ArrayLike

CUBIC = 0.03
LOG_TWO_PI = log(2.0 * pi)


class Parameters(NamedTuple):
    level: ArrayLike
    decay: ArrayLike
    vol: ArrayLike


TRUTH = Parameters(-0.2, 0.02, 0.2)


class History(NamedTuple):
    h: jax.Array
    r: jax.Array
    r2: jax.Array


class DensityTerms(NamedTuple):
    prior: jax.Array
    initial: jax.Array
    transitions: jax.Array
    readings: jax.Array
    jacobian: jax.Array

    @property
    def total(self) -> jax.Array:
        return self.prior + self.initial + self.transitions + self.readings + self.jacobian


class Chart(NamedTuple):
    center: jax.Array
    slopes: jax.Array
    exponents: jax.Array
    reference: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicalCoordinates:
    """Store the physical log-variance path."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InnovationCoordinates:
    """Store independent prior driving shocks, including h0 itself."""


type Coordinates = PhysicalCoordinates | InnovationCoordinates | Chart
PHYSICAL = PhysicalCoordinates()
INNOVATIONS = InnovationCoordinates()


def to_eta(level: ArrayLike, decay: ArrayLike, vol: ArrayLike) -> jax.Array:
    return jnp.stack((jnp.asarray(level), jnp.log(decay), jnp.log(vol)))


def from_eta(eta: jax.Array) -> Parameters:
    return Parameters(eta[0], jnp.exp(eta[1]), jnp.exp(eta[2]))


def normal_logpdf(value: ArrayLike, mean: ArrayLike, variance: ArrayLike) -> jax.Array:
    return -0.5 * (LOG_TWO_PI + jnp.log(variance) + (jnp.asarray(value) - mean) ** 2 / variance)


def log_prior(eta: jax.Array) -> jax.Array:
    """Normalized prior density in eta, including both logarithm Jacobians."""
    return (
        normal_logpdf(eta[0], 0.0, 4.0)
        + normal_logpdf(jnp.exp(eta[1]), 0.0, 0.1**2)
        + log(2.0)
        + eta[1]
        + normal_logpdf(jnp.exp(eta[2]), 0.0, 0.5**2)
        + log(2.0)
        + eta[2]
    )


def transition_mean(eta: jax.Array, h: jax.Array) -> jax.Array:
    return (1.0 - jnp.exp(eta[1])) * h - CUBIC * h**3


def transition_logpdf(eta: jax.Array, previous: jax.Array, current: jax.Array) -> jax.Array:
    return normal_logpdf(current, transition_mean(eta, previous), jnp.exp(2.0 * eta[2]))


def initial_logpdf(h: jax.Array) -> jax.Array:
    return normal_logpdf(h, 0.0, 1.0)


def residuals(eta: jax.Array, h: jax.Array) -> jax.Array:
    """Unstandardized nonlinear transition residuals, shape (T-1,)."""
    return h[1:] - transition_mean(eta, h[:-1])


def reading_log_potential(eta: jax.Array, h: jax.Array, r2: jax.Array) -> jax.Array:
    """Per-day log likelihood with data-only constants omitted."""
    predictor = eta[0] + h
    return -0.5 * (predictor + r2 * jnp.exp(-predictor))


def reading_logpdf(eta: jax.Array, h: jax.Array, r2: jax.Array) -> jax.Array:
    """Normalized Gamma(1/2, rate=exp(-mu-h)/2) log density, per day."""
    return reading_log_potential(eta, h, r2) - 0.5 * (LOG_TWO_PI + jnp.log(r2))


def reading_score(eta: jax.Array, h: jax.Array, r2: jax.Array) -> jax.Array:
    return 0.5 * (r2 * jnp.exp(-eta[0] - h) - 1.0)


def reading_curvature(eta: jax.Array, h: jax.Array, r2: jax.Array) -> jax.Array:
    """Nonnegative negative second derivative with respect to physical h."""
    return 0.5 * r2 * jnp.exp(-eta[0] - h)


def joint_terms(eta: jax.Array, h: jax.Array, r2: jax.Array) -> DensityTerms:
    return DensityTerms(
        log_prior(eta),
        initial_logpdf(h[0]),
        jnp.sum(transition_logpdf(eta, h[:-1], h[1:])),
        jnp.sum(reading_logpdf(eta, h, r2)),
        jnp.zeros_like(eta[0]),
    )


def log_joint(eta: jax.Array, h: jax.Array, r2: jax.Array) -> jax.Array:
    return joint_terms(eta, h, r2).total


def shocks(eta: jax.Array, h: jax.Array) -> jax.Array:
    return jnp.concatenate((h[:1], residuals(eta, h) * jnp.exp(-eta[2])))


def rebuild(eta: jax.Array, eps: jax.Array) -> jax.Array:
    def _step(previous: jax.Array, noise: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = transition_mean(eta, previous) + jnp.exp(eta[2]) * noise
        return current, current

    _, tail = jax.lax.scan(_step, eps[0], eps[1:])
    return jnp.concatenate((eps[:1], tail))


def simulate(days: int, seed: int, truth: Parameters = TRUTH) -> History:
    """Daily EM simulation; any shorter call with this seed is an exact prefix."""
    if days < 1:
        raise ValueError("A history needs at least one observed day.")
    path_key, reading_key = jax.random.split(jax.random.key(seed))
    indices = jnp.arange(days)

    def _noise(key: jax.Array) -> jax.Array:
        return jax.vmap(lambda index: jax.random.normal(jax.random.fold_in(key, index)))(indices)

    eta = to_eta(*truth)
    h = rebuild(eta, _noise(path_key))
    r = jnp.exp(0.5 * (eta[0] + h)) * _noise(reading_key)
    return History(h, r, r**2)


def decode(chart: Chart, eta: jax.Array, z: jax.Array) -> jax.Array:
    difference = eta - chart.reference
    return (
        chart.center
        + chart.slopes @ difference
        + jnp.exp(chart.exponents @ difference) * (z - chart.center)
    )


def encode(chart: Chart, eta: jax.Array, h: jax.Array) -> jax.Array:
    difference = eta - chart.reference
    return chart.center + (h - chart.center - chart.slopes @ difference) * jnp.exp(
        -chart.exponents @ difference
    )


def log_jacobian(chart: Chart, eta: jax.Array) -> jax.Array:
    return jnp.sum(chart.exponents @ (eta - chart.reference))


def chart_ablation(chart: Chart, *, translation: bool = True, scale: bool = True) -> Chart:
    return Chart(
        chart.center,
        chart.slopes if translation else jnp.zeros_like(chart.slopes),
        chart.exponents if scale else jnp.zeros_like(chart.exponents),
        chart.reference,
    )


def identity_chart(reference: jax.Array, days: int) -> Chart:
    return Chart(
        jnp.zeros(days, dtype=reference.dtype),
        jnp.zeros((days, 3), dtype=reference.dtype),
        jnp.zeros((days, 3), dtype=reference.dtype),
        reference,
    )


def stored_path(coordinates: Coordinates, eta: jax.Array, h: jax.Array) -> jax.Array:
    match coordinates:
        case PhysicalCoordinates():
            return h
        case InnovationCoordinates():
            return shocks(eta, h)
        case Chart():
            return encode(coordinates, eta, h)
        case _:
            assert_never(coordinates)


def physical_path(coordinates: Coordinates, eta: jax.Array, held: jax.Array) -> jax.Array:
    match coordinates:
        case PhysicalCoordinates():
            return held
        case InnovationCoordinates():
            return rebuild(eta, held)
        case Chart():
            return decode(coordinates, eta, held)
        case _:
            assert_never(coordinates)


def held_terms(
    coordinates: Coordinates, eta: jax.Array, held: jax.Array, r2: jax.Array
) -> DensityTerms:
    """Physical-density ledger; the last field pays the coordinate Jacobian."""
    terms = joint_terms(eta, physical_path(coordinates, eta, held), r2)
    match coordinates:
        case PhysicalCoordinates():
            jacobian = jnp.zeros_like(eta[0])
        case InnovationCoordinates():
            jacobian = (held.shape[0] - 1) * eta[2]
        case Chart():
            jacobian = log_jacobian(coordinates, eta)
        case _:
            assert_never(coordinates)
    return DensityTerms(*terms[:4], jacobian)


def held_log_density(
    coordinates: Coordinates, eta: jax.Array, held: jax.Array, r2: jax.Array
) -> jax.Array:
    """Exact density in the chosen coordinates, including all normalizers."""
    if isinstance(coordinates, InnovationCoordinates):
        return (
            log_prior(eta)
            + jnp.sum(initial_logpdf(held))
            + jnp.sum(reading_logpdf(eta, rebuild(eta, held), r2))
        )
    return held_terms(coordinates, eta, held, r2).total


class GaussianPath(NamedTuple):
    """Normalized Gaussian: precision L diag(pivots) L.T, L unit lower bidiagonal."""

    mean: jax.Array
    pivots: jax.Array
    multipliers: jax.Array


class GaussianMarginals(NamedTuple):
    variance: jax.Array


def _precision_factor(diagonal: jax.Array, off_diagonal: jax.Array) -> tuple[jax.Array, jax.Array]:
    def _step(previous: jax.Array, values: tuple[jax.Array, jax.Array]):
        current, off = values
        multiplier = off / previous
        pivot = current - multiplier * off
        return pivot, (pivot, multiplier)

    _, (tail, multipliers) = jax.lax.scan(_step, diagonal[0], (diagonal[1:], off_diagonal))
    return jnp.concatenate((diagonal[:1], tail)), multipliers


def _backsolve(multipliers: jax.Array, rhs: jax.Array) -> jax.Array:
    def _step(following: jax.Array, values: tuple[jax.Array, jax.Array]):
        multiplier, value = values
        current = value - multiplier * following
        return current, current

    _, head = jax.lax.scan(_step, rhs[-1], (multipliers, rhs[:-1]), reverse=True)
    return jnp.concatenate((head, rhs[-1:]))


def _precision_solve(pivots: jax.Array, multipliers: jax.Array, rhs: jax.Array) -> jax.Array:
    def _step(previous: jax.Array, values: tuple[jax.Array, jax.Array]):
        multiplier, value = values
        current = value - multiplier * previous
        return current, current

    _, tail = jax.lax.scan(_step, rhs[0], (multipliers, rhs[1:]))
    forward = jnp.concatenate((rhs[:1], tail))
    return _backsolve(multipliers, forward / pivots)


def gaussian_from_precision(
    diagonal: jax.Array, off_diagonal: jax.Array, information: jax.Array
) -> GaussianPath:
    """Construct from SPD tridiagonal precision and information; no regularization."""
    pivots, multipliers = _precision_factor(diagonal, off_diagonal)
    return GaussianPath(_precision_solve(pivots, multipliers, information), pivots, multipliers)


def gaussian_logpdf(distribution: GaussianPath, path: jax.Array) -> jax.Array:
    residual = path - distribution.mean
    transformed = residual.at[:-1].add(distribution.multipliers * residual[1:])
    return 0.5 * (
        jnp.sum(jnp.log(distribution.pivots))
        - path.shape[0] * LOG_TWO_PI
        - jnp.sum(distribution.pivots * transformed**2)
    )


def gaussian_sample(key: jax.Array, distribution: GaussianPath) -> jax.Array:
    noise = jax.random.normal(key, distribution.mean.shape, dtype=distribution.mean.dtype)
    return distribution.mean + _backsolve(
        distribution.multipliers, noise / jnp.sqrt(distribution.pivots)
    )


def gaussian_marginals(distribution: GaussianPath) -> GaussianMarginals:
    def _step(following: jax.Array, values: tuple[jax.Array, jax.Array]):
        pivot, multiplier = values
        variance = 1.0 / pivot + multiplier**2 * following
        return variance, variance

    terminal = 1.0 / distribution.pivots[-1]
    _, head = jax.lax.scan(
        _step, terminal, (distribution.pivots[:-1], distribution.multipliers), reverse=True
    )
    return GaussianMarginals(jnp.concatenate((head, terminal[None])))


class FrozenSurrogate(NamedTuple):
    path: jax.Array
    transition_slopes: jax.Array
    reading_predictor: jax.Array
    reading_precision: jax.Array
    reading_gradient: jax.Array


def freeze_surrogate(eta: jax.Array, path: jax.Array, r2: jax.Array) -> FrozenSurrogate:
    return FrozenSurrogate(
        path,
        1.0 - jnp.exp(eta[1]) - 3.0 * CUBIC * path[:-1] ** 2,
        eta[0] + path,
        reading_curvature(eta, path, r2),
        reading_score(eta, path, r2),
    )


def surrogate_precision(
    eta: jax.Array, frozen: FrozenSurrogate
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Frozen Jacobians/reading quadratic; vary offsets and transition covariance."""
    slopes = frozen.transition_slopes
    offsets = transition_mean(eta, frozen.path[:-1]) - slopes * frozen.path[:-1]
    precision = jnp.exp(-2.0 * eta[2])
    diagonal = frozen.reading_precision.at[0].add(1.0)
    diagonal = diagonal.at[:-1].add(slopes**2 * precision).at[1:].add(precision)
    information = frozen.reading_gradient + frozen.reading_precision * (
        frozen.reading_predictor - eta[0]
    )
    information = (
        information.at[:-1].add(-slopes * offsets * precision).at[1:].add(offsets * precision)
    )
    return diagonal, -slopes * precision, information


def surrogate_path(eta: jax.Array, frozen: FrozenSurrogate) -> GaussianPath:
    return gaussian_from_precision(*surrogate_precision(eta, frozen))


def refine_path(
    eta: jax.Array,
    path_start: jax.Array,
    r2: jax.Array,
    *,
    iterations: int = 12,
    backtracking_steps: int = 16,
) -> jax.Array:
    """Finite Gauss--Newton refinement with Armijo backtracking on the true target."""

    def _iteration(_: int, path: jax.Array) -> jax.Array:
        proposal = surrogate_path(eta, freeze_surrogate(eta, path, r2)).mean
        direction = proposal - path
        density, gradient = jax.value_and_grad(lambda value: log_joint(eta, value, r2))(path)
        slope = jnp.dot(gradient, direction)

        def _backtrack(index: int, state: tuple[jax.Array, jax.Array]):
            selected, accepted = state
            fraction = 0.5**index
            candidate = path + fraction * direction
            qualifies = log_joint(eta, candidate, r2) >= density + 1e-4 * fraction * slope
            choose = qualifies & ~accepted
            return jnp.where(choose, candidate, selected), accepted | qualifies

        result, _ = jax.lax.fori_loop(0, backtracking_steps, _backtrack, (path, jnp.asarray(False)))
        return result

    return jax.lax.fori_loop(0, iterations, _iteration, path_start)


def fit_chart(
    eta_reference: jax.Array, path_start: jax.Array, r2: jax.Array, *, iterations: int = 12
) -> Chart:
    """Freeze a refined surrogate; differentiate its mean, not its construction.

    Structural log-vol scaling applies to every stored deviation, including h0.
    The physical initial distribution remains N(0,1); its chart factor is exact.
    """
    path = refine_path(eta_reference, path_start, r2, iterations=iterations)
    frozen = freeze_surrogate(eta_reference, path, r2)

    def _mean(eta: jax.Array) -> jax.Array:
        return surrogate_path(eta, frozen).mean

    center = _mean(eta_reference)
    slopes = jax.jacfwd(_mean)(eta_reference)
    exponents = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), slopes.shape)
    return Chart(center, slopes, exponents, eta_reference)


class LinearGaussianModel(NamedTuple):
    initial_mean: jax.Array
    initial_variance: jax.Array
    transition_slopes: jax.Array
    transition_offsets: jax.Array
    transition_variances: jax.Array
    observation_slopes: jax.Array
    observation_offsets: jax.Array
    observation_variances: jax.Array


class KalmanFilterResult(NamedTuple):
    log_likelihood_terms: jax.Array

    @property
    def log_likelihood(self) -> jax.Array:
        return jnp.sum(self.log_likelihood_terms)


def linear_gaussian_twin(
    days: int,
    *,
    decay: ArrayLike = 0.02,
    vol: ArrayLike = 0.2,
    level: ArrayLike = -0.2,
    reading_sd: ArrayLike = 0.5,
) -> LinearGaussianModel:
    """Separate oracle: linear drift and additive Gaussian observations of level+h."""
    return LinearGaussianModel(
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.full(days - 1, 1.0 - jnp.asarray(decay)),
        jnp.zeros(days - 1),
        jnp.full(days - 1, jnp.asarray(vol) ** 2),
        jnp.ones(days),
        jnp.full(days, level),
        jnp.full(days, jnp.asarray(reading_sd) ** 2),
    )


def linear_gaussian_posterior(model: LinearGaussianModel, readings: jax.Array) -> GaussianPath:
    precision = 1.0 / model.transition_variances
    slopes, offsets = model.transition_slopes, model.transition_offsets
    diagonal = model.observation_slopes**2 / model.observation_variances
    information = (
        model.observation_slopes
        * (readings - model.observation_offsets)
        / model.observation_variances
    )
    diagonal = diagonal.at[0].add(1.0 / model.initial_variance)
    information = information.at[0].add(model.initial_mean / model.initial_variance)
    diagonal = diagonal.at[:-1].add(slopes**2 * precision).at[1:].add(precision)
    information = (
        information.at[:-1].add(-slopes * offsets * precision).at[1:].add(offsets * precision)
    )
    return gaussian_from_precision(diagonal, -slopes * precision, information)


def kalman_filter(model: LinearGaussianModel, readings: jax.Array) -> KalmanFilterResult:
    def _step(carry: tuple[jax.Array, jax.Array], values: tuple[jax.Array, ...]):
        mean, variance = carry
        slope, offset, noise, loading, intercept, reading_noise, reading = values
        prediction = slope * mean + offset
        prediction_variance = slope**2 * variance + noise
        innovation = reading - intercept - loading * prediction
        innovation_variance = loading**2 * prediction_variance + reading_noise
        gain = prediction_variance * loading / innovation_variance
        updated_mean = prediction + gain * innovation
        updated_variance = prediction_variance * reading_noise / innovation_variance
        log_likelihood = normal_logpdf(innovation, 0.0, innovation_variance)
        return (updated_mean, updated_variance), (
            prediction,
            prediction_variance,
            updated_mean,
            updated_variance,
            log_likelihood,
        )

    values = (
        jnp.concatenate((jnp.zeros(1), model.transition_slopes)),
        jnp.concatenate((model.initial_mean[None], model.transition_offsets)),
        jnp.concatenate((model.initial_variance[None], model.transition_variances)),
        model.observation_slopes,
        model.observation_offsets,
        model.observation_variances,
        readings,
    )
    _, result = jax.lax.scan(_step, (jnp.asarray(0.0), jnp.asarray(0.0)), values)
    return KalmanFilterResult(result[-1])


def linear_gaussian_log_joint(
    model: LinearGaussianModel, path: jax.Array, readings: jax.Array
) -> jax.Array:
    return (
        normal_logpdf(path[0], model.initial_mean, model.initial_variance)
        + jnp.sum(
            normal_logpdf(
                path[1:],
                model.transition_slopes * path[:-1] + model.transition_offsets,
                model.transition_variances,
            )
        )
        + jnp.sum(
            normal_logpdf(
                readings,
                model.observation_slopes * path + model.observation_offsets,
                model.observation_variances,
            )
        )
    )
