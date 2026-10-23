"""Support code for the latent-path proposals walkthrough notebook.

A one-dimensional stochastic process on a time grid, observed through readings, and
seven exact MCMC updates of its whole path. Six of them first jitter the current
path into an auxiliary point ``u ~ N(x, δ/2)`` and then draw a new path near ``u``.
They differ in how the candidates at each step are drawn:

- **Separable** candidates ignore the other steps, so dSMC can stitch them in
  parallel: Particle-RWM (today's dSMC proposal), Particle-aMALA (``u`` shifted by
  the gradient of the step's log density), and smoother marginals (no ``u``:
  candidates from a Gaussian smoother's estimate of each step).
- **Sequential** candidates extend a candidate at the previous step through the
  dynamics, in a conditional SMC with backward sampling: guided (Particle-aGRAD
  with κ = 0), Particle-aGRAD (κ = 1: ``u`` shifted by the reading's gradient) and
  twisted Particle-aGRAD (κ = 0; each proposal also sees every later ``u``).

The seventh, the auxiliary Kalman path move, draws one whole path from a Gaussian
model of the posterior and accepts or rejects it against the exact density.

Separable candidates are stitched by forward filtering and backward sampling over
the candidate grid: the exact distribution over every combination of candidates. No
stitcher does better, dSMC's tree included, so slowness cannot be blamed on the tree.
References: Particle-aMALA and Particle-aGRAD, arXiv 2401.14868 (Algorithms 1, 3 and
6, §4.4); guided proposals, arXiv 2303.00301 §3.3.2; smoother marginals, arXiv
2202.02264 §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import contourpy
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.linalg
import scipy.stats
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ── Palette: validated categorical slots 1, 2, 3 and 7 (blue, orange, aqua, violet) ──
# Colour says what a proposal adds; line style and marker say separable or sequential.
C_PLAIN = "#2a78d6"
C_GRADIENT = "#eb6834"
C_AHEAD = "#1baf7a"
C_KALMAN = "#6250d6"
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#9aa0a6"
C_GRID = "#e6e5e1"
C_LINES = "#d3d2cc"
C_SURFACE = "#fcfcfb"

UPDATES = ("local", "amala", "marginal", "guided", "agrad", "twisted", "kalman")
SEPARABLE = ("local", "amala", "marginal")
SEQUENTIAL = ("guided", "agrad", "twisted")
NAMES = {
    "local": "Particle-RWM (today's dSMC)",
    "amala": "Particle-aMALA",
    "marginal": "Smoother marginals",
    "guided": "Guided (Particle-aGRAD, κ = 0)",
    "agrad": "Particle-aGRAD (κ = 1)",
    "twisted": "Twisted Particle-aGRAD (κ = 0)",
    "kalman": "Kalman path move",
}
SHORT = {
    "local": "today's (RWM)",
    "amala": "aMALA",
    "marginal": "smoother marginals",
    "guided": "guided",
    "agrad": "aGRAD",
    "twisted": "twisted",
    "kalman": "Kalman move",
}
COLORS = {
    "local": C_PLAIN,
    "guided": C_PLAIN,
    "amala": C_GRADIENT,
    "agrad": C_GRADIENT,
    "marginal": C_AHEAD,
    "twisted": C_AHEAD,
    "kalman": C_KALMAN,
}
DASH = {kind: "dash" if kind in SEQUENTIAL else "solid" for kind in UPDATES}
SYMBOL = {
    kind: "diamond" if kind in SEQUENTIAL else "square" if kind == "kalman" else "circle"
    for kind in UPDATES
}

# ── C0 ground truth for the toy: decay, input effect and noise as in C0's x ──
DECAY = 0.6
FORCING = 0.5
DIFFUSION = 0.45
NOISE = 0.2
INITIAL_SD = 0.4
DAYS = 2.0
BINARY_WIDTH = 0.05  # binary readings: P(reading = 1) = Φ(x / BINARY_WIDTH)


@dataclass(frozen=True)
class Toy:
    """One simulated history on a grid, with its readings (noisy values, or binary)."""

    dt: float
    cubic: float
    forcing: np.ndarray
    observed: np.ndarray
    readings: np.ndarray
    truth: np.ndarray
    binary: bool = False

    @property
    def steps(self) -> int:
        return int(self.truth.size)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.steps) * self.dt


def make_toy(
    *,
    intervals_per_day: int,
    every: int = 1,
    cubic: float = 0.0,
    days: float = DAYS,
    binary: bool = False,
    seed: int = 7,
) -> Toy:
    """Simulate dx = (-a x - c x^3 + b u) dt + s dW on the grid; read every `every` steps.

    Binary readings say whether the state is above zero, through a soft threshold.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / intervals_per_day
    steps = round(days * intervals_per_day)
    times = np.arange(steps) * dt
    forcing = np.where(np.floor(times * 2).astype(int) % 2 == 0, 1.0, -1.0)
    truth = np.zeros(steps)
    truth[0] = INITIAL_SD * rng.normal()
    for t in range(1, steps):
        previous = truth[t - 1]
        drift = -DECAY * previous - cubic * previous**3 + FORCING * forcing[t - 1]
        truth[t] = previous + drift * dt + DIFFUSION * np.sqrt(dt) * rng.normal()
    observed = np.arange(steps) % every == 0
    observed[-1] = True
    values = (truth > 0.0).astype(float) if binary else truth + NOISE * rng.normal(size=steps)
    readings = np.where(observed, values, np.nan)
    return Toy(dt, cubic, forcing, observed, readings, truth, binary)


def two_steps(intervals_per_day: int, *, binary: bool = False, seed: int = 1) -> Toy:
    """Two neighbouring grid steps, each with a reading."""
    return make_toy(
        intervals_per_day=intervals_per_day, days=2.0 / intervals_per_day, binary=binary, seed=seed
    )


def _transition_mean(toy_dt, cubic, forcing, previous):
    drift = -DECAY * previous - cubic * previous**3 + FORCING * forcing
    return previous + drift * toy_dt


def log_density(toy: Toy, path: np.ndarray) -> np.ndarray:
    """Exact Euler–Maruyama path density with the readings, for one path or a stack of them."""
    variance = DIFFUSION**2 * toy.dt
    mean = _transition_mean(toy.dt, toy.cubic, toy.forcing[:-1], path[..., :-1])
    values, readings = path[..., toy.observed], toy.readings[toy.observed]
    if toy.binary:
        fit = scipy.stats.norm.logcdf((2.0 * readings - 1.0) * values / BINARY_WIDTH)
    else:
        fit = -0.5 * (values - readings) ** 2 / NOISE**2
    return (
        -0.5 * path[..., 0] ** 2 / INITIAL_SD**2
        - 0.5 * np.sum((path[..., 1:] - mean) ** 2, axis=-1) / variance
        + np.sum(fit, axis=-1)
    )


@dataclass(frozen=True)
class GaussianModel:
    """Gaussian path model in information form: log density -x'Λx/2 + h'x."""

    precision: np.ndarray
    shift: np.ndarray

    def log_density(self, path: np.ndarray) -> float:
        return float(-0.5 * path @ self.precision @ path + self.shift @ path)

    @property
    def mean(self) -> np.ndarray:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(self.precision), self.shift)

    @property
    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.precision)


def gaussian_model(toy: Toy, around: np.ndarray) -> GaussianModel:
    """Expand the transition means around a path; noisy-value readings are Gaussian already."""
    steps, variance = toy.steps, DIFFUSION**2 * toy.dt
    slope = 1.0 + (-DECAY - 3.0 * toy.cubic * around[:-1] ** 2) * toy.dt
    offset = (
        _transition_mean(toy.dt, toy.cubic, toy.forcing[:-1], around[:-1]) - slope * around[:-1]
    )
    precision = np.zeros((steps, steps))
    shift = np.zeros(steps)
    precision[0, 0] += 1.0 / INITIAL_SD**2
    index = np.arange(1, steps)
    precision[index, index] += 1.0 / variance
    precision[index - 1, index - 1] += slope**2 / variance
    precision[index, index - 1] -= slope / variance
    precision[index - 1, index] -= slope / variance
    shift[index] += offset / variance
    shift[index - 1] -= slope * offset / variance
    precision[toy.observed, toy.observed] += 1.0 / NOISE**2
    shift[toy.observed] += toy.readings[toy.observed] / NOISE**2
    return GaussianModel(precision, shift)


def reference_model(toy: Toy) -> GaussianModel:
    """Gaussian model expanded around its own linear-dynamics mean (exact when cubic is 0)."""
    linear = gaussian_model(
        Toy(toy.dt, 0.0, toy.forcing, toy.observed, toy.readings, toy.truth), np.zeros(toy.steps)
    )
    return gaussian_model(toy, linear.mean)


def dynamics_model(toy: Toy) -> GaussianModel:
    """What the linear dynamics alone say about the path, before any reading."""
    unread = Toy(toy.dt, 0.0, toy.forcing, np.zeros(toy.steps, dtype=bool), toy.readings, toy.truth)
    return gaussian_model(unread, np.zeros(toy.steps))


def readings_model(toy: Toy) -> GaussianModel:
    """What noisy-value readings alone say, as a function of the path (every step read)."""
    return GaussianModel(np.eye(toy.steps) / NOISE**2, toy.readings / NOISE**2)


def auxiliary_model(model: GaussianModel, auxiliary: np.ndarray, delta: float) -> GaussianModel:
    """The posterior times N(u; x, δ/2): where an update's second stage draws."""
    return GaussianModel(
        model.precision + 2.0 / delta * np.eye(auxiliary.size),
        model.shift + 2.0 / delta * auxiliary,
    )


def posterior_draws(model: GaussianModel, count: int, seed: int) -> np.ndarray:
    """Exact draws from the Gaussian model."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(model.precision)
    noise = rng.normal(size=(model.shift.size, count))
    return (model.mean[:, None] + scipy.linalg.solve_triangular(chol.T, noise, lower=False)).T


# ── Two-step latents: posteriors you can draw ───────────────────────────────


@dataclass(frozen=True)
class Plane:
    """A two-step latent: its posterior's 86% and 39% regions, a Gaussian with the posterior's
    mean and covariance, posterior draws, and the current path every update starts from."""

    label: str
    toy: Toy
    model: GaussianModel
    outer: tuple[np.ndarray, ...]
    inner: tuple[np.ndarray, ...]
    draws: np.ndarray
    start: np.ndarray


def _ellipse(mean: np.ndarray, covariance: np.ndarray, radius: float) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 121)
    circle = np.vstack((np.cos(angle), np.sin(angle)))
    return (mean[:, None] + radius * np.linalg.cholesky(covariance) @ circle).T


def make_plane(label: str, *, intervals_per_day: int, binary: bool = False, seed: int = 1) -> Plane:
    """Build a two-step latent. With binary readings the posterior is computed on a fine grid."""
    toy = two_steps(intervals_per_day, binary=binary, seed=seed)
    if not binary:
        model = reference_model(toy)
        draws = posterior_draws(model, 400, seed=seed + 10)
        return Plane(
            label,
            toy,
            model,
            (_ellipse(model.mean, model.covariance, 2.0),),
            (_ellipse(model.mean, model.covariance, 1.0),),
            draws,
            draws[0],
        )
    span = 4.0 * INITIAL_SD
    first = np.linspace(-span, span, 801)
    second = np.linspace(-span, span, 801)
    grid = np.stack(np.meshgrid(first, second, indexing="xy"), axis=-1)
    log_values = log_density(toy, grid)
    density = np.exp(log_values - log_values.max())
    density /= density.sum()
    flat = grid.reshape(-1, 2)
    mean = density.ravel() @ flat
    centred = flat - mean
    covariance = (centred * density.ravel()[:, None]).T @ centred
    precision = np.linalg.inv(covariance)
    ordered = np.sort(density.ravel())[::-1]
    mass = np.cumsum(ordered)
    generator = contourpy.contour_generator(x=first, y=second, z=density)

    def region(level: float) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(line)
            for line in generator.lines(float(ordered[np.searchsorted(mass, level)]))
        )

    rng = np.random.default_rng(seed + 10)
    cells = rng.choice(flat.shape[0], size=400, p=density.ravel())
    cell = first[1] - first[0]
    draws = flat[cells] + rng.uniform(-0.5 * cell, 0.5 * cell, size=(400, 2))
    return Plane(
        label,
        toy,
        GaussianModel(precision, precision @ mean),
        region(0.865),
        region(0.393),
        draws,
        draws[0],
    )


# ── Separable updates, stitched exactly (forward filtering, backward sampling) ──


class Stitch(NamedTuple):
    """One separable update: the picked path, the candidate grid and its stitching scores."""

    path: jax.Array
    pool: jax.Array
    auxiliary: jax.Array
    filtered: jax.Array
    links: jax.Array


def _arrays(toy: Toy):
    return (
        jnp.asarray(toy.dt),
        jnp.asarray(toy.cubic),
        jnp.asarray(toy.forcing),
        jnp.asarray(toy.observed),
        jnp.asarray(toy.readings),
        jnp.asarray(toy.binary),
    )


def _reading(arrays, values, t):
    """Log density of the reading at step t; zero where there is none."""
    _, _, _, observed, readings, binary = arrays
    reading = jnp.nan_to_num(readings[t])
    threshold = norm.logcdf((2.0 * reading - 1.0) * values / BINARY_WIDTH)
    return jnp.where(
        observed[t], jnp.where(binary, threshold, norm.logpdf(reading, values, NOISE)), 0.0
    )


def _reading_gradient(arrays, values, t):
    _, _, _, observed, readings, binary = arrays
    reading = jnp.nan_to_num(readings[t])
    sign = 2.0 * reading - 1.0
    scaled = sign * values / BINARY_WIDTH
    threshold = sign / BINARY_WIDTH * jnp.exp(norm.logpdf(scaled) - norm.logcdf(scaled))
    return jnp.where(observed[t], jnp.where(binary, threshold, (reading - values) / NOISE**2), 0.0)


@partial(jax.jit, static_argnames=("particles", "kind"))
def _separable_sweep(
    key, reference, arrays, delta, marginal_mean, marginal_sd, *, particles, kind
) -> Stitch:
    dt, cubic, forcing = arrays[:3]
    steps = reference.shape[0]
    variance = DIFFUSION**2 * dt
    half = 0.5 * delta
    times = jnp.arange(steps)
    auxiliary_key, draw_key, back_key = jax.random.split(key, 3)

    def gradient(previous, values, t):
        """Gradient in x_t of the filtering density: the link from x_{t-1} plus the reading."""
        mean = _transition_mean(dt, cubic, forcing[jnp.maximum(t, 1) - 1], previous)
        prior = jnp.where(t == 0, -values / INITIAL_SD**2, -(values - mean) / variance)
        return prior + _reading_gradient(arrays, values, t)

    noise = jax.random.normal(draw_key, (steps, particles - 1))
    if kind == "marginal":
        auxiliary = marginal_mean
        candidates = marginal_mean[:, None] + marginal_sd[:, None] * noise
    else:
        kappa = 1.0 if kind == "amala" else 0.0
        before = jnp.concatenate((reference[:1], reference[:-1]))
        drift = kappa * half * jax.vmap(gradient)(before, reference, times)
        auxiliary = reference + drift + jnp.sqrt(half) * jax.random.normal(auxiliary_key, (steps,))
        candidates = auxiliary[:, None] + jnp.sqrt(half) * noise
    pool = jnp.concatenate((reference[:, None], candidates), axis=1)
    unary = jax.vmap(lambda values, t: _reading(arrays, values, t))(pool, times)
    if kind == "marginal":
        # Independent proposals pay their density; the auxiliary proposal's density
        # cancels against its auxiliary factor, so it pays nothing.
        unary = unary - norm.logpdf(pool, marginal_mean[:, None], marginal_sd[:, None])

    def tilt(previous, values, t):
        """aMALA's auxiliary factor N(u; x + (δ/2)∇, δ/2) over the proposal's N(u; x, δ/2)."""
        shifted = values + half * gradient(previous, values, t)
        return norm.logpdf(auxiliary[t], shifted, jnp.sqrt(half)) - norm.logpdf(
            auxiliary[t], values, jnp.sqrt(half)
        )

    first = unary[0] + norm.logpdf(pool[0], 0.0, INITIAL_SD)
    if kind == "amala":
        first = first + tilt(pool[0], pool[0], 0)

    def link(t):
        previous = pool[t - 1][None, :]
        current = pool[t][:, None]
        score = (
            -0.5 * (current - _transition_mean(dt, cubic, forcing[t - 1], previous)) ** 2 / variance
        )
        if kind == "amala":
            score = score + tilt(previous, current, t)
        return score

    links = jax.vmap(link)(jnp.arange(1, steps))

    def forward(previous, t):
        current = unary[t] + logsumexp(previous[None, :] + links[t - 1], axis=1)
        return current, current

    _, rest = jax.lax.scan(forward, first, jnp.arange(1, steps))
    filtered = jnp.concatenate((first[None], rest))
    last = jax.random.categorical(back_key, filtered[-1])

    def backward(chosen, inputs):
        t, step_key = inputs
        index = jax.random.categorical(step_key, filtered[t] + links[t][chosen])
        return index, index

    keys = jax.random.split(jax.random.fold_in(back_key, 1), steps - 1)
    _, earlier = jax.lax.scan(backward, last, (jnp.arange(steps - 2, -1, -1), keys))
    indices = jnp.concatenate((earlier[::-1], last[None]))
    return Stitch(pool[jnp.arange(steps), indices], pool, auxiliary, filtered, links)


# ── Sequential updates: conditional SMC with backward sampling ──────────────────


class Lineage(NamedTuple):
    """One sequential update: the picked path, every particle, and how they descend."""

    path: jax.Array
    bank: jax.Array
    ancestors: jax.Array
    log_weights: jax.Array
    auxiliary: jax.Array
    twist: jax.Array


@partial(jax.jit, static_argnames=("particles", "kind"))
def _sequential_sweep(key, reference, arrays, delta, *, particles, kind) -> Lineage:
    """Particle n at step t is drawn from N(x_t; m(x_{t-1}^{a_n}), q) N(u_t; x_t, δ/2) ψ_t(x_t).

    ψ_t(x) = p(u_{t+1:T} | x) under the dynamics linearized at u for the twisted update,
    and 1 otherwise. The linearization point must not be the current path: the particle
    system targets the path given u, so its proposals may depend on u but not on the
    path it replaces. The particle system targets the twisted filters, so backward
    sampling divides ψ_t back out (Corenflos & Finke §4.4).
    """
    dt, cubic, forcing = arrays[:3]
    steps = reference.shape[0]
    variance = DIFFUSION**2 * dt
    half = 0.5 * delta
    times = jnp.arange(steps)
    auxiliary_key, first_key, forward_key, back_key = jax.random.split(key, 4)
    kappa = 1.0 if kind == "agrad" else 0.0
    drift = kappa * half * _reading_gradient(arrays, reference, times)
    auxiliary = reference + drift + jnp.sqrt(half) * jax.random.normal(auxiliary_key, (steps,))
    zero = jnp.zeros((), reference.dtype)
    if kind == "twisted":
        before = auxiliary[:-1]
        slope = 1.0 + (-DECAY - 3.0 * cubic * before**2) * dt
        offset = _transition_mean(dt, cubic, forcing[:-1], before) - slope * before

        def absorb(later, inputs):
            precision, shift = later
            f, b, u = inputs
            joint = 1.0 / half + precision
            pull = u / half + shift
            earlier = (
                f**2 * joint / (1.0 + variance * joint),
                f * (pull - b * joint) / (1.0 + variance * joint),
            )
            return earlier, earlier

        _, (precisions, shifts) = jax.lax.scan(
            absorb, (zero, zero), (slope[::-1], offset[::-1], auxiliary[1:][::-1])
        )
        twist = jnp.stack(
            (
                jnp.concatenate((precisions[::-1], zero[None])),
                jnp.concatenate((shifts[::-1], zero[None])),
            ),
            axis=1,
        )
    else:
        twist = jnp.zeros((steps, 2), reference.dtype)

    def log_twist(values, t):
        return -0.5 * twist[t, 0] * values**2 + twist[t, 1] * values

    def tilt(values, t):
        shifted = values + kappa * half * _reading_gradient(arrays, values, t)
        return norm.logpdf(auxiliary[t], shifted, jnp.sqrt(half))

    precision0 = 1.0 / INITIAL_SD**2 + 1.0 / half + twist[0, 0]
    mean0 = (auxiliary[0] / half + twist[0, 1]) / precision0
    first = (
        (mean0 + jax.random.normal(first_key, (particles,)) / jnp.sqrt(precision0))
        .at[0]
        .set(reference[0])
    )
    first_weights = (
        norm.logpdf(first, 0.0, INITIAL_SD)
        + _reading(arrays, first, 0)
        + tilt(first, 0)
        - norm.logpdf(first, mean0, 1.0 / jnp.sqrt(precision0))
        + log_twist(first, 0)
    )

    def forward(carry, inputs):
        previous, previous_weights = carry
        t, step_key = inputs
        resample_key, propose_key = jax.random.split(step_key)
        drawn = jax.random.categorical(resample_key, previous_weights, shape=(particles - 1,))
        ancestors = jnp.concatenate((jnp.zeros(1, jnp.int32), drawn.astype(jnp.int32)))
        parent = previous[ancestors]
        prior_mean = _transition_mean(dt, cubic, forcing[t - 1], parent)
        precision = 1.0 / variance + 1.0 / half + twist[t, 0]
        mean = (prior_mean / variance + auxiliary[t] / half + twist[t, 1]) / precision
        current = (
            (mean + jax.random.normal(propose_key, (particles,)) / jnp.sqrt(precision))
            .at[0]
            .set(reference[t])
        )
        weights = (
            norm.logpdf(current, prior_mean, jnp.sqrt(variance))
            + _reading(arrays, current, t)
            + tilt(current, t)
            - norm.logpdf(current, mean, 1.0 / jnp.sqrt(precision))
            + log_twist(current, t)
            - log_twist(parent, t - 1)
        )
        return (current, weights), (current, weights, ancestors)

    keys = jax.random.split(forward_key, steps - 1)
    _, (later, later_weights, ancestors) = jax.lax.scan(
        forward, (first, first_weights), (jnp.arange(1, steps), keys)
    )
    bank = jnp.concatenate((first[None], later))
    log_weights = jnp.concatenate((first_weights[None], later_weights))
    last = jax.random.categorical(back_key, log_weights[-1])

    def backward(chosen, inputs):
        t, step_key = inputs
        prior_mean = _transition_mean(dt, cubic, forcing[t], bank[t])
        link = -0.5 * (bank[t + 1, chosen] - prior_mean) ** 2 / variance
        pick = jax.random.categorical(step_key, log_weights[t] + link - log_twist(bank[t], t))
        return pick, pick

    back_keys = jax.random.split(jax.random.fold_in(back_key, 1), steps - 1)
    _, earlier = jax.lax.scan(backward, last, (jnp.arange(steps - 2, -1, -1), back_keys))
    picks = jnp.concatenate((earlier[::-1], last[None]))
    return Lineage(bank[jnp.arange(steps), picks], bank, ancestors, log_weights, auxiliary, twist)


@partial(jax.jit, static_argnames=("kind", "particles", "sweeps"))
def _chain(key, start, arrays, delta, marginal_mean, marginal_sd, *, kind, particles, sweeps):
    def body(path, step_key):
        if kind in SEPARABLE:
            new = _separable_sweep(
                step_key,
                path,
                arrays,
                delta,
                marginal_mean,
                marginal_sd,
                particles=particles,
                kind=kind,
            ).path
        else:
            new = _sequential_sweep(
                step_key, path, arrays, delta, particles=particles, kind=kind
            ).path
        return new, new

    _, paths = jax.lax.scan(body, start, jax.random.split(key, sweeps))
    return paths


# ── The auxiliary Kalman path move ───────────────────────────────────────────


def run_kalman(
    toy: Toy, start: np.ndarray, model: GaussianModel, *, sweeps: int, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Chain of Kalman moves at δ = ∞ (independent whole-path proposals from ``model``).

    The acceptance ratio compares the exact density with the model's own density.
    """
    rng = np.random.default_rng(seed)
    proposals = posterior_draws(model, sweeps, seed=seed + 1)
    path = np.asarray(start, dtype=float)
    current = float(log_density(toy, path)) - model.log_density(path)
    paths = np.empty((sweeps, toy.steps))
    accepted = 0
    for sweep, proposal in enumerate(proposals):
        candidate = float(log_density(toy, proposal)) - model.log_density(proposal)
        if np.log(rng.uniform()) < candidate - current:
            path, current = proposal, candidate
            accepted += 1
        paths[sweep] = path
    return paths, accepted / sweeps


def run(
    kind: str,
    toy: Toy,
    start: np.ndarray,
    model: GaussianModel,
    *,
    sweeps: int,
    particles: int,
    delta: float,
    seed: int = 0,
) -> np.ndarray:
    """Chain of one update from ``start``; returns the path after each sweep.

    ``model`` is the Kalman move's Gaussian model and the smoother whose marginals the
    smoother-marginal update draws from.
    """
    if kind == "kalman":
        return run_kalman(toy, start, model, sweeps=sweeps, seed=seed)[0]
    paths = _chain(
        jax.random.key(seed),
        jnp.asarray(start, dtype=jnp.float32),
        _arrays(toy),
        delta,
        jnp.asarray(model.mean),
        jnp.asarray(np.sqrt(np.diag(model.covariance))),
        kind=kind,
        particles=particles,
        sweeps=sweeps,
    )
    return np.asarray(paths)


# ── One update, seen up close ───────────────────────────────────────────────


@dataclass(frozen=True)
class Candidates:
    """One particle update: candidates per step (column 0 is the current path) and the pick.

    ``chance`` is filled for two steps only: the probability that the update picks
    candidate i at step 1 together with candidate j at step 2.
    """

    kind: str
    values: np.ndarray
    ancestors: np.ndarray | None
    auxiliary: np.ndarray | None
    chance: np.ndarray | None
    new_path: np.ndarray


@dataclass(frozen=True)
class WholePaths:
    """One Kalman move: whole-path proposals from its Gaussian model, and the result."""

    proposals: np.ndarray
    acceptance: float
    new_path: np.ndarray


type Picture = Candidates | WholePaths


def _softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    weights = np.exp(scores - scores.max(axis=axis, keepdims=True))
    return weights / weights.sum(axis=axis, keepdims=True)


def one_update(
    kind: str,
    toy: Toy,
    start: np.ndarray,
    model: GaussianModel,
    *,
    particles: int,
    delta: float,
    seed: int = 0,
) -> Picture:
    """A single update from ``start``, with everything it proposed."""
    if kind == "kalman":
        proposals = posterior_draws(model, 12, seed=seed + 1)
        gain = float(log_density(toy, proposals[0])) - model.log_density(proposals[0])
        loss = float(log_density(toy, start)) - model.log_density(start)
        acceptance = float(min(1.0, np.exp(gain - loss)))
        accepted = np.random.default_rng(seed).uniform() < acceptance
        return WholePaths(proposals, acceptance, proposals[0] if accepted else np.asarray(start))
    key = jax.random.key(seed)
    reference = jnp.asarray(start, dtype=jnp.float32)
    if kind in SEPARABLE:
        stitch = _separable_sweep(
            key,
            reference,
            _arrays(toy),
            delta,
            jnp.asarray(model.mean),
            jnp.asarray(np.sqrt(np.diag(model.covariance))),
            particles=particles,
            kind=kind,
        )
        chance = None
        if toy.steps == 2:
            filtered, links = (
                np.asarray(stitch.filtered, dtype=float),
                np.asarray(stitch.links[0], dtype=float),
            )
            chance = (
                _softmax(filtered[1])[:, None] * _softmax(filtered[0][None, :] + links, axis=1)
            ).T
        auxiliary = None if kind == "marginal" else np.asarray(stitch.auxiliary)
        return Candidates(
            kind, np.asarray(stitch.pool), None, auxiliary, chance, np.asarray(stitch.path)
        )
    lineage = _sequential_sweep(key, reference, _arrays(toy), delta, particles=particles, kind=kind)
    bank = np.asarray(lineage.bank, dtype=float)
    chance = None
    if toy.steps == 2:
        log_weights, twist = (
            np.asarray(lineage.log_weights, dtype=float),
            np.asarray(lineage.twist, dtype=float),
        )
        variance = DIFFUSION**2 * toy.dt
        prior_mean = _transition_mean(toy.dt, toy.cubic, toy.forcing[0], bank[0])
        link = -0.5 * (bank[1][:, None] - prior_mean[None, :]) ** 2 / variance
        backward = (
            log_weights[0][None, :]
            + link
            - (-0.5 * twist[0, 0] * bank[0] ** 2 + twist[0, 1] * bank[0])
        )
        chance = (_softmax(log_weights[1])[:, None] * _softmax(backward, axis=1)).T
    return Candidates(
        kind,
        bank,
        np.asarray(lineage.ancestors),
        np.asarray(lineage.auxiliary),
        chance,
        np.asarray(lineage.path),
    )


# ── Frequency analysis ──────────────────────────────────────────────────────


def basis(steps: int) -> np.ndarray:
    """Orthonormal cosine modes, rows ordered from smoothest to wiggliest."""
    return scipy.fft.dct(np.eye(steps), norm="ortho", axis=0)


def periods(steps: int) -> np.ndarray:
    """Period of each cosine mode in grid steps (mode 0 is the constant shift)."""
    index = np.arange(steps)
    return np.where(index == 0, np.inf, 2.0 * steps / np.maximum(index, 1))


def mode_sd(model: GaussianModel) -> np.ndarray:
    modes = basis(model.shift.size)
    return np.sqrt(np.diag(modes @ model.covariance @ modes.T))


def refresh(paths: np.ndarray, model: GaussianModel) -> np.ndarray:
    """Per-sweep refresh of each mode: E[jump^2] / (2 posterior variance), about 1 - lag-1."""
    coefficients = paths @ basis(paths.shape[1]).T
    jumps = np.mean(np.diff(coefficients, axis=0) ** 2, axis=0)
    return jumps / (2.0 * mode_sd(model) ** 2)


def band_mean(values: np.ndarray, steps: int, low: float, high: float) -> float:
    """Average over modes whose period lies in [low, high) grid steps; ``high = inf`` keeps the level."""
    period = periods(steps)
    below = period <= high if np.isinf(high) else period < high
    return float(np.mean(values[(period >= low) & below]))


def smooth_projector(steps: int, min_period: float) -> np.ndarray:
    """Projection onto the cosine shapes whose period exceeds ``min_period`` steps."""
    modes = basis(steps)[periods(steps) > min_period]
    return modes.T @ modes


def binned(values: np.ndarray, steps: int, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Average per-shape values in log-spaced period bins, for a readable curve."""
    period = periods(steps)[1:]
    edges = np.geomspace(2.0, period.max() * 1.0001, bins + 1)
    centers, means = [], []
    for low, high in pairwise(edges):
        inside = (period >= low) & (period < high)
        if inside.any():
            centers.append(float(np.exp(np.mean(np.log(period[inside])))))
            means.append(float(np.mean(values[1:][inside])))
    return np.asarray(centers), np.asarray(means)


def _period_ticks(steps: int) -> list[int]:
    return [2**power for power in range(1, int(np.log2(2 * steps)) + 1)]


SMOOTH_DAYS = 0.5  # "smooth" = periods longer than half a day
WIGGLE_STEPS = 8  # "wiggles" = periods shorter than 8 grid steps


def smooth_period(toy: Toy) -> float:
    """The smooth/wiggly split in grid steps."""
    return SMOOTH_DAYS / toy.dt


def shape_summary(values: np.ndarray, toy: Toy) -> tuple[float, float]:
    """Per-sweep refresh averaged over smooth shapes and over wiggles."""
    return (
        band_mean(values, toy.steps, smooth_period(toy), np.inf),
        band_mean(values, toy.steps, 2, WIGGLE_STEPS),
    )


def axes(model: GaussianModel) -> tuple[np.ndarray, np.ndarray]:
    """Two steps: the posterior's variance along its wiggly and smooth axes, and the axes."""
    return np.linalg.eigh(model.covariance)


def axis_refresh(chain: np.ndarray, model: GaussianModel) -> tuple[float, float]:
    """Two steps: refresh per update along the smooth (long) and wiggly (short) axes."""
    variances, vectors = axes(model)
    jumps = np.mean(np.diff(chain @ vectors, axis=0) ** 2, axis=0) / (2.0 * variances)
    return float(jumps[1]), float(jumps[0])


def ceiling(delta: float, variance: float | np.ndarray) -> float | np.ndarray:
    """Best refresh any jitter-then-redraw update can reach along a direction of this variance.

    The new path is an exact draw from the posterior times N(u; x, δ/2), and u is the
    current path plus noise of variance δ/2, so the lag-1 correlation is 2v / (2v + δ).
    """
    return delta / (delta + 2.0 * variance)


def smooth_ceiling(delta: float, toy: Toy, model: GaussianModel) -> float:
    """``ceiling`` averaged over the smooth shapes of a long path."""
    smooth = periods(toy.steps) >= smooth_period(toy)
    return float(np.mean(ceiling(delta, mode_sd(model)[smooth] ** 2)))


# ── Running every update ────────────────────────────────────────────────────

PLANE_DELTAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SCAN_DELTAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class Study:
    """One update on one toy: a single update up close, and a chain from the same start."""

    kind: str
    delta: float
    picture: Picture
    chain: np.ndarray
    acceptance: float | None


def study(
    kind: str,
    toy: Toy,
    start: np.ndarray,
    model: GaussianModel,
    *,
    particles: int,
    delta: float,
    sweeps: int,
    burn: int,
    seed: int = 0,
) -> Study:
    """Run one update from ``start``: once up close, and as a chain."""
    picture = one_update(kind, toy, start, model, particles=particles, delta=delta, seed=seed)
    if kind == "kalman":
        chain, acceptance = run_kalman(toy, start, model, sweeps=sweeps, seed=seed)
        return Study(kind, delta, picture, chain[burn:], acceptance)
    chain = run(kind, toy, start, model, sweeps=sweeps, particles=particles, delta=delta, seed=seed)
    return Study(kind, delta, picture, chain[burn:], None)


def plane_study(kind: str, plane: Plane, *, particles: int, seed: int, sweeps: int = 3000) -> Study:
    """Two steps: run the update at the δ in ``PLANE_DELTAS`` that refreshes the smooth axis most."""
    delta = np.inf
    if kind not in ("marginal", "kalman"):
        scores = [
            axis_refresh(
                run(
                    kind,
                    plane.toy,
                    plane.start,
                    plane.model,
                    sweeps=sweeps,
                    particles=particles,
                    delta=value,
                ),
                plane.model,
            )[0]
            for value in PLANE_DELTAS
        ]
        delta = PLANE_DELTAS[int(np.argmax(scores))]
    return study(
        kind,
        plane.toy,
        plane.start,
        plane.model,
        particles=particles,
        delta=delta,
        sweeps=sweeps,
        burn=0,
        seed=seed,
    )


def _smooth_refresh(
    kind: str,
    toy: Toy,
    start: np.ndarray,
    model: GaussianModel,
    *,
    delta: float,
    particles: int,
    sweeps: int,
) -> float:
    chain = run(kind, toy, start, model, sweeps=sweeps, particles=particles, delta=delta)[
        sweeps // 10 :
    ]
    return shape_summary(refresh(chain, model), toy)[0]


def scan_delta(
    toy: Toy, model: GaussianModel, start: np.ndarray, *, particles: int, sweeps: int = 450
) -> dict[str, tuple[float, ...]]:
    """Smooth refresh of every particle update at every δ in ``SCAN_DELTAS``."""
    scan = {
        kind: tuple(
            _smooth_refresh(
                kind, toy, start, model, delta=delta, particles=particles, sweeps=sweeps
            )
            for delta in SCAN_DELTAS
        )
        for kind in UPDATES
        if kind not in ("marginal", "kalman")
    }
    marginal = _smooth_refresh(
        "marginal", toy, start, model, delta=np.inf, particles=particles, sweeps=sweeps
    )
    return {**scan, "marginal": (marginal,) * len(SCAN_DELTAS)}


def best_deltas(scan: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Each update's δ with the best smooth refresh in a ``scan_delta`` result."""
    best = {
        kind: SCAN_DELTAS[int(np.argmax(values))]
        for kind, values in scan.items()
        if kind != "marginal"
    }
    return {**best, "marginal": np.inf, "kalman": np.inf}


LENGTHS = (2, 8, 32, 128, 256)
LENGTH_DELTAS = {
    "local": (0.03, 0.1, 0.3, 1.0),
    "marginal": (np.inf,),
    "twisted": (0.3, 1.0, 3.0, 10.0),
    "kalman": (np.inf,),
}


def scan_length(*, particles: int, sweeps: int = 800) -> dict[str, tuple[float, ...]]:
    """Smooth refresh at each update's best δ as the path grows, on C0's grid."""
    result: dict[str, list[float]] = {kind: [] for kind in LENGTH_DELTAS}
    for steps in LENGTHS:
        toy = make_toy(intervals_per_day=128, days=steps / 128)
        model = reference_model(toy)
        start = posterior_draws(model, 1, seed=1)[0]
        for kind, deltas in LENGTH_DELTAS.items():
            result[kind].append(
                max(
                    _smooth_refresh(
                        kind, toy, start, model, delta=delta, particles=particles, sweeps=sweeps
                    )
                    for delta in deltas
                )
            )
    return {kind: tuple(values) for kind, values in result.items()}


# ── Figures ────────────────────────────────────────────────────────────────

DIVERGING = [
    [0.0, "#184f95"],
    [0.25, "#6da7ec"],
    [0.5, "#f0efec"],
    [0.75, "#ee8f8a"],
    [1.0, "#a5262a"],
]


def _layout(fig: go.Figure, *, title: str, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": C_INK}, "x": 0.0, "xanchor": "left"},
        height=height,
        paper_bgcolor=C_SURFACE,
        plot_bgcolor=C_SURFACE,
        font={"color": C_INK_2, "size": 12},
        margin={"l": 70, "r": 40, "t": 80, "b": 60},
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "font": {"size": 11}},
        showlegend=legend,
        hovermode="closest",
    )
    fig.update_annotations(font={"size": 12, "color": C_INK_2})
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID, gridwidth=1, zeroline=False, linecolor=C_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, gridwidth=1, zeroline=False, linecolor=C_GRID)
    return fig


def _dedupe_legend(fig: go.Figure) -> go.Figure:
    seen: set[str] = set()
    for trace in fig.data:
        if trace.showlegend is False or not trace.name:
            continue
        if trace.name in seen:
            trace.showlegend = False
        seen.add(trace.name)
    return fig


def _band(
    times: np.ndarray, centre: np.ndarray, half: np.ndarray, name: str, *, legend: bool
) -> go.Scatter:
    return go.Scatter(
        x=np.r_[times, times[::-1]],
        y=np.r_[centre + half, (centre - half)[::-1]],
        fill="toself",
        fillcolor="rgba(154,160,166,0.22)",
        line={"width": 0},
        name=name,
        legendgroup=name,
        showlegend=legend,
        hoverinfo="skip",
    )


# ── The plane of two steps ──


def _curves(
    fig: go.Figure,
    curves: Sequence[np.ndarray],
    *,
    row: int,
    col: int,
    name: str,
    legend: bool,
    line: Mapping[str, object],
    fill: str | None = None,
) -> None:
    for index, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=curve[:, 0],
                y=curve[:, 1],
                mode="lines",
                fill="toself" if fill else None,
                fillcolor=fill,
                line=dict(line),
                name=name,
                legendgroup=name,
                showlegend=legend and index == 0,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _posterior(fig: go.Figure, plane: Plane, *, row: int, col: int, legend: bool) -> None:
    """The posterior's 86% region shaded and its 39% region outlined (2 and 1 SD for a Gaussian)."""
    name = "posterior (86% and 39% regions)"
    _curves(
        fig,
        plane.outer,
        row=row,
        col=col,
        name=name,
        legend=legend,
        line={"width": 1.0, "color": C_MUTED},
        fill="rgba(154,160,166,0.16)",
    )
    _curves(
        fig,
        plane.inner,
        row=row,
        col=col,
        name=name,
        legend=False,
        line={"width": 1.4, "color": C_MUTED},
    )


def plane_range(plane: Plane) -> tuple[list[float], list[float]]:
    """Square axis ranges centred on the posterior, wide enough for its long axis."""
    variances, _ = axes(plane.model)
    half = 2.4 * float(np.sqrt(variances[1])) + 2.5 * float(np.sqrt(variances[0]))
    mean = plane.model.mean
    return [mean[0] - half, mean[0] + half], [mean[1] - half, mean[1] + half]


def _axis_name(index: int, letter: str) -> str:
    return letter if index == 1 else f"{letter}{index}"


def _square(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    ranges: tuple[list[float], list[float]],
    index: int,
    titles: bool = True,
) -> None:
    fig.update_xaxes(
        range=ranges[0],
        title={"text": "x₁ (step 1)" if titles else "", "standoff": 4},
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=ranges[1],
        scaleanchor=_axis_name(index, "x"),
        scaleratio=1,
        title={"text": "x₂ (step 2)" if titles else "", "standoff": 2},
        row=row,
        col=col,
    )


def _point(
    fig: go.Figure,
    xy: np.ndarray,
    *,
    row: int,
    col: int,
    name: str,
    symbol: str,
    color: str,
    size: int,
    legend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[xy[0]],
            y=[xy[1]],
            mode="markers",
            marker={
                "symbol": symbol,
                "size": size,
                "color": color,
                "line": {"width": 1.5, "color": C_SURFACE},
            },
            name=name,
            legendgroup=name,
            showlegend=legend,
            hovertemplate=f"{name}<br>x₁ %{{x:.3f}}<br>x₂ %{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _auxiliary_point(
    fig: go.Figure,
    auxiliary: np.ndarray,
    ranges: tuple[list[float], list[float]],
    *,
    row: int,
    col: int,
    index: int,
    legend: bool,
) -> None:
    """Mark u, or point at it from the panel's edge when the jitter threw it off the chart."""
    low = np.array([ranges[0][0], ranges[1][0]])
    high = np.array([ranges[0][1], ranges[1][1]])
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    inside = bool(np.all(np.abs(auxiliary - centre) <= 0.92 * half))
    _point(
        fig,
        auxiliary if inside else np.array([None, None]),
        row=row,
        col=col,
        name="auxiliary point u",
        symbol="x-thin-open",
        color=C_INK_2,
        size=11,
        legend=legend,
    )
    if inside:
        return
    direction = auxiliary - centre
    head = centre + direction * 0.93 / np.max(np.abs(direction) / half)
    tail = centre + (head - centre) * 0.62
    fig.add_annotation(
        x=float(head[0]),
        y=float(head[1]),
        ax=float(tail[0]),
        ay=float(tail[1]),
        xref=_axis_name(index, "x"),
        yref=_axis_name(index, "y"),
        axref=_axis_name(index, "x"),
        ayref=_axis_name(index, "y"),
        showarrow=True,
        arrowhead=2,
        arrowwidth=1.4,
        arrowcolor=C_INK_2,
        text="u",
        font={"size": 10, "color": C_INK_2},
    )


def _plane_update(
    fig: go.Figure,
    plane: Plane,
    picture: Picture,
    *,
    row: int,
    col: int,
    index: int,
    color: str,
    legend: bool,
) -> None:
    """One update in the plane: its candidates or proposals, and where it went."""
    _posterior(fig, plane, row=row, col=col, legend=legend)
    ranges = plane_range(plane)
    if isinstance(picture, WholePaths):
        model = plane.model
        _curves(
            fig,
            (
                _ellipse(model.mean, model.covariance, 2.0),
                _ellipse(model.mean, model.covariance, 1.0),
            ),
            row=row,
            col=col,
            name="the Kalman move's Gaussian (1 and 2 SD)",
            legend=legend,
            line={"width": 1.2, "color": color, "dash": "dash"},
        )
        fig.add_trace(
            go.Scatter(
                x=picture.proposals[:, 0],
                y=picture.proposals[:, 1],
                mode="markers",
                marker={"size": 6, "color": color, "opacity": 0.6},
                name="whole-path proposals",
                legendgroup="proposals",
                showlegend=legend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    else:
        first, second = picture.values[0], picture.values[1]
        for value in first:
            fig.add_trace(
                go.Scatter(
                    x=[value, value],
                    y=ranges[1],
                    mode="lines",
                    line={"width": 0.7, "color": C_LINES},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        for value in second:
            fig.add_trace(
                go.Scatter(
                    x=ranges[0],
                    y=[value, value],
                    mode="lines",
                    line={"width": 0.7, "color": C_LINES},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        if picture.ancestors is not None:
            fig.add_trace(
                go.Scatter(
                    x=first[picture.ancestors[0]],
                    y=second,
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 11,
                        "color": color,
                        "line": {"width": 1.4},
                    },
                    name="each step-2 candidate with its parent",
                    legendgroup="parents",
                    showlegend=legend,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        chance = (
            picture.chance if picture.chance is not None else np.zeros((first.size, second.size))
        )
        grid_x, grid_y = np.meshgrid(first, second, indexing="ij")
        relative = (chance / chance.max()).ravel()
        fig.add_trace(
            go.Scatter(
                x=grid_x.ravel(),
                y=grid_y.ravel(),
                mode="markers",
                marker={
                    "size": 3.0 + 19.0 * np.sqrt(relative),
                    "color": np.where(relative > 0.01, color, C_MUTED),
                    "opacity": 0.75,
                    "line": {"width": 0},
                },
                customdata=chance.ravel(),
                name="possible new paths, sized by their chance",
                legendgroup="crossings",
                showlegend=legend,
                hovertemplate="x₁ %{x:.3f}<br>x₂ %{y:.3f}<br>picked with probability %{customdata:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if picture.auxiliary is not None:
            _auxiliary_point(
                fig, picture.auxiliary, ranges, row=row, col=col, index=index, legend=legend
            )
    _point(
        fig,
        plane.start,
        row=row,
        col=col,
        name="current path",
        symbol="circle",
        color=C_INK,
        size=10,
        legend=legend,
    )
    _point(
        fig,
        picture.new_path,
        row=row,
        col=col,
        name="new path",
        symbol="star",
        color=color,
        size=15,
        legend=legend,
    )


def _plane_chain(
    fig: go.Figure, plane: Plane, chain: np.ndarray, *, row: int, col: int, color: str
) -> None:
    """The first 60 updates of a chain, as a path through the plane."""
    _posterior(fig, plane, row=row, col=col, legend=False)
    visited = chain[:61]
    fig.add_trace(
        go.Scatter(
            x=visited[:, 0],
            y=visited[:, 1],
            mode="lines+markers",
            line={"width": 1.1, "color": color},
            marker={"size": 4, "color": color},
            opacity=0.85,
            name="60 updates",
            showlegend=False,
            hovertemplate="x₁ %{x:.3f}<br>x₂ %{y:.3f}<extra>60 updates</extra>",
        ),
        row=row,
        col=col,
    )
    _point(
        fig,
        visited[0],
        row=row,
        col=col,
        name="current path",
        symbol="circle",
        color=C_INK,
        size=9,
        legend=False,
    )


def _axes_lines(fig: go.Figure, plane: Plane, *, row: int, col: int) -> None:
    variances, vectors = axes(plane.model)
    mean = plane.model.mean
    for index, label, dash in ((1, "smooth", "solid"), (0, "wiggly", "dot")):
        reach = 2.0 * float(np.sqrt(variances[index])) * vectors[:, index]
        fig.add_trace(
            go.Scatter(
                x=[mean[0] - reach[0], mean[0] + reach[0]],
                y=[mean[1] - reach[1], mean[1] + reach[1]],
                mode="lines",
                line={"width": 2.2, "color": C_INK, "dash": dash},
                name=f"{label} axis, ±2 SD",
                legendgroup=label,
                showlegend=True,
                hovertemplate=f"{label} axis: ±2 SD = ±{2 * np.sqrt(variances[index]):.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def fig_two_step_model(plane: Plane) -> go.Figure:
    """What the dynamics say, what the readings say, the posterior, and its two axes as paths."""
    toy, model = plane.toy, plane.model
    dynamics, readings = dynamics_model(toy), readings_model(toy)
    variances, vectors = axes(model)
    smooth_sd, wiggly_sd = float(np.sqrt(variances[1])), float(np.sqrt(variances[0]))
    fig = make_subplots(
        rows=1,
        cols=4,
        horizontal_spacing=0.07,
        subplot_titles=(
            "What the dynamics say",
            "What the two readings say",
            "Posterior: both at once",
            "The two axes, as paths",
        ),
    )
    spread = max(
        2.3 * float(np.sqrt(np.max(np.linalg.eigvalsh(dynamics.covariance)))) / np.sqrt(2.0), 0.8
    )
    centre = 0.5 * (dynamics.mean + readings.mean)
    ranges = ([centre[0] - spread, centre[0] + spread], [centre[1] - spread, centre[1] + spread])
    for col, source in ((1, dynamics), (2, readings)):
        _curves(
            fig,
            (_ellipse(source.mean, source.covariance, 2.0),),
            row=1,
            col=col,
            name="2 SD",
            legend=False,
            line={"width": 1.0, "color": C_MUTED},
            fill="rgba(154,160,166,0.16)",
        )
        _curves(
            fig,
            (_ellipse(source.mean, source.covariance, 1.0),),
            row=1,
            col=col,
            name="1 SD",
            legend=False,
            line={"width": 1.4, "color": C_MUTED},
        )
        _square(fig, row=1, col=col, ranges=ranges, index=col)
    _posterior(fig, plane, row=1, col=3, legend=False)
    _square(fig, row=1, col=3, ranges=ranges, index=3)
    _point(
        fig,
        toy.readings,
        row=1,
        col=2,
        name="readings (y₁, y₂)",
        symbol="diamond",
        color=C_INK_2,
        size=9,
        legend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plane.draws[:150, 0],
            y=plane.draws[:150, 1],
            mode="markers",
            marker={"size": 3, "color": C_INK_2, "opacity": 0.5},
            name="posterior draws",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )
    _axes_lines(fig, plane, row=1, col=3)
    steps = np.array([1, 2])
    for label, path, dash, width in (
        ("posterior mean", model.mean, "solid", 2.5),
        ("+1 SD along the smooth axis", model.mean + smooth_sd * vectors[:, 1], "solid", 1.5),
        ("−1 SD along the smooth axis", model.mean - smooth_sd * vectors[:, 1], "solid", 1.5),
        ("+1 SD along the wiggly axis", model.mean + wiggly_sd * vectors[:, 0], "dot", 1.5),
        ("−1 SD along the wiggly axis", model.mean - wiggly_sd * vectors[:, 0], "dot", 1.5),
    ):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=path,
                mode="lines+markers",
                line={"width": width, "color": C_INK if width > 2 else C_INK_2, "dash": dash},
                marker={"size": 7},
                name=label,
                showlegend=False,
                hovertemplate=label + "<br>step %{x}: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=4,
        )
    fig.add_annotation(
        x=2.05,
        y=float((model.mean + smooth_sd * vectors[:, 1])[1]),
        text="smooth ±1 SD:<br>both steps move",
        showarrow=False,
        xanchor="left",
        font={"size": 10},
        row=1,
        col=4,
    )
    fig.add_annotation(
        x=2.05,
        y=float(model.mean[1]),
        text="wiggly ±1 SD:<br>steps move apart",
        showarrow=False,
        xanchor="left",
        font={"size": 10},
        row=1,
        col=4,
    )
    fig.update_xaxes(tickvals=[1, 2], ticktext=["step 1", "step 2"], range=[0.8, 2.9], row=1, col=4)
    fig.update_yaxes(title={"text": "state", "standoff": 2}, row=1, col=4)
    return _layout(
        fig,
        title=f"Two steps of C0's latent: a thin diagonal ellipse, {smooth_sd / wiggly_sd:.1f} times longer than wide",
        height=360,
        legend=False,
    )


def fig_latents(planes: Sequence[Plane]) -> go.Figure:
    """Four types of latent: the posterior of two neighbouring steps for each."""
    titles = []
    for plane in planes:
        variances, _ = axes(plane.model)
        titles.append(
            f"{plane.label}<br><sup>long axis {np.sqrt(variances[1] / variances[0]):.1f} × the short one</sup>"
        )
    fig = make_subplots(rows=1, cols=len(planes), horizontal_spacing=0.06, subplot_titles=titles)
    for col, plane in enumerate(planes, start=1):
        _posterior(fig, plane, row=1, col=col, legend=col == 1)
        fig.add_trace(
            go.Scatter(
                x=plane.draws[:200, 0],
                y=plane.draws[:200, 1],
                mode="markers",
                marker={"size": 3, "color": C_INK_2, "opacity": 0.45},
                name="posterior draws",
                legendgroup="draws",
                showlegend=col == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        _axes_lines(fig, plane, row=1, col=col)
        _square(fig, row=1, col=col, ranges=plane_range(plane), index=col)
    fig.update_annotations(font={"size": 11})
    return _dedupe_legend(
        _layout(fig, title="Four types of latent, seen through two neighbouring steps", height=430)
    )


def fig_auxiliary(plane: Plane, deltas: Sequence[float], seed: int = 3) -> go.Figure:
    """Stage 1 jitters the current path into u; stage 2 draws near u. Three δ."""
    model, start = plane.model, plane.start
    variances, _ = axes(model)
    titles = tuple(
        f"δ = {delta:g}: at most {ceiling(delta, variances[1]):.2f} smooth, {ceiling(delta, variances[0]):.2f} wiggly"
        for delta in deltas
    )
    fig = make_subplots(rows=1, cols=len(deltas), horizontal_spacing=0.07, subplot_titles=titles)
    ranges = plane_range(plane)
    noise = np.random.default_rng(seed).normal(size=2)
    for col, delta in enumerate(deltas, start=1):
        auxiliary = start + np.sqrt(0.5 * delta) * noise
        _posterior(fig, plane, row=1, col=col, legend=col == 1)
        conditional = auxiliary_model(model, auxiliary, delta)
        _curves(
            fig,
            (
                _ellipse(conditional.mean, conditional.covariance, 1.0),
                _ellipse(conditional.mean, conditional.covariance, 2.0),
            ),
            row=1,
            col=col,
            name="where stage 2 draws (1 and 2 SD)",
            legend=col == 1,
            line={"width": 1.4, "color": C_INK, "dash": "dash"},
        )
        _point(
            fig,
            start,
            row=1,
            col=col,
            name="current path",
            symbol="circle",
            color=C_INK,
            size=10,
            legend=col == 1,
        )
        _auxiliary_point(fig, auxiliary, ranges, row=1, col=col, index=col, legend=col == 1)
        _square(fig, row=1, col=col, ranges=ranges, index=col)
    fig.update_annotations(font={"size": 11})
    return _layout(fig, title="Jitter, then redraw: δ sets how far any update can go", height=420)


def fig_plane_panels(
    plane: Plane, panels: Sequence[tuple[str, Study, bool]], *, title: str
) -> go.Figure:
    """A row of plane panels: (title, study, chain?) shows one update or its first 60 updates."""
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        horizontal_spacing=0.06,
        subplot_titles=[panel[0] for panel in panels],
    )
    ranges = plane_range(plane)
    for col, (_, result, as_chain) in enumerate(panels, start=1):
        color = COLORS[result.kind]
        if as_chain:
            _plane_chain(fig, plane, result.chain, row=1, col=col, color=color)
        else:
            _plane_update(
                fig, plane, result.picture, row=1, col=col, index=col, color=color, legend=True
            )
        _square(fig, row=1, col=col, ranges=ranges, index=col)
    fig.update_annotations(font={"size": 11})
    return _dedupe_legend(_layout(fig, title=title, height=430))


def fig_proposal(kind: str, planes: Sequence[Plane], studies: Sequence[Study]) -> go.Figure:
    """One update on every type of latent (top) and its first 60 updates (bottom)."""
    color = COLORS[kind]
    top, bottom = [], []
    for plane, result in zip(planes, studies, strict=True):
        smooth, wiggly = axis_refresh(result.chain, plane.model)
        detail = f" · δ = {result.delta:g}" if np.isfinite(result.delta) else ""
        if result.acceptance is not None:
            detail = f" · {result.acceptance:.0%} accepted"
        top.append(f"{plane.label}{detail}")
        bottom.append(f"refresh: smooth {smooth:.2f} · wiggly {wiggly:.2f}")
    fig = make_subplots(
        rows=2,
        cols=len(planes),
        horizontal_spacing=0.045,
        vertical_spacing=0.1,
        subplot_titles=top + bottom,
    )
    for col, (plane, result) in enumerate(zip(planes, studies, strict=True), start=1):
        ranges = plane_range(plane)
        _plane_update(
            fig, plane, result.picture, row=1, col=col, index=col, color=color, legend=True
        )
        _square(fig, row=1, col=col, ranges=ranges, index=col, titles=False)
        _plane_chain(fig, plane, result.chain, row=2, col=col, color=color)
        _square(fig, row=2, col=col, ranges=ranges, index=len(planes) + col, titles=False)
    fig.update_annotations(font={"size": 10})
    fig.update_xaxes(tickfont={"size": 9})
    fig.update_yaxes(tickfont={"size": 9})
    family = (
        "separable" if kind in SEPARABLE else "sequential" if kind in SEQUENTIAL else "whole path"
    )
    figure = _layout(
        fig,
        title=f"{NAMES[kind]} · {family}: one update (top) and 60 updates (bottom)",
        height=560,
        legend=False,
    )
    figure.update_layout(margin={"l": 40, "r": 20, "t": 70, "b": 30})
    return figure


# ── The long path ──


def fig_posterior_shapes(toy: Toy, model: GaussianModel, draws: np.ndarray) -> go.Figure:
    """The posterior, each draw split into smooth and wiggly parts, and the shape spectrum."""
    times = toy.times
    split = smooth_period(toy)
    smooth = smooth_projector(toy.steps, split)
    wiggly = np.eye(toy.steps) - smooth
    covariance = model.covariance
    smooth_sd = np.sqrt(np.diag(smooth @ covariance @ smooth))
    wiggly_sd = np.sqrt(np.diag(wiggly @ covariance @ wiggly))
    deviations = draws - model.mean
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Readings, posterior mean ±2 SD, and three whole-path posterior draws",
            f"Smooth part of each draw (periods over {split:.0f} steps)",
            f"Wiggly part of each draw (periods under {split:.0f} steps)",
            "Posterior SD of each cosine shape",
        ),
    )
    sd = np.sqrt(np.diag(covariance))
    fig.add_trace(_band(times, model.mean, 2 * sd, "posterior ±2 SD", legend=True), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=times[toy.observed],
            y=toy.readings[toy.observed],
            mode="markers",
            marker={"size": 4, "color": C_MUTED},
            name="readings",
            hovertemplate="day %{x:.2f}<br>reading %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for index, draw in enumerate(draws):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=draw,
                mode="lines",
                line={"width": 1.2, "color": C_INK_2},
                opacity=0.7,
                name="posterior draws",
                legendgroup="draws",
                showlegend=index == 0,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=model.mean,
            mode="lines",
            line={"width": 2, "color": C_INK},
            name="posterior mean",
            hovertemplate="day %{x:.2f}<br>mean %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for col, projector, spread in ((1, smooth, smooth_sd), (2, wiggly, wiggly_sd)):
        fig.add_trace(
            _band(times, np.zeros_like(times), 2 * spread, "posterior ±2 SD", legend=False),
            row=2,
            col=col,
        )
        for deviation in deviations:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=projector @ deviation,
                    mode="lines",
                    line={"width": 1.5, "color": C_INK_2},
                    opacity=0.8,
                    name="posterior draws",
                    legendgroup="draws",
                    showlegend=False,
                    hovertemplate="day %{x:.2f}<br>%{y:.3f}<extra></extra>",
                ),
                row=2,
                col=col,
            )
    period = periods(toy.steps)[1:]
    spread = mode_sd(model)[1:]
    fig.add_trace(
        go.Scatter(
            x=period,
            y=spread,
            mode="lines",
            line={"width": 2, "color": C_INK},
            name="posterior SD of the shape",
            showlegend=False,
            hovertemplate="period %{x:.1f} steps<br>SD %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[split, split],
            y=[spread.min(), spread.max()],
            mode="lines",
            line={"width": 1, "color": C_MUTED},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=3,
    )
    fig.update_yaxes(title="state", row=1, col=1)
    parts = np.concatenate([deviations @ smooth, deviations @ wiggly])
    smooth_limit = max(2.6 * max(smooth_sd.max(), wiggly_sd.max()), 1.1 * np.abs(parts).max())
    for col in (1, 2):
        fig.update_xaxes(title="time (days)", row=2, col=col)
        fig.update_yaxes(range=[-smooth_limit, smooth_limit], row=2, col=col)
    fig.update_yaxes(title="minus the mean", row=2, col=1)
    fig.update_xaxes(
        type="log",
        autorange="reversed",
        tickvals=_period_ticks(toy.steps)[::2],
        title="period in grid steps (smooth → wiggly)",
        row=2,
        col=3,
    )
    fig.update_yaxes(type="log", tickvals=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], row=2, col=3)
    ratio = (spread[0] / spread[-1]) ** 2
    return _layout(
        fig,
        title=f"{toy.steps} steps: the posterior and its two kinds of shape (smoothest/wiggliest variance ≈ {ratio:,.0f} : 1)",
        height=560,
    )


def local_range(toy: Toy, start: np.ndarray, pictures: Sequence[Picture]) -> tuple[float, float]:
    """One y-range for every local path panel, so proposal spreads compare directly."""
    centre = toy.steps // 2
    window = slice(centre - 15, centre + 15)
    values = [start[window]]
    for picture in pictures:
        cloud = picture.proposals.T if isinstance(picture, WholePaths) else picture.values
        values.append(np.ravel(cloud[window]))
    stacked = np.concatenate(values)
    low, high = float(np.percentile(stacked, 0.5)), float(np.percentile(stacked, 99.5))
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def smooth_scale(toy: Toy, model: GaussianModel) -> float:
    """Colour limit for the smooth-deviation strips: 2.5 posterior SDs of the smooth part."""
    smooth = smooth_projector(toy.steps, smooth_period(toy))
    return float(2.5 * np.sqrt(np.diag(smooth @ model.covariance @ smooth)).max())


def _path_update(
    fig: go.Figure,
    toy: Toy,
    model: GaussianModel,
    start: np.ndarray,
    picture: Picture,
    *,
    color: str,
    legend: bool,
) -> None:
    """One update in a 30-step stretch: candidate clouds, particle lineages, or whole paths."""
    centre = toy.steps // 2
    window = slice(centre - 15, centre + 15)
    times = toy.times[window]
    sd = np.sqrt(np.diag(model.covariance))
    fig.add_trace(
        _band(times, model.mean[window], 2 * sd[window], "posterior ±2 SD", legend=legend),
        row=1,
        col=1,
    )
    if isinstance(picture, WholePaths):
        for index, proposal in enumerate(picture.proposals):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=proposal[window],
                    mode="lines",
                    line={"width": 1, "color": color},
                    opacity=0.45,
                    name="whole-path proposals",
                    legendgroup="whole",
                    showlegend=legend and index == 0,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
    else:
        cloud = picture.values[window]
        if picture.ancestors is not None:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for offset in range(1, cloud.shape[0]):
                step = centre - 15 + offset
                parents = picture.values[step - 1][picture.ancestors[step - 1]]
                for particle in range(cloud.shape[1]):
                    xs += [float(times[offset - 1]), float(times[offset]), None]
                    ys += [float(parents[particle]), float(cloud[offset, particle]), None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"width": 0.6, "color": C_MUTED},
                    opacity=0.5,
                    name="each candidate's parent",
                    legendgroup="parents",
                    showlegend=legend,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=np.repeat(times[:, None], cloud.shape[1], axis=1).ravel(),
                y=cloud.ravel(),
                mode="markers",
                marker={"size": 3.5, "color": C_MUTED, "opacity": 0.6},
                name="candidates",
                legendgroup="cands",
                showlegend=legend,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    for label, path, line in (
        ("current path", start, {"width": 2, "color": C_INK}),
        ("new path", picture.new_path, {"width": 2.5, "color": color}),
    ):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=path[window],
                mode="lines",
                line=line,
                name=label,
                legendgroup=label,
                showlegend=legend,
                hovertemplate="day %{x:.3f}<br>%{y:.3f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )


def fig_path_row(
    toy: Toy,
    model: GaussianModel,
    start: np.ndarray,
    result: Study,
    *,
    local: tuple[float, float],
    zmax: float,
    first: bool,
) -> go.Figure:
    """One update on the long path: a 30-step stretch, 40 sweeps of the smooth part, refresh by shape."""
    color = COLORS[result.kind]
    titles = (
        (
            "One update, in a 30-step stretch",
            "Smooth part of the path over 40 sweeps",
            "Refreshed per sweep, by shape",
        )
        if first
        else ("", "", "")
    )
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.34, 0.36, 0.30],
        horizontal_spacing=0.07,
        subplot_titles=titles,
    )
    _path_update(fig, toy, model, start, result.picture, color=color, legend=first)
    smooth = smooth_projector(toy.steps, smooth_period(toy))
    strip = (result.chain[:40] - model.mean) @ smooth
    fig.add_trace(
        go.Heatmap(
            x=toy.times,
            y=np.arange(strip.shape[0]),
            z=strip,
            zmin=-zmax,
            zmax=zmax,
            colorscale=DIVERGING,
            showscale=False,
            hovertemplate="day %{x:.2f}<br>sweep %{y}<br>%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    values = refresh(result.chain, model)
    centers, means = binned(values, toy.steps)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=means,
            mode="lines+markers",
            line={"width": 2, "color": color, "dash": DASH[result.kind]},
            marker={
                "size": 7,
                "color": color,
                "symbol": SYMBOL[result.kind],
                "line": {"width": 1.5, "color": C_SURFACE},
            },
            name="fraction refreshed",
            showlegend=False,
            hovertemplate="period ≈ %{x:.0f} steps<br>%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(range=list(local), title={"text": "state", "standoff": 2}, row=1, col=1)
    fig.update_xaxes(title={"text": "time (days)", "standoff": 4}, row=1, col=1)
    fig.update_yaxes(title={"text": "sweep", "standoff": 2}, autorange="reversed", row=1, col=2)
    fig.update_xaxes(title={"text": "time (days)", "standoff": 4}, row=1, col=2)
    fig.update_xaxes(
        type="log",
        autorange="reversed",
        tickvals=_period_ticks(toy.steps)[::2],
        title={"text": "period in steps (smooth → wiggly)", "standoff": 4},
        row=1,
        col=3,
    )
    fig.update_yaxes(range=[0, 1.15], row=1, col=3)
    smooth_value, wiggly_value = shape_summary(values, toy)
    family = (
        "separable"
        if result.kind in SEPARABLE
        else "sequential"
        if result.kind in SEQUENTIAL
        else "whole path"
    )
    delta = "" if not np.isfinite(result.delta) else f" · δ = {result.delta:g}"
    title = (
        f"{NAMES[result.kind]} · {family}{delta}: refreshes {smooth_value:.2f} of the smooth shapes "
        f"and {wiggly_value:.2f} of the wiggles per sweep"
    )
    if result.acceptance is not None:
        title += f" · accepted {result.acceptance:.0%}"
    fig.update_annotations(font={"size": 11})
    return _layout(fig, title=title, height=330 if first else 290, legend=first)


def _line(
    kind: str, x: Sequence[float], y: Sequence[float], *, legend: bool, hover: str
) -> go.Scatter:
    color = COLORS[kind]
    return go.Scatter(
        x=list(x),
        y=list(y),
        mode="lines+markers",
        line={"width": 2, "color": color, "dash": DASH[kind]},
        marker={
            "size": 8,
            "color": color,
            "symbol": SYMBOL[kind],
            "line": {"width": 1.5, "color": C_SURFACE},
        },
        name=SHORT[kind],
        legendgroup=kind,
        showlegend=legend,
        hovertemplate=hover + "<extra>" + SHORT[kind] + "</extra>",
    )


def fig_limits(
    scan: Mapping[str, Sequence[float]],
    ceilings: Sequence[float],
    kalman: float,
    lengths: Mapping[str, Sequence[float]],
) -> go.Figure:
    """Smooth refresh against δ (separable, sequential) and against the number of steps."""
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=(
            "δ, separable updates",
            "δ, sequential updates",
            "Steps in the path, each at its best δ",
        ),
    )
    for col, kinds in ((1, SEPARABLE), (2, SEQUENTIAL)):
        fig.add_trace(
            go.Scatter(
                x=list(SCAN_DELTAS),
                y=list(ceilings),
                mode="lines",
                line={"width": 1.5, "color": C_MUTED, "dash": "dot"},
                name="ceiling of jitter-then-redraw",
                legendgroup="ceiling",
                showlegend=col == 1,
                hovertemplate="δ = %{x:g}<br>ceiling %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[SCAN_DELTAS[0], SCAN_DELTAS[-1]],
                y=[kalman, kalman],
                mode="lines",
                line={"width": 2, "color": C_KALMAN},
                name=SHORT["kalman"],
                legendgroup="kalman",
                showlegend=col == 1,
                hovertemplate="%{y:.2f}<extra>Kalman move, δ = ∞</extra>",
            ),
            row=1,
            col=col,
        )
        for kind in kinds:
            fig.add_trace(
                _line(
                    kind,
                    SCAN_DELTAS,
                    scan[kind],
                    legend=True,
                    hover="δ = %{x:g}<br>smooth refresh %{y:.2f}",
                ),
                row=1,
                col=col,
            )
    for kind in LENGTH_DELTAS:
        fig.add_trace(
            _line(kind, LENGTHS, lengths[kind], legend=False, hover="%{x} steps<br>%{y:.2f}"),
            row=1,
            col=3,
        )
    for col in (1, 2):
        fig.update_xaxes(
            type="log",
            tickvals=list(SCAN_DELTAS)[::2],
            ticktext=[f"{v:g}" for v in SCAN_DELTAS[::2]],
            title="δ",
            row=1,
            col=col,
        )
    fig.update_xaxes(
        type="log",
        tickvals=list(LENGTHS),
        ticktext=[str(value) for value in LENGTHS],
        title="steps in the path (C0's grid)",
        row=1,
        col=3,
    )
    fig.update_yaxes(title="smooth-shape refresh per sweep", range=[0, 1.12], row=1, col=1)
    fig.update_annotations(font={"size": 11})
    return _dedupe_legend(_layout(fig, title="What limits each family", height=440))


# ── The 2026-10-23 C0 runs (scratchpad/latent-proposals-20261023/comparison.json) ──

C0_NAMES = {
    "local": "Today's sampler (A0)",
    "marginal": "A0 with smoother marginals",
    "kalman": "A0 + Kalman path move",
}
C0_BANDS = ("> 48 h", "12–48 h", "6–12 h", "3–6 h", "1.5–3 h", "45–90 min", "< 45 min")
C0_REFRESH = {
    "local": {
        "x": (0.078, 0.085, 0.106, 0.16, 0.247, 0.312, 0.345),
        "y": (0.045, 0.05, 0.074, 0.131, 0.223, 0.303, 0.346),
    },
    "marginal": {
        "x": (0.303, 0.305, 0.324, 0.352, 0.37, 0.379, 0.381),
        "y": (0.294, 0.303, 0.339, 0.375, 0.398, 0.407, 0.41),
    },
    "kalman": {
        "x": (0.929, 0.932, 0.935, 0.936, 0.943, 0.948, 0.951),
        "y": (0.929, 0.926, 0.931, 0.934, 0.942, 0.948, 0.951),
    },
}


def fig_c0_bands() -> go.Figure:
    """Per-band refresh in the real C0 fits, one panel per path."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("path x", "path y"),
    )
    for col, path in enumerate(("x", "y"), start=1):
        for key in ("local", "marginal", "kalman"):
            fig.add_trace(
                go.Scatter(
                    x=list(C0_BANDS),
                    y=list(C0_REFRESH[key][path]),
                    mode="lines+markers",
                    line={"width": 2, "color": COLORS[key]},
                    marker={
                        "size": 8,
                        "color": COLORS[key],
                        "symbol": SYMBOL[key],
                        "line": {"width": 1.5, "color": C_SURFACE},
                    },
                    name=C0_NAMES[key],
                    legendgroup=key,
                    showlegend=col == 1,
                    hovertemplate="%{x}<br>refreshed %{y:.2f}<extra>" + C0_NAMES[key] + "</extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title="period of the path component (smooth → wiggly)")
    fig.update_yaxes(title="fraction refreshed per sweep", range=[0, 1.05], row=1, col=1)
    return _layout(fig, title="C0: 2,049 steps per path, 16,000 retained draws per arm", height=440)
