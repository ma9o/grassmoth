"""Support code for the latent-path proposals walkthrough notebook.

A one-dimensional stochastic process on a fine time grid, observed with noise, and
three exact MCMC updates of its whole path:

1. **Separable local proposals** (today's dSMC proposal). Each time step gets N
   candidates drawn around an auxiliary point ``u_t ~ N(x_t, δ/2)``, independently
   of every other step; the candidates are then stitched into one path.
2. **Separable smoother-marginal proposals** (dSMC paper §4.1). Candidates come from
   a Gaussian smoother's per-time marginals, still independently per step.
3. **The auxiliary Kalman path move.** One whole path is drawn from a Gaussian model
   of the dynamics and readings conditioned on ``u ~ N(x, δ/2)``, then accepted or
   rejected against the exact density.

Stitching here is forward filtering and backward sampling over the candidate grid:
the exact distribution over every combination of candidates. No stitching can do
better, dSMC's tree included, so slowness here cannot be blamed on the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.linalg
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from plotly.subplots import make_subplots

# ── Palette: validated categorical slots 1-3 (blue, orange, aqua) ────────────
C_LOCAL = "#2a78d6"
C_MARGINAL = "#eb6834"
C_KALMAN = "#1baf7a"
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#9aa0a6"
C_GRID = "#e6e5e1"
C_SURFACE = "#fcfcfb"
NAMES = {
    "local": "Separable local (today's dSMC)",
    "marginal": "Separable smoother marginals",
    "kalman": "Kalman path move",
}
COLORS = {"local": C_LOCAL, "marginal": C_MARGINAL, "kalman": C_KALMAN}

# ── C0 ground truth for the toy: decay, input effect and noise as in C0's x ──
DECAY = 0.6
FORCING = 0.5
DIFFUSION = 0.45
NOISE = 0.2
INITIAL_SD = 0.4
DAYS = 2.0


@dataclass(frozen=True)
class Toy:
    """One simulated history on a grid, with its readings."""

    dt: float
    cubic: float
    forcing: np.ndarray
    observed: np.ndarray
    readings: np.ndarray
    truth: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.truth.size)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.steps) * self.dt


def make_toy(*, intervals_per_day: int, every: int = 1, cubic: float = 0.0, seed: int = 7) -> Toy:
    """Simulate dx = (-a x - c x^3 + b u) dt + s dW on the grid; read every `every` steps."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / intervals_per_day
    steps = int(DAYS * intervals_per_day)
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
    readings = np.where(observed, truth + NOISE * rng.normal(size=steps), np.nan)
    return Toy(dt, cubic, forcing, observed, readings, truth)


def _transition_mean(toy_dt, cubic, forcing, previous, effect=FORCING):
    drift = -DECAY * previous - cubic * previous**3 + effect * forcing
    return previous + drift * toy_dt


def log_density(toy: Toy, path: np.ndarray) -> float:
    """Exact Euler–Maruyama path density with the readings."""
    variance = DIFFUSION**2 * toy.dt
    mean = _transition_mean(toy.dt, toy.cubic, toy.forcing[:-1], path[:-1])
    readings = toy.readings[toy.observed]
    return float(
        -0.5 * path[0] ** 2 / INITIAL_SD**2
        - 0.5 * np.sum((path[1:] - mean) ** 2) / variance
        - 0.5 * np.sum((path[toy.observed] - readings) ** 2) / NOISE**2
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


def gaussian_model(toy: Toy, around: np.ndarray, effect: float = FORCING) -> GaussianModel:
    """Expand the transition means around a path; readings are Gaussian already.

    ``effect`` is the input effect the model assumes, to show a model frozen at a
    parameter value the chain has since moved away from.
    """
    steps, variance = toy.steps, DIFFUSION**2 * toy.dt
    slope = 1.0 + (-DECAY - 3.0 * toy.cubic * around[:-1] ** 2) * toy.dt
    offset = (
        _transition_mean(toy.dt, toy.cubic, toy.forcing[:-1], around[:-1], effect)
        - slope * around[:-1]
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


def reference_model(toy: Toy, effect: float = FORCING) -> GaussianModel:
    """Gaussian model expanded around its own linear-dynamics mean (exact when cubic is 0)."""
    linear = gaussian_model(
        Toy(toy.dt, 0.0, toy.forcing, toy.observed, toy.readings, toy.truth),
        np.zeros(toy.steps),
        effect,
    )
    return gaussian_model(toy, linear.mean, effect)


def posterior_draws(model: GaussianModel, count: int, seed: int) -> np.ndarray:
    """Exact draws from the Gaussian model."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(model.precision)
    noise = rng.normal(size=(model.shift.size, count))
    return (model.mean[:, None] + scipy.linalg.solve_triangular(chol.T, noise, lower=False)).T


# ── Separable proposals with exact stitching (forward filtering, backward sampling) ──


@partial(jax.jit, static_argnames=("particles", "smoother"))
def _stitch_sweep(
    key, reference, arrays, delta, marginal_mean, marginal_sd, *, particles, smoother
):
    dt, cubic, forcing, observed, readings = arrays
    steps = reference.shape[0]
    variance = DIFFUSION**2 * dt
    draw_key, auxiliary_key, back_key = jax.random.split(key, 3)
    noise = jax.random.normal(draw_key, (steps, particles - 1))
    if smoother:
        candidates = marginal_mean[:, None] + marginal_sd[:, None] * noise
    else:
        auxiliary = reference + jnp.sqrt(0.5 * delta) * jax.random.normal(auxiliary_key, (steps,))
        candidates = auxiliary[:, None] + jnp.sqrt(0.5 * delta) * noise
    pool = jnp.concatenate((reference[:, None], candidates), axis=1)
    emission = jnp.where(
        observed[:, None], norm.logpdf(jnp.nan_to_num(readings)[:, None], pool, NOISE), 0.0
    )
    if smoother:
        # Independent proposals pay their density; the auxiliary proposal's density
        # cancels against its auxiliary factor, so it pays nothing.
        emission = emission - norm.logpdf(pool, marginal_mean[:, None], marginal_sd[:, None])
    first = emission[0] + norm.logpdf(pool[0], 0.0, INITIAL_SD)

    def link(t):
        mean = _transition_mean(dt, cubic, forcing[t - 1], pool[t - 1])
        return -0.5 * (pool[t][:, None] - mean[None, :]) ** 2 / variance

    def forward(previous, t):
        current = emission[t] + logsumexp(previous[None, :] + link(t), axis=1)
        return current, current

    _, rest = jax.lax.scan(forward, first, jnp.arange(1, steps))
    filtered = jnp.concatenate((first[None], rest))
    last = jax.random.categorical(back_key, filtered[-1])

    def backward(chosen, inputs):
        t, step_key = inputs
        index = jax.random.categorical(step_key, filtered[t] + link(t + 1)[chosen])
        return index, index

    keys = jax.random.split(jax.random.fold_in(back_key, 1), steps - 1)
    _, earlier = jax.lax.scan(backward, last, (jnp.arange(steps - 2, -1, -1), keys))
    indices = jnp.concatenate((earlier[::-1], last[None]))
    return pool[jnp.arange(steps), indices], pool


def _arrays(toy: Toy):
    return (
        jnp.asarray(toy.dt),
        jnp.asarray(toy.cubic),
        jnp.asarray(toy.forcing),
        jnp.asarray(toy.observed),
        jnp.asarray(toy.readings),
    )


def run_separable(
    toy: Toy,
    start: np.ndarray,
    *,
    sweeps: int,
    particles: int,
    delta: float = 0.05,
    smoother: GaussianModel | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Chain of separable updates; returns the path after each sweep."""
    arrays = _arrays(toy)
    mean = jnp.zeros(toy.steps) if smoother is None else jnp.asarray(smoother.mean)
    sd = (
        jnp.ones(toy.steps)
        if smoother is None
        else jnp.asarray(np.sqrt(np.diag(smoother.covariance)))
    )

    def body(path, key):
        new, _ = _stitch_sweep(
            key, path, arrays, delta, mean, sd, particles=particles, smoother=smoother is not None
        )
        return new, new

    keys = jax.random.split(jax.random.key(seed), sweeps)
    _, paths = jax.lax.scan(body, jnp.asarray(start), keys)
    return np.asarray(paths)


def one_stitch(
    toy: Toy,
    start: np.ndarray,
    *,
    particles: int,
    delta: float = 0.05,
    smoother: GaussianModel | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One separable update: the stitched path and the candidate pool (column 0 is the current path)."""
    mean = jnp.zeros(toy.steps) if smoother is None else jnp.asarray(smoother.mean)
    sd = (
        jnp.ones(toy.steps)
        if smoother is None
        else jnp.asarray(np.sqrt(np.diag(smoother.covariance)))
    )
    path, pool = _stitch_sweep(
        jax.random.key(seed),
        jnp.asarray(start),
        _arrays(toy),
        delta,
        mean,
        sd,
        particles=particles,
        smoother=smoother is not None,
    )
    return np.asarray(path), np.asarray(pool)


# ── The auxiliary Kalman path move ───────────────────────────────────────────


def run_kalman(
    toy: Toy,
    start: np.ndarray,
    model: GaussianModel,
    *,
    sweeps: int,
    delta: float = np.inf,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Chain of auxiliary Kalman moves; returns paths and the acceptance rate.

    The model is frozen, so the auxiliary and proposal densities cancel and the
    acceptance ratio compares the exact density with the model's own density.
    """
    rng = np.random.default_rng(seed)
    steps = toy.steps
    auxiliary_precision = 0.0 if np.isinf(delta) else 2.0 / delta
    precision = model.precision + auxiliary_precision * np.eye(steps)
    factor = scipy.linalg.cho_factor(precision, lower=True)
    path = np.asarray(start, dtype=float)
    current = log_density(toy, path) - model.log_density(path)
    paths = np.empty((sweeps, steps))
    accepted = 0
    for sweep in range(sweeps):
        shift = model.shift
        if auxiliary_precision > 0.0:
            auxiliary = path + rng.normal(size=steps) / np.sqrt(auxiliary_precision)
            shift = shift + auxiliary_precision * auxiliary
        mean = scipy.linalg.cho_solve(factor, shift)
        proposal = mean + scipy.linalg.solve_triangular(
            factor[0].T, rng.normal(size=steps), lower=False
        )
        candidate = log_density(toy, proposal) - model.log_density(proposal)
        if np.log(rng.uniform()) < candidate - current:
            path, current = proposal, candidate
            accepted += 1
        paths[sweep] = path
    return paths, accepted / sweeps


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
    """Average over modes whose period lies in [low, high) grid steps."""
    period = periods(steps)
    return float(np.mean(values[(period >= low) & (period < high)]))


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


# ── Shapes and update views ─────────────────────────────────────────────────

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


@dataclass(frozen=True)
class UpdateView:
    """One update seen locally (its proposals), globally (its chain) and by shape."""

    key: str
    cloud: np.ndarray
    connected: bool
    new_path: np.ndarray
    chain: np.ndarray
    acceptance: float | None


def update_views(
    toy: Toy,
    model: GaussianModel,
    start: np.ndarray,
    *,
    particles: int,
    delta: float,
    sweeps: int = 450,
    burn: int = 50,
    seed: int = 0,
) -> dict[str, UpdateView]:
    """One update from ``start`` for the local picture, and a chain for the rest."""
    local_new, local_pool = one_stitch(toy, start, particles=particles, delta=delta, seed=seed)
    smooth_new, smooth_pool = one_stitch(toy, start, particles=particles, smoother=model, seed=seed)
    # At infinite δ the Kalman proposal is a draw from the Gaussian model, whatever the current path.
    proposals = posterior_draws(model, 12, seed=seed + 1)
    kalman_chain, acceptance = run_kalman(toy, start, model, sweeps=sweeps, seed=seed)
    local_chain = run_separable(
        toy, start, sweeps=sweeps, particles=particles, delta=delta, seed=seed
    )
    smooth_chain = run_separable(
        toy, start, sweeps=sweeps, particles=particles, smoother=model, seed=seed
    )
    return {
        "local": UpdateView("local", local_pool[:, 1:], False, local_new, local_chain[burn:], None),
        "marginal": UpdateView(
            "marginal", smooth_pool[:, 1:], False, smooth_new, smooth_chain[burn:], None
        ),
        "kalman": UpdateView(
            "kalman", proposals.T, True, proposals[0], kalman_chain[burn:], acceptance
        ),
    }


def local_range(toy: Toy, start: np.ndarray, views: dict[str, UpdateView]) -> tuple[float, float]:
    """One y-range for every local panel, so proposal spreads compare directly."""
    centre = toy.steps // 2
    window = slice(centre - 15, centre + 15)
    values = np.concatenate(
        [start[window], *(np.ravel(view.cloud[window]) for view in views.values())]
    )
    low, high = float(values.min()), float(values.max())
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def smooth_scale(toy: Toy, model: GaussianModel) -> float:
    """Colour limit for the smooth-deviation strips: 2.5 posterior SDs of the smooth part."""
    smooth = smooth_projector(toy.steps, smooth_period(toy))
    return float(2.5 * np.sqrt(np.diag(smooth @ model.covariance @ smooth)).max())


# ── Figures ────────────────────────────────────────────────────────────────

DIVERGING = [
    [0.0, "#184f95"],
    [0.25, "#6da7ec"],
    [0.5, "#f0efec"],
    [0.75, "#ee8f8a"],
    [1.0, "#a5262a"],
]
C_DRAW = "#6250d6"


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
                line={"width": 1.2, "color": C_DRAW},
                opacity=0.75,
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
                    line={"width": 1.5, "color": C_DRAW},
                    opacity=0.85,
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
    fig.update_xaxes(title="time (days)", row=1, col=1)
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
        title=f"The posterior and its two kinds of shape (smoothest/wiggliest variance ratio ≈ {ratio:,.0f})",
        height=600,
    )


def fig_update_row(
    toy: Toy,
    model: GaussianModel,
    start: np.ndarray,
    view: UpdateView,
    *,
    local_range: tuple[float, float],
    zmax: float,
    first: bool,
) -> go.Figure:
    """One update: its proposals locally, its chain's smooth part globally, and refresh by shape."""
    color = COLORS[view.key]
    times = toy.times
    centre = toy.steps // 2
    window = slice(centre - 15, centre + 15)
    local_times = times[window]
    titles = (
        (
            "Locally: proposals in a 30-step stretch",
            "Globally: smooth part of the path, 40 sweeps",
            "By shape: fraction refreshed per sweep",
        )
        if first
        else ("", "", "")
    )
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.32, 0.38, 0.30],
        horizontal_spacing=0.09,
        subplot_titles=titles,
    )
    sd = np.sqrt(np.diag(model.covariance))
    fig.add_trace(
        _band(local_times, model.mean[window], 2 * sd[window], "posterior ±2 SD", legend=True),
        row=1,
        col=1,
    )
    if view.connected:
        for index, proposal in enumerate(view.cloud.T):
            fig.add_trace(
                go.Scatter(
                    x=local_times,
                    y=proposal[window],
                    mode="lines",
                    line={"width": 1, "color": color},
                    opacity=0.45,
                    name="proposed whole paths",
                    legendgroup="cloud",
                    showlegend=index == 0,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
    else:
        cloud = view.cloud[window]
        fig.add_trace(
            go.Scatter(
                x=np.repeat(local_times[:, None], cloud.shape[1], axis=1).ravel(),
                y=cloud.ravel(),
                mode="markers",
                marker={"size": 4, "color": C_MUTED, "opacity": 0.6},
                name="candidates, drawn per step",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    for label, path, line in (
        ("current path", start, {"width": 2, "color": C_INK}),
        ("new path", view.new_path, {"width": 2.5, "color": color}),
    ):
        fig.add_trace(
            go.Scatter(
                x=local_times,
                y=path[window],
                mode="lines",
                line=line,
                name=label,
                hovertemplate="day %{x:.3f}<br>%{y:.3f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )
    smooth = smooth_projector(toy.steps, smooth_period(toy))
    strip = (view.chain[:40] - model.mean) @ smooth
    fig.add_trace(
        go.Heatmap(
            x=times,
            y=np.arange(strip.shape[0]),
            z=strip,
            zmin=-zmax,
            zmax=zmax,
            colorscale=DIVERGING,
            showscale=first,
            colorbar={
                "title": {"text": "smooth<br>deviation", "font": {"size": 10}},
                "len": 0.9,
                "thickness": 10,
                "x": 0.672,
                "xanchor": "left",
                "tickfont": {"size": 9},
            },
            hovertemplate="day %{x:.2f}<br>sweep %{y}<br>%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    values = refresh(view.chain, model)
    centers, means = binned(values, toy.steps)
    smooth_value, wiggly_value = shape_summary(values, toy)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=means,
            mode="lines+markers",
            line={"width": 2, "color": color},
            marker={"size": 7, "color": color, "line": {"width": 2, "color": C_SURFACE}},
            name="fraction refreshed",
            showlegend=False,
            hovertemplate="period ≈ %{x:.0f} steps<br>%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(range=list(local_range), title="state", row=1, col=1)
    fig.update_xaxes(title="time (days)", row=1, col=1)
    fig.update_yaxes(title="sweep", autorange="reversed", row=1, col=2)
    fig.update_xaxes(title="time (days)", row=1, col=2)
    fig.update_xaxes(
        type="log",
        autorange="reversed",
        tickvals=_period_ticks(toy.steps)[::2],
        title="period in grid steps (smooth → wiggly)",
        row=1,
        col=3,
    )
    fig.update_yaxes(range=[0, 1.15], row=1, col=3)
    title = (
        f"{NAMES[view.key]}: refreshes {smooth_value:.2f} of smooth shapes and "
        f"{wiggly_value:.2f} of wiggles per sweep"
    )
    if view.acceptance is not None:
        title += f" (acceptance {view.acceptance:.2f})"
    return _layout(fig, title=title, height=340 if first else 310)


class GridScan(TypedDict):
    """Refresh measurements at each grid density, grouped by reading density."""

    intervals_per_day: tuple[int, ...]
    dense: dict[str, tuple[float, ...]]
    sparse: dict[str, tuple[float, ...]]


# Smooth-shape refresh (periods over half a day) on this toy, seed 7, 32 candidates,
# best of δ in {0.01, 0.03, 0.1, 0.3, 1} for the local proposal, 500 sweeps (2026-10-23).
GRID_SCAN: GridScan = {
    "intervals_per_day": (16, 32, 64, 128, 256),
    "dense": {
        "local": (0.675, 0.551, 0.389, 0.279, 0.169),
        "marginal": (0.956, 0.856, 0.767, 0.645, 0.593),
        "kalman": (1.006, 0.981, 1.007, 1.028, 0.974),
    },
    "sparse": {
        "local": (0.562, 0.367, 0.182, 0.071, 0.028),
        "marginal": (0.861, 0.772, 0.592, 0.421, 0.227),
        "kalman": (0.999, 0.970, 1.017, 1.019, 0.978),
    },
}


def scan_grid(
    every: int, *, sweeps: int = 500, particles: int = 32
) -> dict[str, tuple[float, ...]]:
    """Recompute one reading density of ``GRID_SCAN`` (about half a minute)."""
    result: dict[str, list[float]] = {"local": [], "marginal": [], "kalman": []}
    for intervals in GRID_SCAN["intervals_per_day"]:
        toy = make_toy(intervals_per_day=intervals, every=every)
        model = reference_model(toy)
        start = posterior_draws(model, 1, seed=1)[0]

        def smooth(paths: np.ndarray, toy: Toy = toy, model: GaussianModel = model) -> float:
            return shape_summary(refresh(paths[50:], model), toy)[0]

        result["local"].append(
            max(
                smooth(run_separable(toy, start, sweeps=sweeps, particles=particles, delta=delta))
                for delta in (0.01, 0.03, 0.1, 0.3, 1.0)
            )
        )
        result["marginal"].append(
            smooth(run_separable(toy, start, sweeps=sweeps, particles=particles, smoother=model))
        )
        result["kalman"].append(smooth(run_kalman(toy, start, model, sweeps=sweeps)[0]))
    return {key: tuple(values) for key, values in result.items()}


def fig_knobs(
    deltas: list[float],
    delta_smooth: list[float],
    current: dict[str, float],
    scan: GridScan,
    effects: list[float],
    mismatch_acceptance: list[float],
    mismatch_smooth: list[float],
) -> go.Figure:
    """Smooth-shape refresh against δ, grid density and a frozen Kalman model's error."""
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=(
            "δ, separable local",
            "Grid density (dashed: every 16th reading)",
            "Kalman model frozen at a wrong effect",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=delta_smooth,
            mode="lines+markers",
            line={"width": 2, "color": C_LOCAL},
            marker={"size": 8, "color": C_LOCAL, "line": {"width": 2, "color": C_SURFACE}},
            name=NAMES["local"],
            legendgroup="local",
            hovertemplate="δ = %{x:g}<br>smooth refresh %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for key in ("marginal", "kalman"):
        fig.add_trace(
            go.Scatter(
                x=[deltas[0], deltas[-1]],
                y=[current[key]] * 2,
                mode="lines",
                line={"width": 2, "color": COLORS[key]},
                name=NAMES[key],
                legendgroup=key,
                hovertemplate="%{y:.2f}<extra>" + NAMES[key] + "</extra>",
            ),
            row=1,
            col=1,
        )
    intervals = list(scan["intervals_per_day"])
    for readings, dash in ((scan["dense"], "solid"), (scan["sparse"], "dash")):
        for key in ("local", "marginal", "kalman"):
            fig.add_trace(
                go.Scatter(
                    x=intervals,
                    y=list(readings[key]),
                    mode="lines+markers",
                    line={"width": 2, "color": COLORS[key], "dash": dash},
                    marker={
                        "size": 7,
                        "color": COLORS[key],
                        "line": {"width": 2, "color": C_SURFACE},
                    },
                    name=NAMES[key],
                    legendgroup=key,
                    showlegend=False,
                    hovertemplate="%{x} intervals/day<br>%{y:.2f}<extra>" + NAMES[key] + "</extra>",
                ),
                row=1,
                col=2,
            )
    fig.add_trace(
        go.Scatter(
            x=effects,
            y=mismatch_acceptance,
            mode="lines+markers",
            line={"width": 2, "color": C_KALMAN, "dash": "dot"},
            marker={"size": 7, "color": C_KALMAN, "line": {"width": 2, "color": C_SURFACE}},
            name="Kalman acceptance",
            hovertemplate="assumed %{x:.1f}<br>acceptance %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=effects,
            y=mismatch_smooth,
            mode="lines+markers",
            line={"width": 2, "color": C_KALMAN},
            marker={"size": 7, "color": C_KALMAN, "line": {"width": 2, "color": C_SURFACE}},
            name=NAMES["kalman"],
            legendgroup="kalman",
            showlegend=False,
            hovertemplate="assumed %{x:.1f}<br>smooth refresh %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[effects[0], effects[-1]],
            y=[max(delta_smooth)] * 2,
            mode="lines",
            line={"width": 2, "color": C_LOCAL},
            name=NAMES["local"],
            legendgroup="local",
            showlegend=False,
            hovertemplate="best δ: %{y:.2f}<extra>" + NAMES["local"] + "</extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(
        type="log",
        tickvals=deltas,
        ticktext=[f"{value:g}" for value in deltas],
        title="δ",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="log",
        tickvals=intervals,
        ticktext=[str(value) for value in intervals],
        title="intervals per day (C0: 128)",
        row=1,
        col=2,
    )
    fig.update_xaxes(
        autorange="reversed", title="input effect the model assumes (truth 0.5)", row=1, col=3
    )
    fig.update_yaxes(title="smooth-shape refresh per sweep", range=[0, 1.12], row=1, col=1)
    return _layout(fig, title="What changes the picture", height=420)


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
                        "line": {"width": 2, "color": C_SURFACE},
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
