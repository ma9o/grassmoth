"""Support code for the particle-sampler surface walkthrough.

The notebook re-runs every sampler family of the 2026-10-14 T=512 warmup benchmark
on a posterior with two coordinates, ``theta`` (a parameter) and ``x`` (the latent
path, collapsed to one coordinate), so each move can be drawn on the surface
``p(theta, x | y)``.

The particle kernels follow production one-to-one at a single time step:

- ``dsmc_leaf`` is the conditional DSMC sweep of
  ``models/ssm/inference/marginal_particle_gibbs/dsmc.py`` with ``T = 1``: an auxiliary
  point ``z ~ N(x_ref, delta/2)``, ``N - 1`` particles around a gradient-shifted ``z``,
  weights against the posterior mixture over the parameter labels, and one draw.
- ``mpgibbs_sweep`` is the collapsed parameter/path move of
  ``marginal_particle_gibbs/kernel.py``: an auxiliary parameter proposal with the
  Barker label correction, the smoother above against both labels, then the label draw.
- ``run_laps`` replays ``warmup/laps_ensemble.py`` (BlackJAX LAPS with the production
  settings) while recording the ensemble at every step.

Everything else (exact conditionals, marginals and diagnostics) is closed form or a
fine one-dimensional grid, so each sampler can be checked against the exact answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation import laps, laps_burn_in
from blackjax.mcmc.adjusted_mclmc import build_kernel as build_adjusted_kernel
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    IntegratorState,
    generate_isokinetic_integrator,
    mclachlan_coefficients,
)
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
WorldKind = Literal["ridge", "funnel"]
SmootherKind = Literal["dsmc", "exact"]
ParameterProposal = Literal["random_walk", "pseudo_langevin"]
LeafProposal = Literal["auxiliary", "bootstrap"]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _normal_logpdf(value: FloatArray | float, mean: FloatArray | float, var: FloatArray | float):
    return -0.5 * (_LOG_2PI + np.log(var) + np.square(np.subtract(value, mean)) / var)


def _normalize(logits: FloatArray) -> FloatArray:
    return logits - logsumexp(logits)


def _draw_index(rng: np.random.Generator, log_probs: FloatArray) -> int:
    probs = np.exp(_normalize(log_probs))
    return int(rng.choice(probs.size, p=probs / probs.sum()))


# ── Worlds ────────────────────────────────────────────────────────────────────


class World(Protocol):
    """A two-coordinate posterior ``p(theta) p(x | theta) p(y | x)``."""

    @property
    def kind(self) -> WorldKind: ...

    def log_prior(self, theta: FloatArray | float) -> FloatArray: ...

    def log_latent(self, x: FloatArray | float, theta: FloatArray | float) -> FloatArray: ...

    def log_obs(self, x: FloatArray | float) -> FloatArray: ...

    def grad_theta(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray: ...

    def grad_x(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray: ...

    def x_given_theta(self, theta: FloatArray | float) -> tuple[FloatArray, FloatArray]: ...

    def draw_latent_prior(
        self, theta: float, rng: np.random.Generator, size: int
    ) -> FloatArray: ...

    def draw_theta(self, x: float, rng: np.random.Generator) -> float: ...

    def log_evidence(self, theta: FloatArray | float) -> FloatArray: ...

    def theta_box(self) -> tuple[float, float]: ...

    def x_box(self) -> tuple[float, float]: ...

    def jax_params(self) -> jax.Array: ...


@dataclass(frozen=True)
class Ridge:
    """``theta`` is a baseline; the latent wanders around it; ``y`` reads the latent.

    ``tau`` is how far the latent may wander from what ``theta`` predicts (process
    noise) and ``sigma`` how blurred the reading is (observation noise).
    """

    tau: float
    sigma: float
    y: float = 1.5
    prior_sd: float = 1.0

    @property
    def kind(self) -> WorldKind:
        return "ridge"

    def log_prior(self, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(theta, 0.0, self.prior_sd**2))

    def log_latent(self, x: FloatArray | float, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(x, theta, self.tau**2))

    def log_obs(self, x: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(self.y, x, self.sigma**2))

    def grad_theta(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
        return np.asarray(-np.divide(theta, self.prior_sd**2) + np.subtract(x, theta) / self.tau**2)

    def grad_x(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
        return np.asarray(
            -np.subtract(x, theta) / self.tau**2 + np.subtract(self.y, x) / self.sigma**2
        )

    def x_given_theta(self, theta: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        var = 1.0 / (1.0 / self.tau**2 + 1.0 / self.sigma**2)
        mean = var * (np.divide(theta, self.tau**2) + self.y / self.sigma**2)
        return np.asarray(mean), np.full(np.shape(mean), var)

    def theta_given_x(self, x: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        """Exact parameter conditional: what arm G draws."""
        var = 1.0 / (1.0 / self.prior_sd**2 + 1.0 / self.tau**2)
        mean = var * np.divide(x, self.tau**2)
        return np.asarray(mean), np.full(np.shape(mean), var)

    def draw_latent_prior(self, theta: float, rng: np.random.Generator, size: int) -> FloatArray:
        return theta + self.tau * rng.standard_normal(size)

    def draw_theta(self, x: float, rng: np.random.Generator) -> float:
        mean, var = self.theta_given_x(x)
        return float(mean + np.sqrt(var) * rng.standard_normal())

    def log_evidence(self, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(self.y, theta, self.tau**2 + self.sigma**2))

    def moments(self) -> tuple[FloatArray, FloatArray]:
        """Exact posterior mean and covariance of ``(theta, x)``."""
        precision = np.array(
            [
                [1.0 / self.prior_sd**2 + 1.0 / self.tau**2, -1.0 / self.tau**2],
                [-1.0 / self.tau**2, 1.0 / self.tau**2 + 1.0 / self.sigma**2],
            ]
        )
        covariance = np.linalg.inv(precision)
        return covariance @ np.array([0.0, self.y / self.sigma**2]), covariance

    def theta_box(self) -> tuple[float, float]:
        mean, cov = self.moments()
        half = 4.2 * float(np.sqrt(cov[0, 0]))
        return float(mean[0]) - half, float(mean[0]) + half

    def x_box(self) -> tuple[float, float]:
        mean, cov = self.moments()
        half = 4.2 * float(np.sqrt(cov[1, 1]))
        return float(mean[1]) - half, float(mean[1]) + half

    def jax_params(self) -> jax.Array:
        return jnp.asarray([self.tau, self.sigma, self.y, self.prior_sd])


_FUNNEL_GRID = np.linspace(-7.0, 4.0, 4001)


@dataclass(frozen=True)
class Funnel:
    """``theta`` is the log spread of the latent (a noise scale); ``y`` reads the latent.

    With ``y = 0`` the posterior is symmetric in ``x``: the linear correlation of
    ``theta`` and ``x`` is zero, yet each one sets the other's scale.
    """

    sigma: float
    y: float = 0.0
    prior_mean: float = -0.5
    prior_sd: float = 1.0

    @property
    def kind(self) -> WorldKind:
        return "funnel"

    def log_prior(self, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(theta, self.prior_mean, self.prior_sd**2))

    def log_latent(self, x: FloatArray | float, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(x, 0.0, np.exp(np.multiply(2.0, theta))))

    def log_obs(self, x: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(self.y, x, self.sigma**2))

    def grad_theta(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
        return np.asarray(
            -np.subtract(theta, self.prior_mean) / self.prior_sd**2
            - 1.0
            + np.square(x) * np.exp(np.multiply(-2.0, theta))
        )

    def grad_x(self, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
        return np.asarray(
            -np.multiply(x, np.exp(np.multiply(-2.0, theta))) + np.subtract(self.y, x) / self.sigma**2
        )

    def x_given_theta(self, theta: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        var = 1.0 / (np.exp(np.multiply(-2.0, theta)) + 1.0 / self.sigma**2)
        return np.asarray(var * self.y / self.sigma**2), np.asarray(var)

    def theta_logpdf_given_x(self, x: float) -> FloatArray:
        """Unnormalized parameter conditional on the shared grid."""
        return np.asarray(self.log_prior(_FUNNEL_GRID) + self.log_latent(x, _FUNNEL_GRID))

    def draw_latent_prior(self, theta: float, rng: np.random.Generator, size: int) -> FloatArray:
        return np.exp(theta) * rng.standard_normal(size)

    def draw_theta(self, x: float, rng: np.random.Generator) -> float:
        log_density = self.theta_logpdf_given_x(x)
        density = np.exp(log_density - log_density.max())
        cdf = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(_FUNNEL_GRID))
        cdf = np.concatenate(([0.0], cdf / cdf[-1]))
        return float(np.interp(rng.uniform(), cdf, _FUNNEL_GRID))

    def log_evidence(self, theta: FloatArray | float) -> FloatArray:
        return np.asarray(
            _normal_logpdf(self.y, 0.0, np.exp(np.multiply(2.0, theta)) + self.sigma**2)
        )

    def theta_box(self) -> tuple[float, float]:
        low, high = np.interp([0.002, 0.998], _marginal_cdf(self), _FUNNEL_GRID)
        return float(low) - 0.4, float(high) + 0.4

    def x_box(self) -> tuple[float, float]:
        _, var = self.x_given_theta(self.theta_box()[1])
        half = 3.6 * float(np.sqrt(var))
        return self.y - half, self.y + half

    def jax_params(self) -> jax.Array:
        return jnp.asarray([self.sigma, self.y, self.prior_mean, self.prior_sd])


def log_joint(world: World, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
    """``log p(theta) + log p(x | theta) + log p(y | x)``."""
    return world.log_prior(theta) + world.log_latent(x, theta) + world.log_obs(x)


@dataclass(frozen=True)
class Surface:
    """The joint density on a grid with its exact parameter marginal."""

    theta: FloatArray
    x: FloatArray
    log_density: FloatArray  # (x, theta)
    theta_marginal: FloatArray  # density of p(theta | y) on ``theta``

    @property
    def density(self) -> FloatArray:
        return np.exp(self.log_density - self.log_density.max())


def surface(world: World, resolution: int = 161) -> Surface:
    """Evaluate the joint on a grid and the exact marginal ``p(theta | y)``."""
    theta = np.linspace(*world.theta_box(), resolution)
    x = np.linspace(*world.x_box(), resolution)
    log_density = log_joint(world, theta[None, :], x[:, None])
    log_marginal = world.log_prior(theta) + world.log_evidence(theta)
    marginal = np.exp(log_marginal - log_marginal.max())
    marginal /= np.trapezoid(marginal, theta)
    return Surface(theta, x, np.asarray(log_density), marginal)


def _marginal_cdf(world: Funnel) -> FloatArray:
    log_marginal = world.log_prior(_FUNNEL_GRID) + world.log_evidence(_FUNNEL_GRID)
    density = np.exp(log_marginal - log_marginal.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]))))
    return cdf / cdf[-1]


def exact_draws(world: World, size: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Independent posterior draws: ``theta`` from its marginal, then ``x | theta``."""
    rng = np.random.default_rng(seed)
    if isinstance(world, Ridge):
        mean, cov = world.moments()
        theta = mean[0] + np.sqrt(cov[0, 0]) * rng.standard_normal(size)
    elif isinstance(world, Funnel):
        theta = np.interp(rng.uniform(size=size), _marginal_cdf(world), _FUNNEL_GRID)
    else:
        raise TypeError(world)
    x_mean, x_var = world.x_given_theta(theta)
    return theta, x_mean + np.sqrt(x_var) * rng.standard_normal(size)


@dataclass(frozen=True)
class Coupling:
    """How much of the parameter's posterior uncertainty is uncertainty about the latent.

    ``share = 1 - E[Var(theta | x, y)] / Var(theta | y)`` is the lag-1
    autocorrelation of ``theta`` under exact two-block Gibbs, for any model. For a
    Gaussian it is ``rho^2`` and every lag k has autocorrelation ``share^k``.
    """

    mean: float
    var: float
    conditional_var: float

    @property
    def share(self) -> float:
        return 1.0 - self.conditional_var / self.var

    @property
    def gibbs_ess_per_sweep(self) -> float:
        """``(1 - share) / (1 + share)``: exact for a Gaussian surface."""
        return (1.0 - self.share) / (1.0 + self.share)


def coupling(world: World) -> Coupling:
    """Exact marginal moments of ``theta`` and its mean conditional variance."""
    if isinstance(world, Ridge):
        mean, cov = world.moments()
        conditional = float(world.theta_given_x(0.0)[1])
        return Coupling(float(mean[0]), float(cov[0, 0]), conditional)
    if isinstance(world, Funnel):
        weights = np.diff(_marginal_cdf(world), prepend=0.0)
        mean = float(np.sum(weights * _FUNNEL_GRID))
        var = float(np.sum(weights * np.square(_FUNNEL_GRID - mean)))
        _, latent = exact_draws(world, 600, seed=0)
        log_density = world.log_prior(_FUNNEL_GRID)[None, :] + world.log_latent(
            latent[:, None], _FUNNEL_GRID[None, :]
        )
        density = np.exp(log_density - log_density.max(axis=1, keepdims=True))
        density /= density.sum(axis=1, keepdims=True)
        centers = density @ _FUNNEL_GRID
        spreads = np.sum(density * np.square(_FUNNEL_GRID[None, :] - centers[:, None]), axis=1)
        return Coupling(mean, var, float(np.mean(spreads)))
    raise TypeError(world)


# ── Samplers ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Chain:
    """Sweep-by-sweep states of one chain.

    ``blocked`` chains move ``theta`` first and then ``x`` (a staircase on the
    surface); the others move both coordinates in one step.
    """

    theta: FloatArray
    x: FloatArray
    theta_moved: BoolArray
    x_moved: BoolArray
    blocked: bool


@dataclass(frozen=True)
class Sweep:
    """Everything one particle sweep did, for drawing it."""

    thetas: FloatArray  # parameter labels; index 0 is the current value
    label_log_prior: FloatArray  # normalized label prior (with the Barker correction)
    aux: float  # auxiliary point z of the local proposal
    center: float  # proposal centre after the gradient shift
    particles: FloatArray  # index 0 is the reference
    psi: FloatArray  # (particles, labels) potentials
    weights: FloatArray  # selection probabilities
    chosen: int
    label_probs: FloatArray  # label probabilities given the chosen latent
    label: int


def dsmc_leaf(
    world: World,
    thetas: FloatArray,
    label_log_prior: FloatArray,
    x_ref: float,
    rng: np.random.Generator,
    *,
    particles: int,
    delta: float,
    kappa: float,
) -> tuple[float, float, float, FloatArray, FloatArray, FloatArray, int]:
    """Conditional DSMC at one time step against the mixture over parameter labels.

    Returns the new latent, the auxiliary point, the proposal centre, the particles,
    their per-label potentials, the selection probabilities and the chosen index.
    """
    var = 0.5 * delta
    aux = x_ref + np.sqrt(var) * rng.standard_normal()
    # Mixture-weighted gradient at the auxiliary point, as in production's leaf.
    component = label_log_prior + world.log_latent(aux, thetas) + world.log_obs(aux)
    gradient = float(np.sum(np.exp(_normalize(component)) * world.grad_x(thetas, aux)))
    center = aux + kappa * var * gradient
    free = center + np.sqrt(var) * rng.standard_normal(particles - 1)
    bank = np.concatenate(([x_ref], free))
    proposal = _normal_logpdf(bank, center, var)
    pseudo_observation = _normal_logpdf(bank, aux, var)
    psi = (
        world.log_latent(bank[:, None], thetas[None, :])
        + world.log_obs(bank)[:, None]
        - proposal[:, None]
        + pseudo_observation[:, None]
    )
    evidence = logsumexp(label_log_prior[None, :] + psi, axis=1)
    weights = np.exp(_normalize(evidence))
    chosen = _draw_index(rng, evidence)
    return float(bank[chosen]), float(aux), float(center), bank, np.asarray(psi), weights, chosen


def bootstrap_leaf(
    world: World, theta: float, x_ref: float, rng: np.random.Generator, *, particles: int
) -> tuple[float, FloatArray, FloatArray, int]:
    """Classic particle Gibbs at one time step: propose from ``p(x | theta)``."""
    bank = np.concatenate(([x_ref], world.draw_latent_prior(theta, rng, particles - 1)))
    log_weights = world.log_obs(bank)
    chosen = _draw_index(rng, log_weights)
    return float(bank[chosen]), bank, np.exp(_normalize(log_weights)), chosen


def gibbs(world: World, sweeps: int, start: tuple[float, float], seed: int) -> Chain:
    """Exact two-block Gibbs: ``theta ~ p(theta | x)``, then ``x ~ p(x | theta, y)``."""
    rng = np.random.default_rng(seed)
    theta, x = start
    thetas, xs = np.empty(sweeps), np.empty(sweeps)
    for sweep in range(sweeps):
        theta = world.draw_theta(x, rng)
        mean, var = world.x_given_theta(theta)
        x = float(mean + np.sqrt(var) * rng.standard_normal())
        thetas[sweep], xs[sweep] = theta, x
    moved = np.ones(sweeps, dtype=bool)
    return Chain(thetas, xs, moved, moved, blocked=True)


def _adapt(value: float, hit: bool, target: float, warm: bool) -> float:
    """Production's simple rule during warmup: ``value * exp(0.05 (hit - target))``."""
    return value * float(np.exp(0.05 * (float(hit) - target))) if warm else value


def particle_gibbs(
    world: World,
    sweeps: int,
    start: tuple[float, float],
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float,
    leaf: LeafProposal = "auxiliary",
    warmup: int = 0,
) -> Chain:
    """Exact parameter draw, then one conditional particle sweep for the latent.

    With the auxiliary leaf this is the benchmark's arm G (exact Gaussian parameters
    + conditional DSMC); arms D and F replace the exact draw by LAPS or NUTS. During
    ``warmup`` sweeps ``delta`` adapts towards production's 0.75 latent move rate;
    the returned chain excludes them.
    """
    rng = np.random.default_rng(seed)
    theta, x = start
    thetas, xs = np.empty(sweeps), np.empty(sweeps)
    x_moved = np.zeros(sweeps, dtype=bool)
    for sweep in range(-warmup, sweeps):
        theta = world.draw_theta(x, rng)
        if leaf == "auxiliary":
            new_x = dsmc_leaf(
                world,
                np.array([theta]),
                np.zeros(1),
                x,
                rng,
                particles=particles,
                delta=delta,
                kappa=kappa,
            )[0]
        else:
            new_x = bootstrap_leaf(world, theta, x, rng, particles=particles)[0]
        delta = _adapt(delta, new_x != x, 0.75, sweep < 0)
        if sweep >= 0:
            x_moved[sweep] = new_x != x
            thetas[sweep], xs[sweep] = theta, new_x
        x = new_x
    return Chain(thetas, xs, np.ones(sweeps, dtype=bool), x_moved, blocked=True)


def mpgibbs_sweep(
    world: World,
    theta: float,
    x_ref: float,
    rng: np.random.Generator,
    *,
    particles: int,
    delta: float,
    kappa: float,
    step: float,
    scale: float,
    proposal: ParameterProposal,
    anchor_x: float,
    smoother: SmootherKind,
) -> Sweep:
    """One marginalized particle Gibbs move with two parameter labels.

    ``u | theta ~ N(theta + 2 step scale^2 g(theta), 2 step scale^2)`` and
    ``theta' | u ~ N(u, 2 step scale^2)``, with ``g`` the complete-data gradient at
    the fixed ``anchor_x`` (production's fixed pilot path) for pseudo-Langevin and
    ``g = 0`` for the random walk. The latent is then updated against the mixture
    over ``{theta, theta'}`` and a label is drawn given the new latent.
    ``smoother="exact"`` replaces the particle sweep by an exact mixture draw, the
    infinite-particle limit.
    """
    var = 2.0 * step * scale**2
    drift = var * float(world.grad_theta(theta, anchor_x)) if proposal == "pseudo_langevin" else 0.0
    u = theta + drift + np.sqrt(var) * rng.standard_normal()
    thetas = np.array([theta, u + np.sqrt(var) * rng.standard_normal()])
    if proposal == "pseudo_langevin":
        gradients = world.grad_theta(thetas, anchor_x)
        correction = gradients * (u - thetas) - step * np.square(scale * gradients)
    else:
        correction = np.zeros(2)
    parameter_log_probs = world.log_prior(thetas) + correction
    label_log_prior = _normalize(parameter_log_probs)
    if smoother == "exact":
        # Infinite particles: pick a mixture component by evidence, draw its slice.
        component = _draw_index(rng, label_log_prior + world.log_evidence(thetas))
        mean, var_x = world.x_given_theta(float(thetas[component]))
        new_x = float(mean + np.sqrt(var_x) * rng.standard_normal())
        aux = center = new_x
        bank, psi, weights, chosen = np.array([new_x]), np.zeros((1, 2)), np.ones(1), 0
    else:
        new_x, aux, center, bank, psi, weights, chosen = dsmc_leaf(
            world,
            thetas,
            label_log_prior,
            x_ref,
            rng,
            particles=particles,
            delta=delta,
            kappa=kappa,
        )
    label_logits = parameter_log_probs + world.log_latent(new_x, thetas) + world.log_obs(new_x)
    label = _draw_index(rng, label_logits)
    return Sweep(
        thetas=thetas,
        label_log_prior=label_log_prior,
        aux=aux,
        center=center,
        particles=bank,
        psi=psi,
        weights=weights,
        chosen=chosen,
        label_probs=np.exp(_normalize(label_logits)),
        label=label,
    )


@dataclass(frozen=True)
class MPGibbsChain:
    """An mPGibbs chain with the tuning it froze after warmup."""

    chain: Chain
    step: float
    delta: float


def mpgibbs(
    world: World,
    sweeps: int,
    start: tuple[float, float],
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float,
    step: float,
    scale: float,
    proposal: ParameterProposal,
    smoother: SmootherKind,
    anchor_x: float,
    warmup: int = 0,
) -> MPGibbsChain:
    """Chain of ``mpgibbs_sweep`` moves.

    During ``warmup`` sweeps the parameter step adapts towards production's 0.35
    label-switch rate and ``delta`` towards the 0.75 latent move rate; both are then
    frozen and the warmup sweeps are dropped from the chain.
    """
    rng = np.random.default_rng(seed)
    theta, x = start
    thetas, xs = np.empty(sweeps), np.empty(sweeps)
    theta_moved, x_moved = np.zeros(sweeps, dtype=bool), np.zeros(sweeps, dtype=bool)
    for sweep in range(-warmup, sweeps):
        move = mpgibbs_sweep(
            world,
            theta,
            x,
            rng,
            particles=particles,
            delta=delta,
            kappa=kappa,
            step=step,
            scale=scale,
            proposal=proposal,
            anchor_x=anchor_x,
            smoother=smoother,
        )
        new_theta, new_x = float(move.thetas[move.label]), float(move.particles[move.chosen])
        step = _adapt(step, move.label == 1, 0.35, sweep < 0)
        delta = _adapt(delta, new_x != x, 0.75, sweep < 0 and smoother == "dsmc")
        if sweep >= 0:
            theta_moved[sweep], x_moved[sweep] = move.label == 1, new_x != x
            thetas[sweep], xs[sweep] = new_theta, new_x
        theta, x = new_theta, new_x
    return MPGibbsChain(Chain(thetas, xs, theta_moved, x_moved, blocked=False), step, delta)


def pmmh(
    world: World,
    sweeps: int,
    start: tuple[float, float],
    seed: int,
    *,
    particles: int,
    step: float,
) -> Chain:
    """Particle marginal Metropolis–Hastings with a bootstrap likelihood estimate."""
    rng = np.random.default_rng(seed)

    def estimate(theta: float) -> tuple[float, FloatArray, FloatArray]:
        bank = world.draw_latent_prior(theta, rng, particles)
        log_weights = world.log_obs(bank)
        return float(logsumexp(log_weights) - np.log(particles)), bank, log_weights

    theta, x = start
    log_evidence, bank, log_weights = estimate(theta)
    thetas, xs = np.empty(sweeps), np.empty(sweeps)
    theta_moved = np.zeros(sweeps, dtype=bool)
    for sweep in range(sweeps):
        proposal = theta + step * rng.standard_normal()
        proposal_evidence, proposal_bank, proposal_weights = estimate(proposal)
        log_ratio = (
            float(world.log_prior(proposal)) + proposal_evidence
            - float(world.log_prior(theta)) - log_evidence
        )
        if np.log(rng.uniform()) < log_ratio:
            theta, log_evidence, bank, log_weights = (
                proposal,
                proposal_evidence,
                proposal_bank,
                proposal_weights,
            )
            theta_moved[sweep] = True
            x = float(bank[_draw_index(rng, log_weights)])
        thetas[sweep], xs[sweep] = theta, x
    return Chain(thetas, xs, theta_moved, theta_moved.copy(), blocked=False)


# ── Diagnostics ───────────────────────────────────────────────────────────────


def autocorrelation(values: FloatArray, max_lag: int) -> FloatArray:
    """Sample autocorrelation at lags ``0..max_lag``."""
    centered = values - values.mean()
    size = centered.size
    spectrum = np.fft.rfft(centered, 2 * size)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum))[:size] / size
    if autocovariance[0] <= 0.0:
        return np.ones(max_lag + 1)
    return autocovariance[: max_lag + 1] / autocovariance[0]


def ess(values: FloatArray) -> float:
    """Effective sample size of one chain (Geyer's initial monotone sequence)."""
    size = values.size
    rho = autocorrelation(values, size - 1)
    if np.all(rho == 1.0):
        return 1.0
    pairs = rho[: size - 1 : 2] + rho[1:size:2]
    positive = np.flatnonzero(pairs <= 0.0)
    pairs = pairs[: positive[0]] if positive.size else pairs
    pairs = np.minimum.accumulate(pairs)
    tau = max(-1.0 + 2.0 * float(np.sum(pairs)), 1.0 / np.log10(max(size, 10)))
    return float(size / tau)


def pooled_ess(chains: FloatArray) -> float:
    """Sum of per-chain ESS over the leading axis of ``chains``."""
    return float(sum(ess(chain) for chain in chains))


# ── Joint gradient samplers (BlackJAX) ────────────────────────────────────────


def _jax_log_joint(kind: WorldKind, params: jax.Array, position: jax.Array) -> jax.Array:
    theta, x = position[0], position[1]

    def normal(value: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
        return -0.5 * (_LOG_2PI + jnp.log(var) + jnp.square(value - mean) / var)

    if kind == "ridge":
        tau, sigma, y, prior_sd = params[0], params[1], params[2], params[3]
        return normal(theta, jnp.asarray(0.0), prior_sd**2) + normal(x, theta, tau**2) + normal(
            y, x, sigma**2
        )
    sigma, y, prior_mean, prior_sd = params[0], params[1], params[2], params[3]
    return (
        normal(theta, prior_mean, prior_sd**2)
        + normal(x, jnp.asarray(0.0), jnp.exp(2.0 * theta))
        + normal(y, x, sigma**2)
    )


@dataclass(frozen=True)
class LapsRun:
    """The LAPS ensemble at every step, and what it hands to the particle sampler."""

    positions: FloatArray  # (steps, members, 2): unadjusted steps, then adjusted
    step_size: FloatArray  # (steps,)
    active: BoolArray  # unadjusted steps actually executed before the early stop
    unadjusted_steps: int  # length of the unadjusted block in ``positions``
    acceptance: FloatArray  # adjusted-phase acceptance per step
    inverse_mass: FloatArray  # (2,) diagonal preconditioner learned in burn-in
    samples: FloatArray  # (chains, draws, 2) frozen adjusted kernel, for mixing only


@partial(
    jax.jit,
    static_argnames=(
        "kind",
        "members",
        "unadjusted",
        "adjusted",
        "steps_per_sample",
        "chains",
        "draws",
    ),
)
def _laps_program(
    kind: WorldKind,
    params: jax.Array,
    key: jax.Array,
    *,
    members: int,
    unadjusted: int,
    adjusted: int,
    steps_per_sample: int,
    chains: int,
    draws: int,
):
    def log_density(position: jax.Array) -> jax.Array:
        return _jax_log_joint(kind, params, position)

    dimension = 2
    initial_key, burn_key, adjusted_key, sample_key = jax.random.split(key, 4)
    positions = jax.random.normal(initial_key, (members, dimension))
    values, gradients = jax.vmap(jax.value_and_grad(log_density))(positions)
    signs = jnp.where(jnp.mean(-positions * gradients, axis=0) < 1, -1.0, 1.0)
    scaled = gradients / jnp.max(jnp.abs(gradients), axis=-1, keepdims=True)
    momentum = signs * scaled / jnp.linalg.norm(scaled, axis=-1, keepdims=True)
    state = IntegratorState(positions, momentum, values, gradients)

    adaptation = laps_burn_in.Adaptation(
        dimension,
        microcanonical=True,
        alpha=1.9,
        C=0.1,
        r_end=0.01,
        bias_type=3,
        save_num=max(2, round(0.2 * unadjusted)),
    )
    burn_kernel = laps_burn_in.build_kernel(log_density, dimension, microcanonical=True)

    def burn(carry, index):
        current, tuning, active = carry
        sampling_key, summary_key = jax.random.split(jax.random.fold_in(burn_key, index))
        moved, info = jax.vmap(lambda member_key, member: burn_kernel(member_key, member, tuning))(
            jax.random.split(sampling_key, members), current
        )
        summaries = jax.vmap(adaptation.summary_statistics_fn, in_axes=(0, 0, None))(
            moved, info, summary_key
        )
        new_tuning, summary = adaptation.update(
            tuning, jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), summaries)
        )
        keep_going = (
            adaptation.while_cond(summary, index)
            & jnp.isfinite(new_tuning.step_size)
            & (new_tuning.step_size > 0)
        )
        current = jax.tree.map(lambda new, old: jnp.where(active, new, old), moved, current)
        tuning = jax.tree.map(lambda new, old: jnp.where(active, new, old), new_tuning, tuning)
        return (current, tuning, active & keep_going), (current.position, tuning.step_size, active)

    (state, burn_tuning, _), (burn_positions, burn_steps, burn_active) = jax.lax.scan(
        burn, (state, adaptation.initial_state, jnp.asarray(True)), jnp.arange(unadjusted)
    )

    adjusted_kernel = build_adjusted_kernel(
        integrator=generate_isokinetic_integrator(mclachlan_coefficients)
    )
    inverse_mass = burn_tuning.inverse_mass_matrix
    adjusted_adaptation = laps.Adaptation(
        burn_tuning._replace(step_size=burn_tuning.step_size / jnp.sqrt(jnp.mean(inverse_mass))),
        num_adaptation_samples=adjusted,
        steps_per_sample=steps_per_sample,
        acc_prob_target=0.7,
    )

    def move(chain_key, chain_state, step_size):
        return adjusted_kernel(
            rng_key=chain_key,
            state=chain_state,
            logdensity_fn=log_density,
            step_size=step_size,
            integration_steps_params=(steps_per_sample,),
            inverse_mass_matrix=inverse_mass,
            L_proposal_factor=1.25,
        )

    def refine(carry, index):
        current, tuning = carry
        sampling_key, summary_key = jax.random.split(jax.random.fold_in(adjusted_key, index))
        current, info = jax.vmap(lambda chain_key, chain: move(chain_key, chain, tuning.step_size))(
            jax.random.split(sampling_key, members), current
        )
        summaries = jax.vmap(adjusted_adaptation.summary_statistics_fn, in_axes=(0, 0, None))(
            current, info, summary_key
        )
        tuning, summary = adjusted_adaptation.update(
            tuning, jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), summaries)
        )
        return (current, tuning), (current.position, tuning.step_size, summary["acc_prob"])

    adjusted_state = HMCState(state.position, state.logdensity, state.logdensity_grad)
    (adjusted_state, tuning), (adjusted_positions, adjusted_steps, acceptance) = jax.lax.scan(
        refine, (adjusted_state, adjusted_adaptation.initial_state), jnp.arange(adjusted)
    )

    # Mixing check only: production discards LAPS draws and samples with the particle kernel.
    sampler_state = jax.tree.map(lambda leaf: leaf[:chains], adjusted_state)

    def sample(current, index):
        current, _ = jax.vmap(lambda chain_key, chain: move(chain_key, chain, tuning.step_size))(
            jax.random.split(jax.random.fold_in(sample_key, index), chains), current
        )
        return current, current.position

    _, samples = jax.lax.scan(sample, sampler_state, jnp.arange(draws))
    return (
        jnp.concatenate([burn_positions, adjusted_positions]),
        jnp.concatenate([burn_steps, adjusted_steps]),
        burn_active,
        acceptance,
        inverse_mass,
        jnp.swapaxes(samples, 0, 1),
    )


def run_laps(
    world: World,
    seed: int,
    *,
    members: int = 64,
    unadjusted: int = 1000,
    adjusted: int = 50,
    steps_per_sample: int = 15,
    chains: int = 4,
    draws: int = 1000,
) -> LapsRun:
    """Production LAPS settings (benchmark budget by default), recording every step."""
    positions, step_size, active, acceptance, inverse_mass, samples = _laps_program(
        world.kind,
        world.jax_params(),
        jax.random.key(seed),
        members=members,
        unadjusted=unadjusted,
        adjusted=adjusted,
        steps_per_sample=steps_per_sample,
        chains=chains,
        draws=draws,
    )
    return LapsRun(
        positions=np.asarray(positions, dtype=np.float64),
        step_size=np.asarray(step_size, dtype=np.float64),
        active=np.asarray(active, dtype=bool),
        unadjusted_steps=unadjusted,
        acceptance=np.asarray(acceptance, dtype=np.float64),
        inverse_mass=np.asarray(inverse_mass, dtype=np.float64),
        samples=np.asarray(samples, dtype=np.float64),
    )


@partial(jax.jit, static_argnames=("kind", "warmup", "draws"))
def _nuts_program(
    kind: WorldKind, params: jax.Array, key: jax.Array, start: jax.Array, *, warmup: int, draws: int
):
    def log_density(position: jax.Array) -> jax.Array:
        return _jax_log_joint(kind, params, position)

    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts, log_density, is_mass_matrix_diagonal=False
    )
    (state, tuned), _ = adaptation.run(warmup_key, start, num_steps=warmup)
    kernel = blackjax.nuts(log_density, **tuned)

    def step(current, step_key):
        current, info = kernel.step(step_key, current)
        return current, (current.position, info.num_integration_steps)

    _, (positions, integration) = jax.lax.scan(step, state, jax.random.split(sample_key, draws))
    return positions, integration


def run_nuts(
    world: World, seed: int, start: tuple[float, float], *, warmup: int = 500, draws: int = 2000
) -> tuple[Chain, float]:
    """Joint NUTS on ``(theta, x)``: the 2D analogue of the golden reference.

    Returns the chain and the mean number of gradient evaluations per draw.
    """
    positions, integration = _nuts_program(
        world.kind,
        world.jax_params(),
        jax.random.key(seed),
        jnp.asarray(start),
        warmup=warmup,
        draws=draws,
    )
    positions = np.asarray(positions, dtype=np.float64)
    moved = np.ones(draws, dtype=bool)
    chain = Chain(positions[:, 0], positions[:, 1], moved, moved, blocked=False)
    return chain, float(np.mean(np.asarray(integration)))
