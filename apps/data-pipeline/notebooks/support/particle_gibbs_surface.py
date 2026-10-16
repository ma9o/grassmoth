"""Toy probability laws and adapters to the production inference engines.

The two-coordinate target has one parameter and one latent state at a single time.
Production mPGibbs, DSMC, LAPS initialization and NUTS execute that declared target;
this module only constructs inputs and projects their returned data for plotting.
Analytic Gibbs and grid quadrature are explicitly separate mathematical references.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Literal

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpy.typing import NDArray

from nof1_causal_lab.artifacts.sampling import (
    DsmcSpec,
    LapsInitializationSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    SamplerSpec,
)
from nof1_causal_lab.compute.budget import ComputeBudget, SharedMemoryBudget
from nof1_causal_lab.models.ssm.compile.observation_partition import CompiledObservationPartition
from nof1_causal_lab.models.ssm.compile.state_space import ParallelRecurrences, StateSpace
from nof1_causal_lab.models.ssm.inference.joint_coordinates import build_joint_coordinates
from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.inference.samplers.nuts import NutsDARun, run_nuts_da
from nof1_causal_lab.models.ssm.inference.samplers.particle.context import (
    build_smoother_context,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.contract import SmootherSetup
from nof1_causal_lab.models.ssm.inference.samplers.particle.dsmc import step as dsmc_step
from nof1_causal_lab.models.ssm.inference.samplers.particle.dsmc import (
    trace_step as dsmc_trace_step,
)
from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from nof1_causal_lab.models.ssm.inference.warmup.laps import (
    LapsInitializationResult,
    prepare_laps_initialization,
)
from nof1_causal_lab.models.ssm.inference.warmup.laps_ensemble import LapsTraceRecord
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCState
    from nof1_causal_lab.models.ssm.inference.tracing import (
        CategoricalTraceRecord,
        SmootherTraceRecord,
    )

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
WorldKind = Literal["ridge", "bounded_scale"]
ParameterProposal = Literal["random_walk", "pseudo_langevin"]
_LOG_2PI = float(np.log(2.0 * np.pi))
_DEFAULTS = MarginalParticleGibbsSpec()
_BUDGET = ComputeBudget(memory=SharedMemoryBudget(available_bytes=512 * 1024**2))


def _normal_logpdf(value: FloatArray | float, mean: FloatArray | float, var: FloatArray | float):
    return -0.5 * (_LOG_2PI + np.log(var) + np.square(np.subtract(value, mean)) / var)


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

    def x_given_theta(self, theta: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        var = 1.0 / (1.0 / self.tau**2 + 1.0 / self.sigma**2)
        mean = var * (np.divide(theta, self.tau**2) + self.y / self.sigma**2)
        return np.asarray(mean), np.full(np.shape(mean), var)

    def theta_given_x(self, x: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        """Exact parameter conditional: what arm G draws."""
        var = 1.0 / (1.0 / self.prior_sd**2 + 1.0 / self.tau**2)
        mean = var * np.divide(x, self.tau**2)
        return np.asarray(mean), np.full(np.shape(mean), var)

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


@dataclass(frozen=True)
class BoundedScale:
    """``theta`` controls a latent standard deviation between 1/2 and 2.

    With ``y = 0`` the posterior is symmetric in ``x``: the linear correlation of
    ``theta`` and ``x`` is zero, yet their dependence is nonlinear. The smooth
    bounded scale removes the funnel's arbitrarily narrow neck.
    """

    sigma: float
    y: float = 0.0
    prior_mean: float = -0.5
    prior_sd: float = 1.0

    @property
    def kind(self) -> WorldKind:
        return "bounded_scale"

    def latent_variance(self, theta: jax.Array) -> jax.Array:
        """The shared scale law for production inference and display evaluations."""
        return jnp.exp(np.log(4.0) * jnp.tanh(theta))

    def parameter_grid(self) -> FloatArray:
        """Finite reference quadrature covering six prior standard deviations."""
        return np.linspace(
            self.prior_mean - 6 * self.prior_sd, self.prior_mean + 6 * self.prior_sd, 4001
        )

    def log_prior(self, theta: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(theta, self.prior_mean, self.prior_sd**2))

    def log_latent(self, x: FloatArray | float, theta: FloatArray | float) -> FloatArray:
        variance = np.asarray(self.latent_variance(jnp.asarray(theta)))
        return np.asarray(_normal_logpdf(x, 0.0, variance))

    def log_obs(self, x: FloatArray | float) -> FloatArray:
        return np.asarray(_normal_logpdf(self.y, x, self.sigma**2))

    def x_given_theta(self, theta: FloatArray | float) -> tuple[FloatArray, FloatArray]:
        variance = np.asarray(self.latent_variance(jnp.asarray(theta)))
        var = 1.0 / (1.0 / variance + 1.0 / self.sigma**2)
        return np.asarray(var * self.y / self.sigma**2), np.asarray(var)

    def draw_theta(self, x: float, rng: np.random.Generator) -> float:
        grid = self.parameter_grid()
        log_density = self.log_prior(grid) + self.log_latent(x, grid)
        density = np.exp(log_density - log_density.max())
        cdf = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))
        cdf = np.concatenate(([0.0], cdf / cdf[-1]))
        return float(np.interp(rng.uniform(), cdf, grid))

    def log_evidence(self, theta: FloatArray | float) -> FloatArray:
        variance = np.asarray(self.latent_variance(jnp.asarray(theta)))
        return np.asarray(_normal_logpdf(self.y, 0.0, variance + self.sigma**2))

    def theta_box(self) -> tuple[float, float]:
        low, high = np.interp([0.002, 0.998], _marginal_cdf(self), self.parameter_grid())
        return float(low) - 0.4, float(high) + 0.4

    def x_box(self) -> tuple[float, float]:
        _, var = self.x_given_theta(self.theta_box()[1])
        half = 4.2 * float(np.sqrt(var))
        return self.y - half, self.y + half


type World = Ridge | BoundedScale


def log_joint(world: World, theta: FloatArray | float, x: FloatArray | float) -> FloatArray:
    """``log p(theta) + log p(x | theta) + log p(y | x)``."""
    return world.log_prior(theta) + world.log_latent(x, theta) + world.log_obs(x)


@dataclass(frozen=True)
class Surface:
    """Exact joint and marginal density evaluations, normalized over a finite display grid."""

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


def _marginal_cdf(world: BoundedScale) -> FloatArray:
    grid = world.parameter_grid()
    log_marginal = world.log_prior(grid) + world.log_evidence(grid)
    density = np.exp(log_marginal - log_marginal.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]))))
    return cdf / cdf[-1]


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
    """Analytic ridge moments; deterministic quadrature for the bounded-scale model."""
    if isinstance(world, Ridge):
        mean, cov = world.moments()
        conditional = float(world.theta_given_x(0.0)[1])
        return Coupling(float(mean[0]), float(cov[0, 0]), conditional)
    if isinstance(world, BoundedScale):
        theta = world.parameter_grid()[::4]
        x = np.linspace(*world.x_box(), 401)
        log_density = log_joint(world, theta[None, :], x[:, None])
        density = np.exp(log_density - log_density.max())
        marginal_x = np.trapezoid(density, theta, axis=1)
        mass = np.trapezoid(marginal_x, x)
        marginal_theta = np.trapezoid(density, x, axis=0) / mass
        mean = float(np.trapezoid(theta * marginal_theta, theta))
        var = float(np.trapezoid(np.square(theta - mean) * marginal_theta, theta))
        centers = np.trapezoid(density * theta, theta, axis=1) / marginal_x
        spreads = np.trapezoid(density * np.square(theta - centers[:, None]), theta, axis=1)
        return Coupling(mean, var, float(np.trapezoid(spreads, x) / mass))
    raise TypeError(world)


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
    aux: float  # auxiliary point drawn around the drifted reference
    center: float  # aMALA+ proposes around the auxiliary point
    kappa: float  # gradient multiplier used by this sweep
    particles: FloatArray  # index 0 is the reference
    weights: FloatArray  # selection probabilities
    chosen: int
    label_probs: FloatArray  # label probabilities given the chosen latent
    label: int
    parameter_aux: float | None  # no parameter proposal in a fixed-theta smoothing sweep


def gibbs(
    world: World, sweeps: int, start: tuple[float, float], seed: int, *, warmup: int = 0
) -> Chain:
    """Conditional-draw reference: analytic ridge or bounded-scale grid quadrature."""
    rng = np.random.default_rng(seed)
    theta, x = start
    total = sweeps + warmup
    thetas, xs = np.empty(total), np.empty(total)
    for sweep in range(total):
        theta = world.draw_theta(x, rng)
        mean, var = world.x_given_theta(theta)
        x = float(mean + np.sqrt(var) * rng.standard_normal())
        thetas[sweep], xs[sweep] = theta, x
    moved = np.ones(total, dtype=bool)
    return _project_chain((thetas, xs, moved, moved), warmup, blocked=True)


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


@lru_cache(maxsize=8)
def production_target(world: World) -> ParticleTarget:
    """Declare the one-time-point toy through the same target contract used by fitting."""
    initial = jnp.asarray([0.0 if isinstance(world, Ridge) else world.prior_mean])

    def _model(position: jax.Array) -> dsx.DynamicalModel:
        mean = position if isinstance(world, Ridge) else jnp.zeros(1)
        variance = world.tau**2 if isinstance(world, Ridge) else world.latent_variance(position[0])
        return dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                mean, covariance_matrix=jnp.full((1, 1), variance)
            ),
            # T=1 has no transition factors. A declared transition is still part of
            # the model contract and of shape tracing inside the production kernel.
            state_evolution=dsx.LinearGaussianStateEvolution(jnp.zeros((1, 1)), jnp.ones((1, 1))),
            observation_model=dsx.LinearGaussianObservation(
                jnp.ones((1, 1)), jnp.full((1, 1), world.sigma**2)
            ),
            t0=0.0,
        )

    _, static = eqx.partition(_model(initial), eqx.is_array)

    def _context(position: jax.Array, grid: jax.Array) -> ParticleContext:
        return eqx.filter(_model(position), eqx.is_array), grid, jnp.zeros(1)

    def _declared(context: ParticleContext) -> dsx.DynamicalModel:
        return eqx.combine(context[0], static)

    prior_mean = 0.0 if isinstance(world, Ridge) else world.prior_mean
    return ParticleTarget(
        parameters=ParameterTransform(
            initial,
            lambda position: {"theta": position[0]},
            lambda position: {"theta": position[0]},
            lambda position: dist.Normal(prior_mean, world.prior_sd).log_prob(position[0]),
        ),
        context=_context,
        dynamical_model=_declared,
        observations=jnp.asarray([[world.y]]),
        times=jnp.asarray([0.0]),
        state_space=StateSpace(1, (), (0,), (0,), (), ParallelRecurrences(())),
        observation_partition=CompiledObservationPartition((0,), ()),
    )


def _initial_state(
    target: ParticleTarget,
    start: tuple[float, float],
    delta: float,
    step: float,
    parameter_gradient_fn: Callable[[jax.Array], jax.Array],
) -> TrajectoryMCMCState:
    return _initialize_chain_state(
        jnp.asarray([start[0]]),
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=parameter_gradient_fn,
        initial_latent_delta=jnp.asarray([delta]),
        param_step_size=step,
        param_min_scale=_DEFAULTS.param_step_size_min,
        param_max_scale=_DEFAULTS.param_step_size_max,
        param_target_accept=_DEFAULTS.param_target_accept,
        initial_latent_trajectory=jnp.asarray([[start[1]]]),
    )


@lru_cache(maxsize=16)
def _marginal_kernel(
    world: World,
    particles: int,
    kappa: float,
    scale: float,
    proposal: ParameterProposal,
    anchor_x: float,
):
    return build_marginal_particle_gibbs_kernel(
        production_target(world),
        num_particles=particles,
        num_parameter_particles=2,
        param_step_size=_DEFAULTS.param_step_size,
        parameter_preconditioner_chol=jnp.asarray([[scale]]),
        parameter_proposal=proposal,
        parameter_reference_path=jnp.asarray([[anchor_x]]),
        dsmc=DsmcSpec(drift="amala_plus" if kappa else "none"),
    )


@partial(jax.jit, static_argnames=("step_fn", "sweeps"))
def _chain_program(state, key, *, step_fn, sweeps: int):
    def _step(current, step_key):
        updated, info = step_fn(current, step_key)
        return updated, (
            updated.position[0],
            updated.latent_trajectory[0, 0],
            info["parameter_accepted"].astype(bool),
            jnp.any(updated.latent_trajectory != current.latent_trajectory),
        )

    return jax.lax.scan(_step, state, jax.random.split(key, sweeps))


def _project_chain(values, discarded: int, *, blocked: bool) -> Chain:
    theta, latent, theta_moved, x_moved = values
    return Chain(
        np.asarray(theta[discarded:], dtype=np.float64),
        np.asarray(latent[discarded:], dtype=np.float64),
        np.asarray(theta_moved[discarded:], dtype=bool),
        np.asarray(x_moved[discarded:], dtype=bool),
        blocked=blocked,
    )


def mpgibbs(
    world: World,
    sweeps: int,
    start: tuple[float, float],
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float = 0.0,
    step: float,
    scale: float = 1.0,
    proposal: ParameterProposal,
    anchor_x: float = 0.0,
    warmup: int = 0,
) -> Chain:
    """Run the production joint transition with fixed tuning; discard ``warmup`` moves."""
    kernel = _marginal_kernel(world, particles, kappa, scale, proposal, anchor_x)
    initial = _initial_state(
        production_target(world), start, delta, step, kernel.parameter_gradient_fn
    )
    _, values = _chain_program(
        initial, jax.random.key(seed), step_fn=kernel.step_fn, sweeps=sweeps + warmup
    )
    return _project_chain(values, warmup, blocked=False)


def mpgibbs_sweep(
    world: World,
    theta: float,
    x_ref: float,
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float = 0.0,
    step: float,
    scale: float = 1.0,
    proposal: ParameterProposal,
    anchor_x: float = 0.0,
) -> Sweep:
    """Project one recorded production transition; all displayed weights were used by it."""
    kernel = _marginal_kernel(world, particles, kappa, scale, proposal, anchor_x)
    initial = _initial_state(
        production_target(world), (theta, x_ref), delta, step, kernel.parameter_gradient_fn
    )
    traced = _recorded_step(initial, jax.random.key(seed), step_fn=kernel.trace_step_fn)
    trace = traced.trace
    return _project_sweep(
        trace.smoother,
        trace.parameters.candidates,
        trace.parameters.label_log_prior,
        trace.label_selection,
        parameter_aux=float(trace.parameters.auxiliary[0]),
        kappa=kappa,
    )


def _project_sweep(
    smoother: SmootherTraceRecord,
    parameters: jax.Array,
    label_log_prior: jax.Array,
    label_selection: CategoricalTraceRecord,
    *,
    parameter_aux: float | None,
    kappa: float,
) -> Sweep:
    """Select the toy's single time and coordinate from the production trace axes."""
    return Sweep(
        thetas=np.asarray(parameters[:, 0], dtype=np.float64),
        label_log_prior=np.asarray(label_log_prior, dtype=np.float64),
        aux=float(smoother.auxiliary[0, 0]),
        center=float(smoother.proposal_centers[0, 0]),
        kappa=kappa,
        particles=np.asarray(smoother.particles[0, :, 0], dtype=np.float64),
        weights=np.asarray(jnp.exp(smoother.root_selection.log_weights), dtype=np.float64),
        chosen=int(smoother.selected_particles[0]),
        label_probs=np.asarray(jnp.exp(label_selection.log_weights), dtype=np.float64),
        label=int(label_selection.selected),
        parameter_aux=parameter_aux,
    )


@partial(jax.jit, static_argnames=("step_fn",))
def _recorded_step(state, key, *, step_fn):
    return step_fn(state, key)


def _smoother_static(target: ParticleTarget, particles: int, kappa: float) -> SmootherSetup:
    return SmootherSetup(
        latent_context_runtime_fn=target.context,
        log_prior_unc_fn=target.log_prior,
        obs_increment_fn=target.observation_increment,
        trajectory_log_prob_fn=target.path_log_prob,
        runtime_observations=target.observations,
        runtime_times=target.times,
        latent_free_mask=target.independent_mask,
        num_particles=particles,
        num_parameter_particles=1,
        drift="amala_plus" if kappa else "none",
        latent_block_coords="all",
        paid_mix=None,
        transition_initial_log_prob_fn=target.initial_log_prob,
        transition_log_prob_fn=target.transition_log_prob,
        transition_pairwise_log_probs_fn=target.pairwise_transition_log_prob,
        diagnostic_metrics=frozenset(),
        intervals=None,
    )


def dsmc_sweep(
    world: World,
    theta: float,
    x_ref: float,
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float = 0.0,
) -> Sweep:
    """Record the production conditional smoother at one fixed parameter value."""
    from nof1_causal_lab.models.ssm.inference.tracing import CategoricalTraceRecord

    target = production_target(world)
    initial = _initial_state(
        target, (theta, x_ref), delta, _DEFAULTS.param_step_size, jnp.zeros_like
    )
    candidates = initial.position[None]
    ctx = build_smoother_context(
        _smoother_static(target, particles, kappa),
        candidates,
        jnp.zeros(1),
        reference=initial.latent_trajectory,
        delta=initial.latent_delta,
        trajectory_dtype=initial.trajectory_log_prob.dtype,
    )
    traced = dsmc_trace_step(
        ctx, jax.random.key(seed), initial.latent_trajectory, lambda _: jnp.zeros(1)
    )
    return _project_sweep(
        traced.trace,
        candidates,
        ctx.initial_label_log_probs,
        CategoricalTraceRecord(traced.result.label_scores.log_probs, jnp.asarray(0)),
        parameter_aux=None,
        kappa=kappa,
    )


@lru_cache(maxsize=8)
def _conditional_kernel(world: World, particles: int, kappa: float):
    """Compose a mathematical parameter reference with the production conditional smoother."""
    target = production_target(world)
    static = _smoother_static(target, particles, kappa)

    def _draw_parameter(x, key):
        if isinstance(world, Ridge):
            mean_per_x, variance = world.theta_given_x(1.0)
            return float(mean_per_x) * x + np.sqrt(float(variance)) * jax.random.normal(key)
        grid = jnp.asarray(world.parameter_grid())
        log_density = jax.vmap(
            lambda theta: target.log_posterior(
                theta[None], x.reshape(1, 1), target.observations, target.times
            )
        )(grid)
        density = jnp.exp(log_density - jnp.max(log_density))
        cdf = jnp.concatenate((jnp.zeros(1), jnp.cumsum((density[1:] + density[:-1]) / 2)))
        return jnp.interp(jax.random.uniform(key), cdf / cdf[-1], grid)

    def _step(state, key):
        parameter_key, path_key = jax.random.split(key)
        position = _draw_parameter(state.latent_trajectory[0, 0], parameter_key)[None]
        context = target.context(position, target.times)
        bound = state._replace(position=position, latent_context=context)
        smoother_context = build_smoother_context(
            static,
            position[None],
            jnp.zeros(1),
            reference=bound.latent_trajectory,
            delta=bound.latent_delta,
            trajectory_dtype=bound.trajectory_log_prob.dtype,
        )
        result = dsmc_step(
            smoother_context, path_key, state.latent_trajectory, lambda _: jnp.zeros(1)
        )
        path = result.latent_path
        return bound._replace(
            latent_trajectory=path,
            trajectory_log_prob=result.label_scores.trajectory_log_probs[0],
            complete_log_posterior=result.label_scores.log_posteriors[0],
        ), {"parameter_accepted": jnp.any(position != state.position)}

    return _step


def particle_gibbs(
    world: World,
    sweeps: int,
    start: tuple[float, float],
    seed: int,
    *,
    particles: int,
    delta: float,
    kappa: float = 0.0,
    warmup: int = 0,
) -> Chain:
    """Reference parameter conditional plus real DSMC; frozen tuning, discarded burn-in.

    Ridge parameters are drawn analytically. Bounded-scale parameters use numerical
    inverse-CDF quadrature, so that comparison is not an exact continuous kernel.
    """
    initial = _initial_state(
        production_target(world), start, delta, _DEFAULTS.param_step_size, jnp.zeros_like
    )
    _, values = _chain_program(
        initial,
        jax.random.key(seed),
        step_fn=_conditional_kernel(world, particles, kappa),
        sweeps=sweeps + warmup,
    )
    return _project_chain(values, warmup, blocked=True)


@dataclass(frozen=True)
class LapsRun:
    """Recorded production initialization, decoded into the displayed scientific coordinates."""

    positions: FloatArray
    step_size: FloatArray
    boundary: int
    acceptance: FloatArray
    starts: FloatArray
    parameter_chol: float


def run_laps(
    world: World,
    seed: int,
    *,
    members: int = 64,
    unadjusted: int = 1000,
    adjusted: int = 50,
    steps_per_sample: int = 15,
    chains: int = 4,
) -> LapsRun | InitializationFailure:
    """Run the real LAPS initializer; all history is discarded warmup, never posterior draws."""
    target = production_target(world)
    result = prepare_laps_initialization(
        target,
        spec=LapsInitializationSpec(
            ensemble_size=members,
            num_unadjusted_steps=unadjusted,
            num_adjusted_steps=adjusted,
            steps_per_sample=steps_per_sample,
        ),
        key=jax.random.key(seed),
        num_chains=chains,
        compute_budget=_BUDGET,
        record_trace=True,
    )
    if isinstance(result, InitializationFailure):
        return result
    assert isinstance(result, LapsInitializationResult)
    trace = result.trace
    assert isinstance(trace, LapsTraceRecord)
    coordinates = build_joint_coordinates(target)
    raw = jnp.concatenate((trace.unadjusted.positions, trace.adjusted.positions))
    theta, paths = jax.jit(jax.vmap(jax.vmap(coordinates.decode)))(raw)
    positions = jnp.stack((theta[..., 0], paths[..., 0, 0]), axis=-1)
    count = trace.unadjusted.positions.shape[0]
    return LapsRun(
        positions=np.asarray(positions, dtype=np.float64),
        step_size=np.asarray(
            jnp.concatenate((trace.unadjusted.step_sizes, trace.adjusted.step_sizes)),
            dtype=np.float64,
        ),
        boundary=count,
        acceptance=np.asarray(trace.adjusted_acceptance, dtype=np.float64),
        starts=np.asarray(
            jnp.stack((result.positions[:, 0], result.paths[:, 0, 0]), axis=-1), dtype=np.float64
        ),
        parameter_chol=float(result.parameter_chol[0, 0]),
    )


@dataclass(frozen=True)
class CouplingRow:
    """A single-chain ESS estimate for one deliberately small comparison run."""

    sampler: Literal["gibbs", "dsmc", "mpgibbs"]
    share: float
    theta_ess_per_sweep: float


def coupling_sweep(
    *,
    sigma: float,
    taus: FloatArray,
    particles: int,
    delta: float,
    step: float,
    proposal: ParameterProposal,
) -> tuple[CouplingRow, ...]:
    """Compare fixed-tuning toy chains; ESS per sweep does not measure computational cost."""
    rows = []
    for index, tau in enumerate(taus):
        world = Ridge(tau=float(tau), sigma=sigma)
        mean, _ = world.moments()
        start = float(mean[0]), float(mean[1])
        seed = 300 + index
        draws, discarded = 800, 200
        chains = (
            ("gibbs", gibbs(world, draws, start, seed=seed, warmup=discarded)),
            (
                "dsmc",
                particle_gibbs(
                    world,
                    draws,
                    start,
                    seed=seed,
                    particles=particles,
                    delta=delta,
                    warmup=discarded,
                ),
            ),
            (
                "mpgibbs",
                mpgibbs(
                    world,
                    draws,
                    start,
                    seed=seed,
                    particles=particles,
                    delta=delta,
                    step=step,
                    proposal=proposal,
                    warmup=discarded,
                ),
            ),
        )
        share = coupling(world).share
        rows.extend(CouplingRow(name, share, ess(chain.theta) / draws) for name, chain in chains)
    return tuple(rows)


@dataclass(frozen=True)
class NutsRun:
    """Production NUTS draws and the diagnostics needed to interpret their movement."""

    chain: Chain
    integration_steps: FloatArray
    divergences: BoolArray


def run_nuts(
    world: World,
    seed: int,
    *,
    warmup: int = 500,
    draws: int = 1000,
) -> NutsRun | InitializationFailure:
    """Run the production NumPyro NUTS adapter on the same exact toy target."""
    result = run_nuts_da(
        production_target(world),
        sampler=SamplerSpec(
            algorithm=NutsDASpec(),
            num_warmup=warmup,
            num_samples_per_chain=draws,
            num_chains=1,
            retain_latent_paths=True,
        ),
        compute_budget=_BUDGET,
        key=jax.random.key(seed),
        clock=time.monotonic,
    )
    if isinstance(result, InitializationFailure):
        return result
    assert isinstance(result, NutsDARun)
    assert result.paths is not None
    theta = np.asarray(result.positions[0, :, 0], dtype=np.float64)
    latent = np.asarray(result.paths[0, :, 0, 0], dtype=np.float64)
    moved = np.concatenate(([True], (np.diff(theta) != 0) | (np.diff(latent) != 0)))
    extra = result.phase_extra_fields["post_warmup"]
    return NutsRun(
        chain=Chain(theta, latent, moved, moved, blocked=False),
        integration_steps=np.asarray(extra["num_steps"][0], dtype=np.float64),
        divergences=np.asarray(extra["diverging"][0], dtype=bool),
    )
