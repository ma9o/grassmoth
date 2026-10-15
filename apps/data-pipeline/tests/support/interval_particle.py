"""Small native particle targets with independently tractable interval likelihoods."""

from __future__ import annotations

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.probability.families import DistributionFamily
from nof1_causal_lab.models.ssm.compile.interval_tree import compile_interval_tree
from nof1_causal_lab.models.ssm.compile.observation_partition import CompiledObservationPartition
from nof1_causal_lab.models.ssm.compile.state_space import ParallelRecurrences, StateSpace
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
from nof1_causal_lab.models.ssm.inference.targets.intervals import ParticleIntervalObservations
from nof1_causal_lab.models.ssm.inference.targets.particle import (
    ParticleContext,
    ParticleTarget,
    identity_state,
)
from tests.support.inference import make_observation_support_runtime
from tests.support.observation import observation_laws


def interval_target(
    times: np.ndarray,
    observations: np.ndarray,
    starts: np.ndarray,
    summaries: tuple[str, ...],
    *,
    families: tuple[DistributionFamily, ...] | None = None,
    covariance: np.ndarray | None = None,
    state_dim: int = 1,
) -> ParticleTarget:
    """Create an AR(1) path and uncertain observation offset with explicit window supports."""
    channels = len(summaries)
    kinds = tuple("point" if summary == "last" else "interval" for summary in summaries)
    names = tuple(f"y{i}" for i in range(channels))
    ends = np.where(np.isfinite(starts), times[:, None], np.nan)
    support = make_observation_support_runtime(
        anchor_times=times,
        manifest_names=names,
        support_kinds=kinds,
        summary_operators=summaries,
        anchor_policies=("support_end",) * channels,
        observation_windows=("1d",) * channels,
        support_start_times=starts,
        support_end_times=ends,
    )
    operator = compile_observation_operator(support)
    families = (DistributionFamily.GAUSSIAN,) * channels if families is None else families
    laws = observation_laws(families, parameters={"obs_r": 5.0, "obs_df": 5.0})
    covariance = np.eye(channels) * 0.2 if covariance is None else covariance

    def model(position: jax.Array) -> dsx.DynamicalModel:
        return dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.zeros(state_dim), covariance_matrix=jnp.eye(state_dim)
            ),
            state_evolution=dsx.LinearGaussianStateEvolution(
                jnp.eye(state_dim) * 0.7, jnp.eye(state_dim) * 0.3
            ),
            observation_model=HeterogeneousObservation(
                MeasurementParams(
                    jnp.ones((channels, state_dim)) / state_dim,
                    jnp.full(channels, position[0]),
                    jnp.asarray(covariance),
                ),
                laws,
            ),
            t0=0.0,
        )

    _, static = eqx.partition(model(jnp.zeros(1)), eqx.is_array)

    def context(position: jax.Array, grid: jax.Array) -> ParticleContext:
        return eqx.filter(model(position), eqx.is_array), grid, jnp.zeros(state_dim)

    def declared(bound: ParticleContext) -> dsx.DynamicalModel:
        return eqx.combine(bound[0], static)

    transform = ParameterTransform(
        jnp.zeros(1),
        lambda z: {"offset": z[0]},
        lambda z: {"offset": z[0]},
        lambda z: dist.Normal(0.0, 1.0).log_prob(z[0]),
    )
    intervals = (
        ParticleIntervalObservations(
            declared,
            operator,
            compile_interval_tree(operator.window_plan, summaries, observations),
            identity_state,
        )
        if operator.requires_interval_summary_handling
        else None
    )
    return ParticleTarget(
        transform,
        context,
        declared,
        jnp.asarray(observations),
        jnp.asarray(times),
        StateSpace(
            state_dim,
            (),
            tuple(range(state_dim)),
            tuple(range(state_dim)),
            (),
            ParallelRecurrences(()),
        ),
        CompiledObservationPartition(tuple(range(channels)), ()),
        intervals=intervals,
    )


def gaussian_interval_case():
    """A generated overlapping-window panel with an analytic joint posterior."""
    steps = 9
    times = np.arange(float(steps))
    windows = ((0, 2), (1, 5), (4, 8))
    rows = np.zeros((len(windows), steps))
    starts = np.full((steps, 1), np.nan)
    for i, (first, last) in enumerate(windows):
        rows[i, first : last + 1] = 1 / (last - first)
        rows[i, (first, last)] *= 0.5
        starts[last, 0] = first
    noise = np.random.default_rng(82).normal(size=steps)
    truth = np.zeros(steps)
    truth[0] = noise[0]
    for t in range(1, steps):
        truth[t] = 0.7 * truth[t - 1] + np.sqrt(0.3) * noise[t]
    values = rows @ truth + 0.8 + np.sqrt(0.2) * np.random.default_rng(93).normal(size=3)
    observations = np.full((steps, 1), np.nan)
    observations[[last for _, last in windows], 0] = values
    target = interval_target(times, observations, starts, ("mean",))
    transition = np.eye(steps)
    for t in range(1, steps):
        transition[t, :t] = 0.7 * transition[t - 1, :t]
    path_covariance = transition @ np.diag([1.0, *([0.3] * (steps - 1))]) @ transition.T
    prior = np.zeros((steps + 1, steps + 1))
    prior[0, 0] = 1.0
    prior[1:, 1:] = path_covariance
    design = np.column_stack([np.ones(3), rows])
    observation_covariance = design @ prior @ design.T + np.eye(3) * 0.2
    gain = np.linalg.solve(observation_covariance, design @ prior).T
    return target, gain @ values, prior - gain @ design @ prior


def initialize_interval_chain(target: ParticleTarget, joint: jax.Array):
    """Initialize the production joint kernel from explicit parameter and path coordinates."""
    return _initialize_chain_state(
        joint[:1],
        observations=target.observations,
        times=target.times,
        target=target,
        initial_latent_delta=jnp.full(target.times.shape, 0.6),
        param_step_size=0.3,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        param_target_accept=0.35,
        initial_latent_trajectory=joint[1:, None],
    )
