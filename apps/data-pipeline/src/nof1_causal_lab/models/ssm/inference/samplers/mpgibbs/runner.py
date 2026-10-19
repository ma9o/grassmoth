"""Run structured particle kernels with Dynestyx parameter/model interpretations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import (
    ParticleWeightMetric,
    ReplicaSwapCounts,
    TemperingDiagnostics,
)
from nof1_causal_lab.artifacts.sampling import FixedLatentStepSpec
from nof1_causal_lab.models.ssm.inference.continuation import restore_state, retain_state
from nof1_causal_lab.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    _latent_summary_from_chain_moments,
)
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.tempering import (
    ExchangeRecord,
    ReplicaExchangeKernel,
    adapt_temperature_ladder,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.adaptation import (
    DsmcAdaptation,
    adapt_scales,
    initial_delta,
    initialize_adaptation,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.diagnostics import (
    build_particle_diagnostic_flags,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.math import _masked_mean
from nof1_causal_lab.models.ssm.inference.samplers.progress import inference_phase
from nof1_causal_lab.models.ssm.inference.samplers.sweeps import (
    compile_projected_sweeps,
    run_sweeps,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.artifacts.arrays import NumericalArray
    from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.kernel import (
        MarginalParticleGibbsKernel,
    )
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.inference.warmup.initialization import ParticleInitialization


@dataclass(frozen=True, kw_only=True)
class ParticleChainResult:
    """Retained particle draws, warmup evidence, and sampler diagnostics."""

    grouped_positions: jnp.ndarray
    observation_log_probs: jnp.ndarray
    chain_extra_fields: Mapping[str, jnp.ndarray]
    warmup_chain_extra_fields: Mapping[str, jnp.ndarray]
    all_chain_extra_fields: Mapping[str, jnp.ndarray]
    starting_values: Mapping[str, jnp.ndarray]
    complete_log_posterior_history: jnp.ndarray
    warmup_complete_log_posterior_history: jnp.ndarray
    all_complete_log_posterior_history: jnp.ndarray
    latent_posterior_summary: Mapping[str, jnp.ndarray] | None
    latent_paths: jnp.ndarray | None
    initial_param_step_size: jnp.ndarray
    final_param_step_size: jnp.ndarray
    initial_latent_delta: jnp.ndarray
    final_latent_delta: jnp.ndarray
    compiled_step: Compiled
    first_step_seconds: float
    sampling_loop_seconds: float
    final_state: tuple[NumericalArray, ...]
    random_key: jax.Array
    last_parameters: jax.Array
    last_paths: jax.Array
    tempering: TemperingDiagnostics | None = None

    @property
    def post_warmup_complete_log_posterior_mean(self) -> jnp.ndarray:
        """Project the retained density history without retaining a duplicate summary."""
        return jnp.mean(self.complete_log_posterior_history, axis=1)

    def __post_init__(self) -> None:
        """Freeze retained, warmup, and combined chain diagnostics and any latent summary mapping."""
        object.__setattr__(
            self, "chain_extra_fields", MappingProxyType(dict(self.chain_extra_fields))
        )
        object.__setattr__(
            self,
            "warmup_chain_extra_fields",
            MappingProxyType(dict(self.warmup_chain_extra_fields)),
        )
        object.__setattr__(
            self, "all_chain_extra_fields", MappingProxyType(dict(self.all_chain_extra_fields))
        )
        if self.latent_posterior_summary is not None:
            object.__setattr__(
                self,
                "latent_posterior_summary",
                MappingProxyType(dict(self.latent_posterior_summary)),
            )
        freeze_fields(self)


def _initialize_chain_state(
    init_position: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    target: ParticleTarget,
    parameter_gradient_fn: Callable[[jax.Array], jax.Array],
    initial_latent_delta: jnp.ndarray,
    param_step_size: float,
    param_min_scale: float,
    param_max_scale: float,
    param_target_accept: float,
    initial_latent_trajectory: jnp.ndarray,
    exact_constraints: ExactStateConstraints | None = None,
) -> TrajectoryMCMCState:
    context = target.context(init_position, times)
    latent_trajectory = jnp.asarray(
        initial_latent_trajectory, dtype=target.initial_moments(context)[0].dtype
    )
    if exact_constraints is not None:
        latent_trajectory = exact_constraints.project(latent_trajectory)
    latent_trajectory = target.replay_path(context, latent_trajectory)
    complete_lp, trajectory_lp = target.log_posterior_from_context(
        init_position,
        context,
        latent_trajectory,
        observations,
    )
    latent_delta_value = jnp.asarray(initial_latent_delta, dtype=latent_trajectory.dtype)
    param_step_value = _clip_scale(
        jnp.asarray(param_step_size, dtype=latent_trajectory.dtype),
        min_scale=param_min_scale,
        max_scale=param_max_scale,
    )
    da_init, _da_update, _ = dual_averaging_adaptation(target=float(param_target_accept))
    return TrajectoryMCMCState(
        position=init_position,
        parameter_gradient=parameter_gradient_fn(init_position),
        latent_context=context,
        latent_trajectory=latent_trajectory,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
        latent_delta=latent_delta_value,
        param_step_size=param_step_value,
        param_da=da_init(param_step_value),
    )


class _Population(NamedTuple):
    chains: TrajectoryMCMCState
    random_key: jax.Array
    iteration: jax.Array
    betas: jax.Array
    adaptation: DsmcAdaptation
    swap_accepted: jax.Array
    swap_attempted: jax.Array
    warmup_attempts: jax.Array
    warmup_rejections: jax.Array
    ladder_frozen_at: jax.Array
    moments: tuple[jax.Array, jax.Array, jax.Array] | None


def run_marginal_particle_gibbs(
    target: ParticleTarget,
    *,
    kernel: MarginalParticleGibbsKernel,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    key: jax.Array,
    adaptation_rate: float,
    retain_latent_paths: bool,
    initialization: ParticleInitialization,
    compute_budget: ComputeBudget,
    compute_latent_posterior_summary: bool = True,
    # Both adaptation policies stop updating after warmup.
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple",
    tempering: ReplicaExchangeKernel | None = None,
    clock: Callable[[], float],
    continuation: SamplerCheckpoint | None = None,
) -> ParticleChainResult:
    """Run marginalized Particle Gibbs chains from paired LAPS starts."""
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    da_param_update = (
        dual_averaging_adaptation(target=float(kernel.target_accept))[1]
        if use_dual_averaging
        else None
    )
    observations = target.observations
    times = target.times
    num_steps = int(observations.shape[0])
    dim = int(target.initial_position.shape[0])
    chain_init_positions = jnp.asarray(
        initialization.positions, dtype=target.initial_position.dtype
    )
    if chain_init_positions.shape != (num_chains, dim):
        raise ValueError(
            "initialization.positions must have shape (num_chains, dim); got "
            f"{chain_init_positions.shape} with num_chains={num_chains} and dim={dim}."
        )
    chain_initial_latents = jnp.asarray(initialization.paths, dtype=observations.dtype)
    if chain_initial_latents.shape[0] != num_chains:
        raise ValueError(
            "initialization.paths must have leading dimension num_chains; got "
            f"{chain_initial_latents.shape[0]} with num_chains={num_chains}."
        )

    initial_latent_delta_value = initial_delta(kernel.dsmc)
    initial_latent_delta = jnp.full(
        (num_steps,),
        jnp.asarray(initial_latent_delta_value, dtype=observations.dtype),
    )
    with inference_phase("mpgibbs", "initializing_chains", clock=clock):
        states = jax.jit(
            jax.vmap(
                lambda position, path: _initialize_chain_state(
                    position,
                    observations=observations,
                    times=times,
                    target=target,
                    parameter_gradient_fn=kernel.parameter_gradient_fn,
                    initial_latent_delta=initial_latent_delta,
                    exact_constraints=kernel.exact_constraints,
                    param_step_size=kernel.initial_param_step_size,
                    param_min_scale=kernel.min_scale,
                    param_max_scale=kernel.max_scale,
                    param_target_accept=kernel.target_accept,
                    initial_latent_trajectory=path,
                )
            )
        )(chain_init_positions, chain_initial_latents)
        jax.block_until_ready(states)
    nonfinite_chains = [
        chain_idx
        for chain_idx in range(num_chains)
        if not bool(jnp.isfinite(states.complete_log_posterior[chain_idx]))
    ]
    if nonfinite_chains:
        raise ValueError(
            "Initial complete log-posterior is non-finite for chain(s) "
            f"{nonfinite_chains}. Every move would be rejected against a non-finite "
            "reference, so the sampler cannot recover from this initialization."
        )
    levels = 1 if tempering is None else tempering.spec.num_temperatures
    betas = jnp.linspace(1.0, 0.0, levels, dtype=states.position.dtype)
    cold_indices = jnp.arange(num_chains) * levels
    replicas = num_chains * levels
    states = jax.tree.map(lambda value: jnp.repeat(value, levels, axis=0), states)
    if tempering is not None:
        states = tempering.rebind(states, betas)
    swap_accepted = jnp.zeros((num_chains, levels - 1), dtype=jnp.int32)
    swap_attempted = jnp.zeros_like(swap_accepted)
    warmup_attempts = jnp.zeros_like(swap_attempted)
    warmup_rejections = jnp.zeros(swap_attempted.shape, dtype=betas.dtype)
    moments = None
    if compute_latent_posterior_summary:
        example = states.latent_trajectory[cold_indices]
        moments = (jnp.zeros_like(example), jnp.zeros_like(example), jnp.asarray(0))
    initial = _Population(
        states,
        key
        if continuation is None
        else jax.random.wrap_key_data(jnp.asarray(continuation.random_key.values)),
        jnp.asarray(0),
        betas,
        initialize_adaptation(states.latent_delta),
        swap_accepted,
        swap_attempted,
        warmup_attempts,
        warmup_rejections,
        jnp.asarray(0),
        moments,
    )
    if continuation is not None and num_warmup == 0:
        initial = restore_state(initial._replace(moments=None), continuation.native_state)._replace(
            moments=moments
        )
        scale = kernel.dsmc.latent_step
        if isinstance(scale, FixedLatentStepSpec):
            delta = jnp.full_like(initial.chains.latent_delta, scale.delta)
            initial = initial._replace(
                chains=initial.chains._replace(latent_delta=delta),
                adaptation=initial.adaptation._replace(delta=delta),
            )
    initial_param_step_size = initial.chains.param_step_size[cold_indices]
    initial_latent_delta = initial.chains.latent_delta[cold_indices]
    diagnostic_flags = build_particle_diagnostic_flags(diagnostic_metrics=kernel.diagnostic_metrics)

    def step_factory(
        batch_size: int,
    ) -> Callable[[_Population, jax.Array], tuple[_Population, dict[str, jax.Array]]]:
        def sweep(
            current: _Population, posterior: jax.Array
        ) -> tuple[_Population, dict[str, jax.Array]]:
            next_key, step_key = random.split(current.random_key)
            keys = random.split(step_key, replicas)
            if tempering is None:
                updated, info = jax.vmap(
                    lambda state, draw_key: kernel.step_fn(state, draw_key, batch_size=batch_size)
                )(current.chains, keys)
                empty = jnp.zeros((num_chains, 0), dtype=jnp.int32)
                exchange = ExchangeRecord(empty, empty, empty.astype(current.betas.dtype))
            else:
                updated, info, exchange = tempering.step(
                    current.chains, keys, current.betas, current.iteration, batch_size=batch_size
                )
            cold = jax.tree.map(lambda value: value[cold_indices], updated)
            cold_info = jax.tree.map(lambda value: value[cold_indices], info)
            record: dict[str, jax.Array] = {
                "log_posterior": cold.complete_log_posterior,
                "parameter_accept_prob": cold_info["parameter_accepted"],
                "latent_accept_prob": _masked_mean(
                    cold_info["latent_accepted"], cold_info["latent_proposed"], axis=-1
                ),
                "selected_parameter_label": cold_info["selected_label"],
                "selected_particle": cold_info["final_particle"],
                **{
                    name: cold_info[name]
                    for name in (
                        "latent_move_rms",
                        "latent_move_max_abs",
                        "latent_move_rms_per_t",
                        "latent_frozen_frac",
                        "latent_frozen_frac_by_d",
                        "final_label_log_probs",
                        "amala_grad_norm_mean",
                        "amala_grad_norm_max",
                    )
                },
            }
            if diagnostic_flags.particle_weights:
                record = {
                    **record,
                    **{
                        metric.history_key: cold_info[metric.history_key]
                        for metric in ParticleWeightMetric
                    },
                }
            if diagnostic_flags.particle_identity:
                record = {
                    **record,
                    **{
                        name: cold_info[name]
                        for name in (
                            "selected_particle_per_t",
                            "reference_path_hit_rate",
                            "selected_particle_unique_count",
                        )
                    },
                }
            if diagnostic_flags.parameter_movement:
                record = {**record, "parameter_jump_rms": cold_info["parameter_jump_rms"]}

            def retain(updated: TrajectoryMCMCState) -> _Population:
                next_moments = current.moments
                if next_moments is not None:
                    latent_sum, latent_sumsq, count = next_moments
                    next_moments = (
                        latent_sum + cold.latent_trajectory,
                        latent_sumsq + cold.latent_trajectory * cold.latent_trajectory,
                        count + 1,
                    )
                return current._replace(
                    chains=updated,
                    random_key=next_key,
                    iteration=current.iteration + 1,
                    swap_accepted=current.swap_accepted + exchange.accepted,
                    swap_attempted=current.swap_attempted + exchange.attempted,
                    moments=next_moments,
                )

            def adapt(updated: TrajectoryMCMCState) -> _Population:
                tuning = adapt_scales(
                    current.adaptation,
                    kernel.dsmc,
                    info["latent_accepted"],
                    info["latent_proposed"],
                    num_warmup,
                )
                updated = updated._replace(latent_delta=tuning.delta)
                if da_param_update is not None:
                    update_param = da_param_update
                    param_da = jax.vmap(
                        lambda da, accepted: _clip_dual_averaging_state(
                            update_param(da, accepted),
                            min_scale=kernel.min_scale,
                            max_scale=kernel.max_scale,
                        )
                    )(updated.param_da, info["parameter_accepted"])
                    log_step = jnp.where(
                        current.iteration + 1 == num_warmup,
                        param_da.log_step_size_avg,
                        param_da.log_step_size,
                    )
                    updated = updated._replace(
                        param_step_size=_clip_scale(
                            jnp.exp(log_step).astype(updated.param_step_size.dtype),
                            min_scale=kernel.min_scale,
                            max_scale=kernel.max_scale,
                        ),
                        param_da=param_da,
                    )
                else:
                    updated = updated._replace(
                        param_step_size=_adapt_scale(
                            updated.param_step_size,
                            accepted=info["parameter_accepted"],
                            target_accept=kernel.target_accept,
                            adaptation_rate=adaptation_rate,
                            min_scale=kernel.min_scale,
                            max_scale=kernel.max_scale,
                        )
                    )
                next_state: _Population = current._replace(
                    chains=updated,
                    random_key=next_key,
                    iteration=current.iteration + 1,
                    adaptation=tuning,
                )
                if tempering is not None:
                    exchange_kernel = tempering
                    next_state = next_state._replace(
                        warmup_attempts=current.warmup_attempts + exchange.attempted,
                        warmup_rejections=current.warmup_rejections
                        + exchange.rejection_probability,
                    )

                    def adapt_ladder(population: _Population) -> _Population:
                        ladder = adapt_temperature_ladder(
                            population.betas,
                            population.warmup_rejections,
                            population.warmup_attempts,
                        )
                        return population._replace(
                            chains=exchange_kernel.rebind(population.chains, ladder),
                            betas=ladder,
                            warmup_attempts=jnp.zeros_like(population.warmup_attempts),
                            warmup_rejections=jnp.zeros_like(population.warmup_rejections),
                            ladder_frozen_at=population.iteration,
                        )

                    adapted: _Population = jax.lax.cond(
                        (next_state.iteration % tempering.spec.adaptation_interval == 0)
                        & (next_state.iteration <= num_warmup // 2),
                        adapt_ladder,
                        lambda population: population,
                        next_state,
                    )
                    next_state = adapted
                return next_state

            next_population: _Population = jax.lax.cond(posterior, retain, adapt, updated)
            return next_population, {
                **record,
                "positions": cold.position,
                "paths": cold.latent_trajectory,
            }

        return sweep

    with inference_phase("mpgibbs", "compiling_sweeps", clock=clock):
        plan = compile_projected_sweeps(
            step_factory,
            initial,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_batch_size=max(kernel.num_particles, (1 << (num_steps - 1).bit_length()) // 2),
            budget=compute_budget,
            target=target,
            path_dtype=states.latent_trajectory.dtype,
            num_chains=num_chains,
            clock=clock,
            label="mpgibbs",
        )
    run = run_sweeps(plan, initial, clock=clock, label="mpgibbs")
    final = run.state
    draw_fields = frozenset(("positions", "paths", "observation_log_probs", "log_posterior"))
    warmup_extra = {
        name: value for name, value in run.history.warmup.items() if name not in draw_fields
    }
    posterior_extra = {
        name: value for name, value in run.history.posterior.items() if name not in draw_fields
    }
    all_extra = {
        name: jnp.concatenate((warmup_extra[name], value), axis=1)
        for name, value in posterior_extra.items()
    }
    latent_summary = None
    if final.moments is not None:
        latent_sum, latent_sumsq, count = final.moments
        denom = jnp.maximum(count, 1).astype(latent_sum.dtype)
        mean = latent_sum / denom
        variance = jnp.maximum(latent_sumsq / denom - mean * mean, 0.0)
        latent_summary = _latent_summary_from_chain_moments(mean, jnp.sqrt(variance))
    complete_lp = run.history.posterior["log_posterior"]
    return ParticleChainResult(
        starting_values={
            "positions": initial.chains.position[cold_indices],
            "paths": initial.chains.latent_trajectory[cold_indices],
            "log_joint": initial.chains.complete_log_posterior[cold_indices],
        },
        grouped_positions=run.history.posterior["positions"],
        observation_log_probs=run.history.posterior["observation_log_probs"],
        chain_extra_fields=posterior_extra,
        warmup_chain_extra_fields=warmup_extra,
        all_chain_extra_fields=all_extra,
        complete_log_posterior_history=complete_lp,
        warmup_complete_log_posterior_history=run.history.warmup["log_posterior"],
        all_complete_log_posterior_history=jnp.concatenate(
            (run.history.warmup["log_posterior"], complete_lp), axis=1
        ),
        latent_posterior_summary=latent_summary,
        latent_paths=run.history.posterior["paths"] if retain_latent_paths else None,
        initial_param_step_size=initial_param_step_size,
        final_param_step_size=final.chains.param_step_size[cold_indices],
        initial_latent_delta=initial_latent_delta,
        final_latent_delta=final.chains.latent_delta[cold_indices],
        compiled_step=plan.compiled_step,
        first_step_seconds=run.first_step_seconds,
        sampling_loop_seconds=run.sampling_loop_seconds,
        final_state=retain_state(final._replace(moments=None)),
        random_key=final.random_key,
        last_parameters=final.chains.position[cold_indices],
        last_paths=final.chains.latent_trajectory[cold_indices],
        tempering=None
        if tempering is None
        else TemperingDiagnostics(
            inverse_temperatures=tuple(float(beta) for beta in jax.device_get(final.betas)),
            ladder_frozen_at_warmup_step=int(final.ladder_frozen_at),
            swaps=tuple(
                tuple(
                    ReplicaSwapCounts(accepted=int(a), attempted=int(n))
                    for a, n in zip(accepted, attempted, strict=True)
                )
                for accepted, attempted in zip(
                    jax.device_get(final.swap_accepted),
                    jax.device_get(final.swap_attempted),
                    strict=True,
                )
            ),
        ),
    )
