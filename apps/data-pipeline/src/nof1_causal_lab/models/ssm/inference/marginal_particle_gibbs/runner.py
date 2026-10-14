"""Run structured particle kernels with Dynestyx parameter/model interpretations."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import (
    ReplicaSwapCounts,
    TemperingDiagnostics,
)
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs._math import _masked_mean
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
)
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.tempering import (
    ExchangeRecord,
    ReplicaExchangeKernel,
    adapt_temperature_ladder,
)
from nof1_causal_lab.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    _latent_summary_from_chain_moments,
    _stack_chain_states,
    _stack_sample_history,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.kernel import (
        MarginalParticleGibbsKernel,
    )
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.inference.warmup.laps import LapsInitializationResult


@dataclass(frozen=True, kw_only=True)
class ParticleChainResult:
    """Retained particle draws, warmup evidence, and sampler diagnostics."""

    grouped_positions: jnp.ndarray
    observation_log_probs: jnp.ndarray
    chain_extra_fields: Mapping[str, jnp.ndarray]
    warmup_chain_extra_fields: Mapping[str, jnp.ndarray]
    all_chain_extra_fields: Mapping[str, jnp.ndarray]
    complete_log_posterior_history: jnp.ndarray
    warmup_complete_log_posterior_history: jnp.ndarray
    all_complete_log_posterior_history: jnp.ndarray
    latent_posterior_summary: Mapping[str, jnp.ndarray] | None
    latent_paths: jnp.ndarray | None
    warmup_latent_paths: jnp.ndarray | None
    all_latent_paths: jnp.ndarray | None
    initial_param_step_size: jnp.ndarray
    final_param_step_size: jnp.ndarray
    initial_latent_delta: jnp.ndarray
    final_latent_delta: jnp.ndarray
    compiled_step: Compiled
    first_step_seconds: float
    sampling_loop_seconds: float
    post_warmup_complete_log_posterior_mean: jnp.ndarray
    tempering: TemperingDiagnostics | None = None

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
        latent_context=context,
        latent_trajectory=latent_trajectory,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
        latent_delta=latent_delta_value,
        param_step_size=param_step_value,
        param_da=da_init(param_step_value),
    )


@functools.partial(jax.jit, static_argnames=("step_fn",))
def _run_batched_step(
    states: TrajectoryMCMCState,
    step_keys: jnp.ndarray,
    *,
    step_fn: Callable[
        [TrajectoryMCMCState, jnp.ndarray], tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]
    ],
) -> tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]:
    return jax.vmap(lambda state, key: step_fn(state, key))(states, step_keys)


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
    initialization: LapsInitializationResult,
    compute_latent_posterior_summary: bool = True,
    # Both adaptation policies stop updating after warmup.
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple",
    tempering: ReplicaExchangeKernel | None = None,
    clock: Callable[[], float],
) -> ParticleChainResult:
    """Run marginalized Particle Gibbs chains from paired LAPS starts."""
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    da_param_update = (
        dual_averaging_adaptation(target=float(kernel.target_accept))[1]
        if use_dual_averaging
        else None
    )
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("marginal_particle_gibbs requires at least one MCMC step.")
    observations = target.observations
    times = target.times
    latent_active = (
        jnp.any(target.independent_mask, axis=-1)
        if kernel.exact_constraints is None
        else jnp.any(kernel.exact_constraints.free_mask & target.independent_mask, axis=-1)
    )
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

    initial_latent_delta_value = kernel.amala_delta_init
    initial_latent_delta = jnp.full(
        (num_steps,),
        jnp.asarray(initial_latent_delta_value, dtype=observations.dtype),
    )
    states = _stack_chain_states(
        [
            _initialize_chain_state(
                chain_init_positions[chain_idx],
                observations=observations,
                times=times,
                target=target,
                initial_latent_delta=initial_latent_delta,
                exact_constraints=kernel.exact_constraints,
                param_step_size=kernel.initial_param_step_size,
                param_min_scale=kernel.min_scale,
                param_max_scale=kernel.max_scale,
                param_target_accept=kernel.target_accept,
                initial_latent_trajectory=chain_initial_latents[chain_idx],
            )
            for chain_idx in range(num_chains)
        ]
    )
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
    initial_param_step_size = states.param_step_size
    initial_latent_delta = states.latent_delta
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
    ladder_frozen_at = 0
    rebind = None if tempering is None else jax.jit(tempering.rebind)
    adapt_ladder = jax.jit(adapt_temperature_ladder)
    latent_acceptance_window = jnp.zeros(
        (
            replicas,
            int(kernel.amala_adaptation_window),
            num_steps,
        ),
        dtype=states.latent_delta.dtype,
    )
    latent_proposal_window = jnp.zeros_like(latent_acceptance_window)
    step_keys = random.split(key, (total_steps, replicas))
    need_public_latent = compute_latent_posterior_summary or retain_latent_paths
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        diagnostic_metrics=kernel.diagnostic_metrics,
    )
    latent_moments: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
    if compute_latent_posterior_summary:
        public_example = states.latent_trajectory[cold_indices]
        latent_moments = (
            jnp.zeros_like(public_example),
            jnp.zeros_like(public_example),
            jnp.asarray(0, dtype=jnp.int32),
        )

    score_observations = jax.jit(
        jax.vmap(lambda context, path: target.observation_log_probs(context, path, observations))
    )
    observation_log_prob_history: list[jnp.ndarray] = []
    position_history: list[jnp.ndarray] = []
    parameter_accept_history: list[jnp.ndarray] = []
    latent_accept_history: list[jnp.ndarray] = []
    complete_lp_history: list[jnp.ndarray] = []
    latent_paths_history: list[jnp.ndarray] = []
    selected_label_history: list[jnp.ndarray] = []
    final_particle_history: list[jnp.ndarray] = []
    selected_particle_per_t_history: list[jnp.ndarray] = []
    reference_path_hit_rate_history: list[jnp.ndarray] = []
    selected_particle_unique_count_history: list[jnp.ndarray] = []
    latent_move_rms_history: list[jnp.ndarray] = []
    latent_move_max_abs_history: list[jnp.ndarray] = []
    latent_move_rms_per_t_history: list[jnp.ndarray] = []
    latent_frozen_frac_history: list[jnp.ndarray] = []
    latent_frozen_frac_by_d_history: list[jnp.ndarray] = []
    parameter_jump_rms_history: list[jnp.ndarray] = []
    final_label_log_probs_history: list[jnp.ndarray] = []
    amala_grad_norm_mean_history: list[jnp.ndarray] = []
    amala_grad_norm_max_history: list[jnp.ndarray] = []

    sampling_loop_started = clock()
    first_step_seconds = 0.0

    def population_step(
        current: TrajectoryMCMCState, keys: jax.Array, ladder: jax.Array, index: jax.Array
    ) -> tuple[TrajectoryMCMCState, dict[str, jax.Array], ExchangeRecord]:
        if tempering is not None:
            return tempering.step(current, keys, ladder, index)
        result: tuple[TrajectoryMCMCState, dict[str, jax.Array]] = _run_batched_step(
            current, keys, step_fn=kernel.step_fn
        )
        updated, info = result
        empty = jnp.zeros((num_chains, 0), dtype=jnp.int32)
        return updated, info, ExchangeRecord(empty, empty, empty.astype(ladder.dtype))

    compiled_step = (
        jax.jit(population_step).lower(states, step_keys[0], betas, jnp.asarray(0)).compile()
    )
    for step_idx in range(total_steps):
        step_started = sampling_loop_started if step_idx == 0 else clock()
        states, replica_info, exchange = compiled_step(
            states, step_keys[step_idx], betas, jnp.asarray(step_idx)
        )
        cold_states = jax.tree.map(lambda value: value[cold_indices], states)
        step_info = jax.tree.map(lambda value: value[cold_indices], replica_info)
        if step_idx == 0:
            states.complete_log_posterior.block_until_ready()
            first_step_seconds = clock() - step_started

        if step_idx >= num_warmup:
            swap_accepted = swap_accepted + exchange.accepted
            swap_attempted = swap_attempted + exchange.attempted
            observation_log_prob_history.append(
                score_observations(cold_states.latent_context, cold_states.latent_trajectory)
            )
        position_history.append(cold_states.position)
        parameter_accept_history.append(step_info["parameter_accepted"])
        latent_accept_history.append(
            _masked_mean(step_info["latent_accepted"], step_info["latent_proposed"], axis=-1)
        )
        complete_lp_history.append(cold_states.complete_log_posterior)
        selected_label_history.append(step_info["selected_label"])
        final_particle_history.append(step_info["final_particle"])
        latent_move_rms_history.append(step_info["latent_move_rms"])
        latent_move_max_abs_history.append(step_info["latent_move_max_abs"])
        latent_move_rms_per_t_history.append(step_info["latent_move_rms_per_t"])
        latent_frozen_frac_history.append(step_info["latent_frozen_frac"])
        latent_frozen_frac_by_d_history.append(step_info["latent_frozen_frac_by_d"])
        final_label_log_probs_history.append(step_info["final_label_log_probs"])
        amala_grad_norm_mean_history.append(step_info["amala_grad_norm_mean"])
        amala_grad_norm_max_history.append(step_info["amala_grad_norm_max"])
        if diagnostic_flags.particle_identity:
            selected_particle_per_t_history.append(step_info["selected_particle_per_t"])
            reference_path_hit_rate_history.append(step_info["reference_path_hit_rate"])
            selected_particle_unique_count_history.append(
                step_info["selected_particle_unique_count"]
            )
        if diagnostic_flags.parameter_movement:
            parameter_jump_rms_history.append(step_info["parameter_jump_rms"])

        if need_public_latent:
            public_latent = cold_states.latent_trajectory
            if step_idx >= num_warmup and latent_moments is not None:
                latent_sum, latent_sumsq, sample_count = latent_moments
                latent_moments = (
                    latent_sum + public_latent,
                    latent_sumsq + public_latent * public_latent,
                    sample_count + 1,
                )
            if retain_latent_paths:
                latent_paths_history.append(public_latent)

        if step_idx < num_warmup:
            step_info = replica_info
            window_slot = step_idx % int(kernel.amala_adaptation_window)
            latent_acceptance_window = latent_acceptance_window.at[:, window_slot, :].set(
                step_info["latent_accepted"].astype(latent_acceptance_window.dtype)
            )
            latent_proposal_window = latent_proposal_window.at[:, window_slot, :].set(
                step_info["latent_proposed"].astype(latent_proposal_window.dtype)
            )
            proposal_count = jnp.sum(latent_proposal_window, axis=1)
            latent_acceptance_rate = jnp.sum(latent_acceptance_window, axis=1) / jnp.maximum(
                proposal_count, 1
            )
            target_accept = jnp.asarray(
                kernel.amala_target_accept,
                dtype=states.latent_delta.dtype,
            )
            learning_rate = jnp.maximum(
                jnp.asarray(step_idx + 1, dtype=states.latent_delta.dtype)
                ** jnp.asarray(
                    kernel.amala_adaptation_gamma,
                    dtype=states.latent_delta.dtype,
                )
                * jnp.asarray(kernel.amala_adaptation_rho, dtype=states.latent_delta.dtype),
                jnp.asarray(
                    kernel.amala_adaptation_rho_min,
                    dtype=states.latent_delta.dtype,
                ),
            )
            delta_update = (
                learning_rate
                * states.latent_delta
                * (latent_acceptance_rate - target_accept)
                / target_accept
            )
            should_adapt_latent_delta = jnp.abs(
                latent_acceptance_rate - target_accept
            ) >= jnp.asarray(
                kernel.amala_adaptation_tolerance,
                dtype=states.latent_delta.dtype,
            )
            should_adapt_latent_delta = should_adapt_latent_delta & (
                (step_idx + 1) > int(kernel.amala_adaptation_window)
            )
            should_adapt_latent_delta = (
                should_adapt_latent_delta & latent_active & (proposal_count > 0)
            )
            next_latent_delta = jnp.where(
                should_adapt_latent_delta,
                states.latent_delta + delta_update,
                states.latent_delta,
            )
            states = states._replace(
                latent_delta=_clip_scale(
                    next_latent_delta,
                    min_scale=kernel.amala_delta_min,
                    max_scale=kernel.amala_delta_max,
                )
            )
            if da_param_update is not None:
                # Dual averaging converges (unlike the constant-rate scheme), and we
                # freeze to the Polyak-averaged step at the final warmup step rather
                # than keeping a noisy live value — so per-chain steps no longer
                # scatter across orders of magnitude.
                updated_param_da = jax.vmap(
                    lambda da_state, accepted: _clip_dual_averaging_state(
                        da_param_update(da_state, accepted),
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    )
                )(states.param_da, step_info["parameter_accepted"])
                scale_dtype = states.param_step_size.dtype
                if step_idx == num_warmup - 1:
                    next_param_step = jnp.exp(updated_param_da.log_step_size_avg)
                else:
                    next_param_step = jnp.exp(updated_param_da.log_step_size)
                states = states._replace(
                    param_step_size=_clip_scale(
                        next_param_step.astype(scale_dtype),
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    ),
                    param_da=updated_param_da,
                )
            else:
                states = states._replace(
                    param_step_size=_adapt_scale(
                        states.param_step_size,
                        accepted=step_info["parameter_accepted"],
                        target_accept=kernel.target_accept,
                        adaptation_rate=adaptation_rate,
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    )
                )
            if tempering is not None and rebind is not None:
                warmup_attempts = warmup_attempts + exchange.attempted
                warmup_rejections = warmup_rejections + exchange.rejection_probability
                if (step_idx + 1) % tempering.spec.adaptation_interval == 0 and (
                    step_idx + 1
                ) <= num_warmup // 2:
                    betas = adapt_ladder(betas, warmup_rejections, warmup_attempts)
                    states = rebind(states, betas)
                    warmup_attempts = jnp.zeros_like(warmup_attempts)
                    warmup_rejections = jnp.zeros_like(warmup_rejections)
                    ladder_frozen_at = step_idx + 1
            continue

    states.complete_log_posterior.block_until_ready()
    sampling_loop_seconds = clock() - sampling_loop_started

    all_grouped_positions = _stack_sample_history(
        position_history,
        num_chains=num_chains,
        trailing_shape=(dim,),
        dtype=chain_init_positions.dtype,
    )
    grouped_positions = all_grouped_positions[:, num_warmup:]
    all_chain_extra_fields = {
        "parameter_accept_prob": _stack_sample_history(
            parameter_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_accept_prob": _stack_sample_history(
            latent_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "selected_parameter_label": _stack_sample_history(
            selected_label_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "selected_particle": _stack_sample_history(
            final_particle_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_rms": _stack_sample_history(
            latent_move_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_max_abs": _stack_sample_history(
            latent_move_max_abs_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_rms_per_t": _stack_sample_history(
            latent_move_rms_per_t_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(latent_move_rms_per_t_history[0].shape[1:])
                if latent_move_rms_per_t_history
                else (int(states.latent_trajectory.shape[0]),)
            ),
            dtype=states.latent_trajectory.dtype,
        ),
        "latent_frozen_frac": _stack_sample_history(
            latent_frozen_frac_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_frozen_frac_by_d": _stack_sample_history(
            latent_frozen_frac_by_d_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(latent_frozen_frac_by_d_history[0].shape[1:])
                if latent_frozen_frac_by_d_history
                else (int(states.latent_trajectory.shape[1]),)
            ),
            dtype=states.latent_trajectory.dtype,
        ),
        "final_label_log_probs": _stack_sample_history(
            final_label_log_probs_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(final_label_log_probs_history[0].shape[1:])
                if final_label_log_probs_history
                else (int(kernel.num_parameter_particles),)
            ),
            dtype=chain_init_positions.dtype,
        ),
        "amala_grad_norm_mean": _stack_sample_history(
            amala_grad_norm_mean_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "amala_grad_norm_max": _stack_sample_history(
            amala_grad_norm_max_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
    }
    if diagnostic_flags.particle_identity:
        all_chain_extra_fields.update(
            {
                "selected_particle_per_t": _stack_sample_history(
                    selected_particle_per_t_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(selected_particle_per_t_history[0].shape[1:]),
                    dtype=states.latent_trajectory.dtype,
                ),
                "reference_path_hit_rate": _stack_sample_history(
                    reference_path_hit_rate_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "selected_particle_unique_count": _stack_sample_history(
                    selected_particle_unique_count_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
    if diagnostic_flags.parameter_movement:
        all_chain_extra_fields["parameter_jump_rms"] = _stack_sample_history(
            parameter_jump_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        )
    chain_extra_fields = {
        name: values[:, num_warmup:] for name, values in all_chain_extra_fields.items()
    }
    warmup_chain_extra_fields = {
        name: values[:, :num_warmup] for name, values in all_chain_extra_fields.items()
    }
    all_complete_log_posterior_history = _stack_sample_history(
        complete_lp_history,
        num_chains=num_chains,
        trailing_shape=(),
        dtype=states.complete_log_posterior.dtype,
    )
    complete_log_posterior_history = all_complete_log_posterior_history[:, num_warmup:]
    warmup_complete_log_posterior_history = all_complete_log_posterior_history[:, :num_warmup]

    latent_summary = None
    if latent_moments is not None:
        latent_sum, latent_sumsq, sample_count = latent_moments
        denom = jnp.maximum(sample_count, 1).astype(latent_sum.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        latent_summary = _latent_summary_from_chain_moments(chain_mean, jnp.sqrt(chain_var))

    latent_paths = None
    all_latent_paths = None
    warmup_latent_paths = None
    if retain_latent_paths:
        latent_trailing_shape = (
            tuple(latent_paths_history[0].shape[1:])
            if latent_paths_history
            else tuple(states.latent_trajectory.shape[1:])
        )
        all_latent_paths = _stack_sample_history(
            latent_paths_history,
            num_chains=num_chains,
            trailing_shape=latent_trailing_shape,
            dtype=states.latent_trajectory.dtype,
        )
        latent_paths = all_latent_paths[:, num_warmup:]
        warmup_latent_paths = all_latent_paths[:, :num_warmup]

    post_warmup_complete_log_posterior_mean = (
        jnp.mean(complete_log_posterior_history, axis=1)
        if num_samples > 0
        else jnp.full((num_chains,), jnp.nan, dtype=states.complete_log_posterior.dtype)
    )

    return ParticleChainResult(
        grouped_positions=grouped_positions,
        observation_log_probs=_stack_sample_history(
            observation_log_prob_history,
            num_chains=num_chains,
            trailing_shape=(num_steps,),
            dtype=states.trajectory_log_prob.dtype,
        ),
        chain_extra_fields=chain_extra_fields,
        warmup_chain_extra_fields=warmup_chain_extra_fields,
        all_chain_extra_fields=all_chain_extra_fields,
        complete_log_posterior_history=complete_log_posterior_history,
        warmup_complete_log_posterior_history=warmup_complete_log_posterior_history,
        all_complete_log_posterior_history=all_complete_log_posterior_history,
        latent_posterior_summary=latent_summary,
        latent_paths=latent_paths,
        warmup_latent_paths=warmup_latent_paths,
        all_latent_paths=all_latent_paths,
        initial_param_step_size=initial_param_step_size,
        final_param_step_size=states.param_step_size[cold_indices],
        initial_latent_delta=initial_latent_delta,
        final_latent_delta=states.latent_delta[cold_indices],
        compiled_step=compiled_step,
        first_step_seconds=first_step_seconds,
        sampling_loop_seconds=sampling_loop_seconds,
        post_warmup_complete_log_posterior_mean=post_warmup_complete_log_posterior_mean,
        tempering=None
        if tempering is None
        else TemperingDiagnostics(
            inverse_temperatures=tuple(float(beta) for beta in jax.device_get(betas)),
            ladder_frozen_at_warmup_step=ladder_frozen_at,
            swaps=tuple(
                tuple(
                    ReplicaSwapCounts(accepted=int(a), attempted=int(n))
                    for a, n in zip(accepted, attempted, strict=True)
                )
                for accepted, attempted in zip(
                    jax.device_get(swap_accepted), jax.device_get(swap_attempted), strict=True
                )
            ),
        ),
    )
