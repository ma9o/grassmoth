"""Run structured particle kernels with Dynestyx parameter/model interpretations."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import DualAveragingAdaptationState, dual_averaging_adaptation
from jax.stages import Compiled

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ParticleWeightMetric,
    ReplicaSwapCounts,
    TemperingDiagnostics,
)
from grassmoth.artifacts.reports.profiling import (
    FitComponentTiming,
    FitProfileReport,
    NutsProfileReport,
    ParticleProfileDimensions,
    TimingSpread,
)
from grassmoth.artifacts.sampling import (
    DsmcSpec,
    FixedLatentStepSpec,
    ParticleAGradSpec,
    ParticleMGradSpec,
    TwistedParticleAGradSpec,
)
from grassmoth.compute.budget import ComputeBudget
from grassmoth.compute.programs import ProgramCompiler, ProgramPreparer, scalar_operand
from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
from grassmoth.models.ssm.inference.continuation import restore_state, retain_state
from grassmoth.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    trajectory_state,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.companion import (
    CompanionKernel,
    CompanionTuning,
    restarted_dual_averaging,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    MarginalParticleGibbsKernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.tempering import (
    ExchangeRecord,
    ReplicaExchangeKernel,
    adapt_temperature_ladder,
)
from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    DELTA_MAX,
    DsmcAdaptation,
    adapt_scales,
    guided_delta_limit,
    initial_delta,
    initialize_adaptation,
    latent_adaptation,
)
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    excluded_particle_diagnostics,
)
from grassmoth.models.ssm.inference.samplers.particle.math import _masked_mean
from grassmoth.models.ssm.inference.samplers.profiling import (
    REPETITIONS,
    measure,
    measure_sweep,
    sum_timings,
)
from grassmoth.models.ssm.inference.samplers.progress import inference_phase
from grassmoth.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase
from grassmoth.models.ssm.inference.samplers.sweeps import (
    ConvergenceCheckpoint,
    RetainedConvergenceCheck,
    compile_projected_sweeps,
    run_sweeps,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization
from grassmoth.models.ssm.inference.warmup.parameter_adaptation import (
    ParameterCovariance,
    adapt_parameter_covariance,
    covariance_windows,
    initialize_parameter_covariance,
)
from grassmoth.models.ssm.shapes import Array, Float, PRNGKeyArray


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
    latent_paths: jnp.ndarray
    initial_param_step_size: jnp.ndarray
    final_param_step_size: jnp.ndarray
    final_parameter_chol: jax.Array
    initial_latent_delta: jnp.ndarray
    final_latent_delta: jnp.ndarray
    compiled_step: Compiled
    first_step_seconds: float
    sampling_loop_seconds: float
    final_state: tuple[NumericalArray, ...]
    random_key: jax.Array
    last_parameters: jax.Array
    last_paths: jax.Array
    retained_draws: int
    stopping_checkpoint: ConvergenceCheckpoint | None
    tempering: TemperingDiagnostics | None = None
    profile: FitProfileReport | None = None

    @property
    def post_warmup_complete_log_posterior_mean(self) -> jnp.ndarray:
        """Project the retained density history without retaining a duplicate summary."""
        return jnp.mean(self.complete_log_posterior_history, axis=1)

    def __post_init__(self) -> None:
        """Freeze retained, warmup, and combined chain diagnostics."""
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
        freeze_fields(self)


def _initialize_chain_state(
    init_position: Float[Array, " U"],
    *,
    observations: Float[Array, "T M"],
    times: Float[Array, " T"],
    target: ParticleTarget,
    parameter_gradient_fn: Callable[[Float[Array, " U"]], Float[Array, " U"]],
    initial_latent_delta: Float[Array, " T"] | Float[Array, "T D"],
    param_step_size: float | Float[Array, ""],
    parameter_chol: Float[Array, "U U"],
    param_min_scale: float | Float[Array, ""],
    param_max_scale: float | Float[Array, ""],
    initial_latent_trajectory: Float[Array, "T D"],
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
    da_init, _da_update, _ = dual_averaging_adaptation(target=0.0)
    return trajectory_state(
        position=init_position,
        parameter_gradient=parameter_gradient_fn(init_position),
        latent_context=context,
        latent_trajectory=latent_trajectory,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
        latent_delta=latent_delta_value,
        param_step_size=param_step_value,
        parameter_chol=parameter_chol,
        param_da=da_init(param_step_value),
    )


class _Population(NamedTuple):
    chains: TrajectoryMCMCState
    random_key: jax.Array
    iteration: jax.Array
    betas: jax.Array
    adaptation: DsmcAdaptation
    parameter_covariance: ParameterCovariance
    swap_accepted: jax.Array
    swap_attempted: jax.Array
    warmup_attempts: jax.Array
    warmup_rejections: jax.Array
    ladder_frozen_at: jax.Array
    companions: CompanionTuning | None
    latent_geometry: jax.Array


# Gain of the simple step-size scheme; a fixed property of warmup, not a setting.
ADAPTATION_RATE = 0.05


def _initial_companion_tuning(
    companion: CompanionKernel,
    states: TrajectoryMCMCState,
    key: PRNGKeyArray,
    num_warmup: int,
    compiler: ProgramCompiler,
) -> CompanionTuning:
    # The NUTS warmup schedule spans this run's warmup; it is frozen when that ends.
    tuning_key = random.fold_in(key, 1)

    def companion_initializer(
        chains: TrajectoryMCMCState, draw_key: PRNGKeyArray
    ) -> CompanionTuning:
        return companion.initial_tuning(chains, draw_key, num_warmup)

    program = compiler(
        companion_initializer,
        states,
        tuning_key,
    )
    return program.call(states, tuning_key)


def run_marginal_particle_gibbs(
    target: ParticleTarget,
    *,
    compiler: ProgramCompiler,
    prepare_program: ProgramPreparer,
    kernel: MarginalParticleGibbsKernel,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    key: PRNGKeyArray,
    adaptation_rate: float = ADAPTATION_RATE,
    initialization: ParticleInitialization,
    compute_budget: ComputeBudget,
    # Both adaptation policies stop updating after warmup.
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple",
    tempering: ReplicaExchangeKernel | None = None,
    companion: CompanionKernel | None = None,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    profile: bool = False,
    continuation: SamplerCheckpoint | None = None,
    convergence: RetainedConvergenceCheck | None = None,
) -> ParticleChainResult:
    """Run marginalized Particle Gibbs chains from paired starts."""
    assert companion is None or tempering is None, "The sampler spec owns this exclusion"
    windows = covariance_windows(num_warmup)
    use_dual_averaging = adaptation_scheme == "dual_averaging"
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

    initial_latent_delta_value = initial_delta(kernel.path_update)
    initial_latent_delta = jnp.full(
        (num_steps,),
        jnp.asarray(initial_latent_delta_value, dtype=observations.dtype),
    )
    curvature_limited = (
        isinstance(
            kernel.path_update, (ParticleAGradSpec, TwistedParticleAGradSpec, ParticleMGradSpec)
        )
        and kernel.path_update.kappa == 1
        and not isinstance(kernel.path_update.latent_step, FixedLatentStepSpec)
    )
    latent_free_mask = (
        target.independent_mask
        if kernel.exact_constraints is None
        else target.independent_mask & kernel.exact_constraints.free_mask
    )

    def particle_chain_initializer(
        position: Float[Array, " U"], path: Float[Array, "T D"]
    ) -> tuple[TrajectoryMCMCState, Float[Array, "T D"]]:
        state = _initialize_chain_state(
            position,
            observations=observations,
            times=times,
            target=target,
            parameter_gradient_fn=kernel.parameter_gradient_fn,
            initial_latent_delta=initial_latent_delta,
            exact_constraints=kernel.exact_constraints,
            param_step_size=kernel.initial_param_step_size[0],
            parameter_chol=kernel.initial_parameter_chol,
            param_min_scale=kernel.min_scale[0],
            param_max_scale=kernel.max_scale[0],
            initial_latent_trajectory=path,
        )
        geometry = jnp.ones_like(path)
        if curvature_limited:
            geometry = guided_delta_limit(
                target, state.latent_context, state.latent_trajectory, latent_free_mask
            )
            # The public per-time scale is the movement multiplier in (0, 1].
            state = state._replace(latent_delta=jnp.full_like(state.latent_delta, 0.25))
        return state, geometry

    with inference_phase(
        "mpgibbs", "initializing_chains", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        initialize = compiler(
            jax.vmap(particle_chain_initializer),
            chain_init_positions,
            chain_initial_latents,
        )
        states, latent_geometry = initialize.call(chain_init_positions, chain_initial_latents)
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
    latent_geometry = jnp.repeat(latent_geometry, levels, axis=0)
    if tempering is not None:
        states = tempering.rebind(states, betas)
    swap_accepted = jnp.zeros((num_chains, levels - 1), dtype=jnp.int32)
    swap_attempted = jnp.zeros_like(swap_accepted)
    warmup_attempts = jnp.zeros_like(swap_attempted)
    warmup_rejections = jnp.zeros(swap_attempted.shape, dtype=betas.dtype)
    warmup_steps = scalar_operand(value=num_warmup, dtype=jnp.int32)
    scale_adaptation = latent_adaptation(kernel.path_update)
    runtime_adaptation_rate = scalar_operand(value=adaptation_rate)
    ladder_interval = scalar_operand(
        value=1 if tempering is None else tempering.spec.adaptation_interval, dtype=jnp.int32
    )
    initial = _Population(
        states,
        key
        if continuation is None
        else jax.random.wrap_key_data(jax.device_put(continuation.random_key.values)),
        jnp.asarray(0),
        betas,
        initialize_adaptation(states.latent_delta),
        initialize_parameter_covariance(chain_init_positions, initialization.parameter_chol),
        swap_accepted,
        swap_attempted,
        warmup_attempts,
        warmup_rejections,
        jnp.asarray(0),
        None
        if companion is None
        else _initial_companion_tuning(companion, states, key, num_warmup, compiler),
        latent_geometry,
    )
    if continuation is not None and num_warmup == 0:
        initial = restore_state(initial, continuation.native_state)
        scale = kernel.path_update.latent_step
        if isinstance(scale, FixedLatentStepSpec):
            delta = jnp.full_like(initial.chains.latent_delta, scale.delta)
            initial = initial._replace(
                chains=initial.chains._replace(latent_delta=delta),
                adaptation=initial.adaptation._replace(delta=delta),
            )
    initial_param_step_size = initial.chains.param_step_size[cold_indices]
    initial_latent_delta = initial.chains.latent_delta[cold_indices]

    def step_factory(
        batch_size: int,
    ) -> Callable[[_Population, jax.Array], tuple[_Population, dict[str, jax.Array]]]:
        def sweep(
            current: _Population, posterior: jax.Array
        ) -> tuple[_Population, dict[str, jax.Array]]:
            next_key, step_key = random.split(current.random_key)
            keys = random.split(step_key, replicas)
            # The frozen metric is shared by every label/candidate in the update.
            # Keep its scalar controller in the state and saved response; expand
            # only the kernel operand, then restore it before companion moves.
            kernel_states = (
                current.chains._replace(
                    latent_delta=current.chains.latent_delta[..., None] * current.latent_geometry
                )
                if curvature_limited
                else current.chains
            )
            if tempering is None:
                updated, info = jax.vmap(
                    lambda state, draw_key: kernel.step_fn(state, draw_key, batch_size=batch_size)
                )(kernel_states, keys)
                empty = jnp.zeros((num_chains, 0), dtype=jnp.int32)
                exchange = ExchangeRecord(empty, empty, empty.astype(current.betas.dtype))
            else:
                updated, info, exchange = tempering.step(
                    kernel_states, keys, current.betas, current.iteration, batch_size=batch_size
                )
            if curvature_limited:
                updated = updated._replace(latent_delta=current.chains.latent_delta)
            moved_companions = current.companions
            companion_accepted = None
            companion_record: dict[str, jax.Array] = {}
            if companion is not None and moved_companions is not None:
                updated, moved_companions, companion_accepted, records = jax.vmap(companion.step)(
                    updated,
                    moved_companions,
                    jax.vmap(lambda draw_key: random.fold_in(draw_key, 1))(keys),
                )
                companion_record = {name: values[cold_indices] for name, values in records.items()}
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
            record = {
                **record,
                **{
                    metric.history_key: cold_info[metric.history_key]
                    for metric in ParticleWeightMetric
                },
                **{
                    name: cold_info[name]
                    for name in (
                        "selected_particle_per_t",
                        "reference_path_hit_rate",
                        "selected_particle_unique_count",
                        "parameter_jump_rms",
                    )
                },
                **companion_record,
            }

            def adapt_steps(
                step_size: jax.Array,
                dual_averaging: DualAveragingAdaptationState,
                accepted: jax.Array,
            ) -> tuple[jax.Array, DualAveragingAdaptationState]:
                """Advance one batch of step scalars under the configured warmup policy."""
                if not use_dual_averaging:
                    return _adapt_scale(
                        step_size,
                        accepted=accepted,
                        target_accept=kernel.target_accept[0],
                        adaptation_rate=runtime_adaptation_rate[0],
                        min_scale=kernel.min_scale[0],
                        max_scale=kernel.max_scale[0],
                    ), dual_averaging
                update = dual_averaging_adaptation(target=kernel.target_accept[0])[1]  # pyright: ignore[reportArgumentType] -- BlackJAX accepts scalar JAX arithmetic; its target annotation only permits float.
                dual_averaging = jax.vmap(
                    lambda state, rate: _clip_dual_averaging_state(
                        update(state, rate),
                        min_scale=kernel.min_scale[0],
                        max_scale=kernel.max_scale[0],
                    )
                )(dual_averaging, accepted)
                log_step = jnp.where(
                    current.iteration + 1 == warmup_steps[0],
                    dual_averaging.log_step_size_avg,
                    dual_averaging.log_step_size,
                )
                return _clip_scale(
                    jnp.exp(log_step).astype(step_size.dtype),
                    min_scale=kernel.min_scale[0],
                    max_scale=kernel.max_scale[0],
                ), dual_averaging

            def retain(updated: TrajectoryMCMCState) -> _Population:
                return current._replace(
                    chains=updated,
                    random_key=next_key,
                    iteration=current.iteration + 1,
                    swap_accepted=current.swap_accepted + exchange.accepted,
                    swap_attempted=current.swap_attempted + exchange.attempted,
                )

            def adapt(updated: TrajectoryMCMCState) -> _Population:
                geometry = (
                    jax.vmap(
                        lambda state: guided_delta_limit(
                            target, state.latent_context, state.latent_trajectory, latent_free_mask
                        )
                    )(updated)
                    if curvature_limited
                    else current.latent_geometry
                )
                maximum = jnp.full_like(
                    updated.latent_delta, 1.0 if curvature_limited else DELTA_MAX
                )
                tuning = adapt_scales(
                    current.adaptation,
                    scale_adaptation,
                    info["latent_accepted"],
                    info["latent_proposed"],
                    warmup_steps[0],
                    maximum,
                )
                updated = updated._replace(latent_delta=tuning.delta)
                param_step_size, param_da = adapt_steps(
                    updated.param_step_size, updated.param_da, info["parameter_accepted"]
                )
                updated = updated._replace(param_step_size=param_step_size, param_da=param_da)
                # NUTS adapted itself in its transition; Metropolis steps adapt here.
                companions = moved_companions
                if companions is not None and companion_accepted is not None:
                    step_size, dual_averaging = jax.vmap(adapt_steps, in_axes=1, out_axes=1)(
                        companions.step_size, companions.dual_averaging, companion_accepted
                    )
                    companions = companions._replace(
                        step_size=step_size, dual_averaging=dual_averaging
                    )
                covariance, parameter_chol, changed = adapt_parameter_covariance(
                    current.parameter_covariance,
                    cold.position,
                    updated.parameter_chol[0],
                    current.iteration,
                    windows,
                )

                def replace_geometry(
                    chains: TrajectoryMCMCState, companion_tuning: CompanionTuning | None
                ) -> tuple[TrajectoryMCMCState, CompanionTuning | None]:
                    if dim == 0:
                        return chains, companion_tuning
                    # Covariance estimates posterior geometry; acceptance feedback
                    # tunes its scalar multiplier after the geometry changes.

                    def restart(
                        step_size: jax.Array,
                    ) -> tuple[jax.Array, DualAveragingAdaptationState]:
                        scale = _clip_scale(
                            step_size,
                            min_scale=kernel.min_scale[0],
                            max_scale=kernel.max_scale[0],
                        )
                        return scale, restarted_dual_averaging(scale)

                    scale, param_da = restart(chains.param_step_size)
                    chains = chains._replace(
                        parameter_chol=jnp.broadcast_to(
                            parameter_chol, chains.parameter_chol.shape
                        ),
                        param_step_size=scale,
                        param_da=param_da,
                    )
                    if companion_tuning is None:
                        return chains, None
                    # NUTS keeps its own mass matrix, so only Metropolis steps restart.
                    steps, dual_averaging = restart(companion_tuning.step_size)
                    return chains, companion_tuning._replace(
                        step_size=steps, dual_averaging=dual_averaging
                    )

                updated, companions = jax.lax.cond(
                    changed,
                    replace_geometry,
                    lambda chains, companion_tuning: (chains, companion_tuning),
                    updated,
                    companions,
                )
                next_state: _Population = current._replace(
                    chains=updated,
                    random_key=next_key,
                    iteration=current.iteration + 1,
                    adaptation=tuning,
                    latent_geometry=geometry,
                    parameter_covariance=covariance,
                    companions=companions,
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
                        (next_state.iteration % ladder_interval[0] == 0)
                        & (next_state.iteration <= warmup_steps[0] // 2),
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

    with inference_phase(
        "mpgibbs", "compiling_sweeps", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        plan = compile_projected_sweeps(
            step_factory,
            initial,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_batch_size=max(
                kernel.path_update.num_particles, (1 << (num_steps - 1).bit_length()) // 2
            )
            if isinstance(kernel.path_update, DsmcSpec)
            else 1,
            budget=compute_budget,
            target=target,
            path_dtype=states.latent_trajectory.dtype,
            num_chains=num_chains,
            clock=clock,
            logger=logger,
            wall_clock=wall_clock,
            label="mpgibbs",
            prepare_program=prepare_program,
            excluded_history=excluded_particle_diagnostics(kernel.diagnostic_metrics),
        )
    dimensions = ParticleProfileDimensions(
        particles=kernel.path_update.num_particles,
        labels=kernel.num_parameter_particles,
        replicas=replicas,
    )

    def profile_state(current: _Population, phase: CompiledSweepPhase) -> FitProfileReport:
        full = measure_sweep(current, phase, clock)
        _, step_key = random.split(current.random_key)
        keys = random.split(step_key, replicas)
        kernel_states = (
            current.chains._replace(
                latent_delta=current.chains.latent_delta[..., None] * current.latent_geometry
            )
            if curvature_limited
            else current.chains
        )
        if tempering is None:
            parts = (kernel.profile_parts(kernel_states, keys, plan.batch_size, compiler),)
        else:
            # Production locally updates every non-reference replica. Splitting
            # these temperature batches is itself part of the measured split cost.
            levels = current.betas.size
            parts = tuple(
                tempering.kernel_for_target(
                    target.at_temperature(tempering.reference, beta)
                ).profile_parts(
                    jax.tree.map(lambda value, index=index: value[index::levels], kernel_states),
                    keys[index::levels],
                    plan.batch_size,
                    compiler,
                )
                for index, beta in enumerate(current.betas[:-1])
            )
        parameter, _ = measure(lambda: tuple(part.parameter_move() for part in parts), clock)
        path, _ = measure(lambda: tuple(part.path_smoother() for part in parts), clock)
        rows = (
            FitComponentTiming(component="full_sweep", timing=TimingSpread.from_seconds(full)),
            FitComponentTiming(
                component="parameter_move", timing=TimingSpread.from_seconds(parameter)
            ),
            FitComponentTiming(component="path_smoother", timing=TimingSpread.from_seconds(path)),
        )
        split = (parameter, path)
        nuts_report = None
        if companion is not None and current.companions is not None:
            # Use the same native companion and keys as production, with frozen
            # tuning on the copied current state. No output advances the chain.
            companion_keys = jax.vmap(lambda key: random.fold_in(key, 1))(keys)
            companion_step = companion.step

            def profile_companion(
                chains: TrajectoryMCMCState, tuning: CompanionTuning, draw_keys: jax.Array
            ) -> tuple[TrajectoryMCMCState, CompanionTuning, jax.Array, dict[str, jax.Array]]:
                return jax.vmap(companion_step)(chains, tuning, draw_keys)

            transition = compiler(
                profile_companion, current.chains, current.companions, companion_keys
            )
            elapsed, (_, _, _, records) = measure(
                lambda: transition.call(current.chains, current.companions, companion_keys), clock
            )
            split = (*split, elapsed)
            rows = (
                *rows,
                FitComponentTiming(
                    component="nuts_companion" if companion.nuts is not None else "mh_companion",
                    timing=TimingSpread.from_seconds(elapsed),
                ),
            )
            if companion.nuts is not None:
                position = current.chains.position[0]
                latent_path = current.chains.latent_trajectory[0]
                energy = companion.nuts.energy

                def profile_conditional_gradient(
                    position: jax.Array, path: jax.Array
                ) -> tuple[jax.Array, jax.Array]:
                    return cast(
                        "tuple[jax.Array, jax.Array]", jax.value_and_grad(energy)(position, path)
                    )

                gradient = compiler(profile_conditional_gradient, position, latent_path)
                elapsed_gradient, _ = measure(lambda: gradient.call(position, latent_path), clock)
                rows = (
                    *rows,
                    FitComponentTiming(
                        component="conditional_gradient",
                        timing=TimingSpread.from_seconds(elapsed_gradient),
                    ),
                )
                nuts_report = NutsProfileReport(
                    leapfrog_steps=tuple(int(value) for value in records["centred_nuts_num_steps"]),
                    step_size=tuple(float(value) for value in records["centred_nuts_step_size"]),
                    tree_depth=tuple(int(value) for value in records["tree_depth"]),
                    gradient_count_representative=num_warmup > 0,
                )
        return FitProfileReport(
            phase="after_warmup" if num_warmup else "after_first_sweep",
            repetitions=REPETITIONS,
            chains=num_chains,
            time_steps=num_steps,
            particle_dimensions=dimensions,
            components=rows,
            split_total=sum_timings(split),
            nuts=nuts_report,
        )

    run = run_sweeps(
        plan,
        initial,
        clock=clock,
        logger=logger,
        wall_clock=wall_clock,
        label="mpgibbs",
        convergence=convergence,
        profile=profile_state if profile else None,
    )
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
    complete_lp = run.history.posterior["log_posterior"]
    return ParticleChainResult(
        starting_values={
            "positions": initial.chains.position[cold_indices],
            "paths": initial.chains.latent_trajectory[cold_indices],
            "log_joint": initial.chains.complete_log_posterior[cold_indices],
        },
        retained_draws=run.retained_draws,
        stopping_checkpoint=run.stopping_checkpoint,
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
        latent_paths=run.history.posterior["paths"],
        initial_param_step_size=initial_param_step_size,
        final_param_step_size=final.chains.param_step_size[cold_indices],
        final_parameter_chol=final.chains.parameter_chol[0],
        initial_latent_delta=initial_latent_delta,
        final_latent_delta=final.chains.latent_delta[cold_indices],
        profile=run.profile
        if run.profile is not None
        else FitProfileReport(
            phase="compile_only",
            repetitions=0,
            chains=num_chains,
            time_steps=num_steps,
            particle_dimensions=dimensions,
        )
        if profile
        else None,
        compiled_step=plan.compiled_step,
        first_step_seconds=run.first_step_seconds,
        sampling_loop_seconds=run.sampling_loop_seconds,
        final_state=retain_state(final),
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
