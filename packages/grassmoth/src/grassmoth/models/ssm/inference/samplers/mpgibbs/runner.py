"""Run structured particle kernels with Dynestyx parameter/model interpretations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from blackjax.adaptation.step_size import DualAveragingAdaptationState, dual_averaging_adaptation
from jax.stages import Compiled
from pydantic import Field

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
from grassmoth.artifacts.reports.profiling import (
    FitComponentTiming,
    FitProfileReport,
    ParticleProfileDimensions,
    TimingSpread,
)
from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    FixedLatentStepSpec,
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.compute.context import ModelExecutionContext
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
from grassmoth.models.ssm.inference.continuation import restore_state, retain_state
from grassmoth.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    restarted_dual_averaging,
    trajectory_state,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    MarginalParticleGibbsKernel,
)
from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    DELTA_MAX,
    INITIAL_MOVEMENT_MULTIPLIER,
    CurvatureMoments,
    ScaleAdaptationState,
    adapt_curvature_geometry,
    adapt_scales,
    guided_curvature,
    guided_delta_limit,
    initial_delta,
    initialize_adaptation,
    initialize_curvature_moments,
    latent_adaptation,
)
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    excluded_particle_diagnostics,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_random import PARTICLE_PRNG
from grassmoth.models.ssm.inference.samplers.particle.math import _masked_mean
from grassmoth.models.ssm.inference.samplers.profiling import (
    REPETITIONS,
    measure,
    measure_sweep,
    sum_timings,
)
from grassmoth.models.ssm.inference.samplers.progress import inference_phase
from grassmoth.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase, SweepProgram
from grassmoth.models.ssm.inference.samplers.sweeps import (
    compile_projected_sweeps,
    run_sweeps,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
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
    all_chain_extra_fields: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]
    starting_values: Mapping[str, jnp.ndarray]
    complete_log_posterior_history: jnp.ndarray
    warmup_complete_log_posterior_history: jnp.ndarray
    all_complete_log_posterior_history: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    latent_paths: jnp.ndarray | np.ndarray[tuple[int, ...], np.dtype[np.generic]]
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
    profile: FitProfileReport | None = None

    @property
    def post_warmup_complete_log_posterior_mean(
        self,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Project the retained density history without retaining a duplicate summary."""
        return np.mean(np.asarray(self.complete_log_posterior_history), axis=1)

    def __post_init__(self) -> None:
        """Freeze retained, warmup, and combined chain diagnostics."""
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
    adaptation: ScaleAdaptationState
    parameter_covariance: ParameterCovariance
    latent_geometry: jax.Array
    curvature_moments: CurvatureMoments


ADAPTATION_RATE: Annotated[
    float,
    Field(
        description=(
            "Gain of the simple step-size scheme, a fixed property of warmup rather than a "
            "setting. Heuristic: no paper or recorded benchmark sets it."
        )
    ),
] = 0.05


def run_marginal_particle_gibbs(
    target: ParticleTarget,
    *,
    kernel: MarginalParticleGibbsKernel,
    sampler: SamplerSpec,
    key: PRNGKeyArray,
    initialization: ParticleInitialization,
    execution_context: ModelExecutionContext,
    continuation: SamplerCheckpoint | None = None,
) -> ParticleChainResult | InitializationFailure:
    """Run marginalized Particle Gibbs chains from paired starts."""
    num_warmup = sampler.num_warmup
    num_chains = sampler.num_chains
    compiler = execution_context.compiler
    clock = execution_context.clock
    windows = covariance_windows(num_warmup)
    # Both adaptation policies stop updating after warmup.
    options = sampler.algorithm
    assert isinstance(options, MarginalParticleGibbsSpec)
    use_dual_averaging = options.adaptation_scheme == "dual_averaging"
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
        isinstance(kernel.path_update.method, GuidedMethodSpec)
        and kernel.path_update.method.kappa == 1
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
            state = state._replace(
                latent_delta=jnp.full_like(state.latent_delta, INITIAL_MOVEMENT_MULTIPLIER)
            )
        return state, geometry

    with inference_phase("mpgibbs", "initializing_chains", execution_context=execution_context):
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
    warmup_steps = scalar_operand(value=num_warmup, dtype=jnp.int32)
    scale_adaptation = latent_adaptation(kernel.path_update)
    runtime_adaptation_rate = scalar_operand(value=ADAPTATION_RATE)
    initial = _Population(
        states,
        key
        if continuation is None
        else jax.random.wrap_key_data(
            jax.device_put(continuation.random_key.values), impl=PARTICLE_PRNG
        ),
        jnp.asarray(0),
        initialize_adaptation(states.latent_delta),
        initialize_parameter_covariance(chain_init_positions, initialization.parameter_chol),
        latent_geometry,
        initialize_curvature_moments(latent_geometry),
    )
    if continuation is not None and num_warmup == 0:
        restored = restore_state(initial, continuation.native_state)
        if isinstance(restored, InitializationFailure):
            return restored
        initial = restored
        scale = kernel.path_update.latent_step
        if isinstance(scale, FixedLatentStepSpec):
            delta = jnp.full_like(initial.chains.latent_delta, scale.delta)
            initial = initial._replace(
                chains=initial.chains._replace(latent_delta=delta),
                adaptation=initial.adaptation._replace(delta=delta),
            )
    initial_param_step_size = initial.chains.param_step_size
    initial_latent_delta = initial.chains.latent_delta

    def sweep(
        current: _Population, posterior: jax.Array
    ) -> tuple[_Population, dict[str, jax.Array]]:
        next_key, step_key = random.split(current.random_key)
        keys = random.split(step_key, num_chains)
        # Keep its scalar controller in the state and saved response; expand
        # only the kernel operand, then restore it before adaptation.
        kernel_states = (
            current.chains._replace(
                latent_delta=current.chains.latent_delta[..., None] * current.latent_geometry
            )
            if curvature_limited
            else current.chains
        )
        updated, info = jax.vmap(kernel.step_fn)(kernel_states, keys)
        if curvature_limited:
            updated = updated._replace(latent_delta=current.chains.latent_delta)
        record: dict[str, jax.Array] = {
            "log_posterior": updated.complete_log_posterior,
            "parameter_accept_prob": info["parameter_accepted"],
            "latent_accept_prob": _masked_mean(
                info["latent_accepted"], info["latent_proposed"], axis=-1
            ),
            "selected_parameter_label": info["selected_label"],
            "selected_particle": info["final_particle"],
            **{
                name: info[name]
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
            **(
                {
                    "path_block_acceptance": info["path_block_acceptance"],
                    "path_update_rms": info["path_update_rms"],
                }
                if isinstance(kernel.path_update.method, AuxiliaryKalmanMethodSpec)
                else {}
            ),
            **{
                metric.history_key: info[metric.history_key]
                for metric in (
                    ()
                    if isinstance(kernel.path_update.method, AuxiliaryKalmanMethodSpec)
                    else ParticleWeightMetric
                )
            },
            **{
                name: info[name]
                for name in (
                    "selected_particle_per_t",
                    "reference_path_hit_rate",
                    "selected_particle_unique_count",
                    "parameter_jump_rms",
                )
            },
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
                chains=updated, random_key=next_key, iteration=current.iteration + 1
            )

        def adapt(updated: TrajectoryMCMCState) -> _Population:
            geometry = current.latent_geometry
            curvature_moments = current.curvature_moments
            if curvature_limited:
                curvature = jax.vmap(
                    lambda state: guided_curvature(
                        target, state.latent_context, state.latent_trajectory, latent_free_mask
                    )
                )(updated)
                curvature_moments, geometry = adapt_curvature_geometry(
                    curvature_moments, curvature, geometry, current.iteration, windows
                )
            maximum = jnp.full_like(updated.latent_delta, 1.0 if curvature_limited else DELTA_MAX)
            tuning = adapt_scales(
                current.adaptation,
                scale_adaptation,
                info["latent_accepted"],
                info["latent_proposed"],
                warmup_steps[0],
                maximum,
            )
            updated = updated._replace(latent_delta=tuning.delta)
            parameter_feedback = info["parameter_accepted"]
            if use_dual_averaging:
                # Rao-Blackwellize the label draw, then pool chains. Identical
                # starts and feedback keep their DA states shared, including
                # covariance restarts and the final averaged step.
                switch_probability = -jnp.expm1(info["final_label_log_probs"][:, 0])
                parameter_feedback = jnp.full_like(switch_probability, jnp.mean(switch_probability))
            param_step_size, param_da = adapt_steps(
                updated.param_step_size, updated.param_da, parameter_feedback
            )
            updated = updated._replace(param_step_size=param_step_size, param_da=param_da)
            covariance, parameter_chol, changed = adapt_parameter_covariance(
                current.parameter_covariance,
                updated.position,
                updated.parameter_chol[0],
                current.iteration,
                windows,
            )

            def replace_geometry(chains: TrajectoryMCMCState) -> TrajectoryMCMCState:
                if dim == 0:
                    return chains
                # Covariance estimates posterior geometry; acceptance feedback
                # tunes its scalar multiplier after the geometry changes.

                scale = _clip_scale(
                    chains.param_step_size,
                    min_scale=kernel.min_scale[0],
                    max_scale=kernel.max_scale[0],
                )
                return chains._replace(
                    parameter_chol=jnp.broadcast_to(parameter_chol, chains.parameter_chol.shape),
                    param_step_size=scale,
                    param_da=restarted_dual_averaging(scale),
                )

            updated = jax.lax.cond(
                changed,
                replace_geometry,
                lambda chains: chains,
                updated,
            )
            return current._replace(
                chains=updated,
                random_key=next_key,
                iteration=current.iteration + 1,
                adaptation=tuning,
                latent_geometry=geometry,
                curvature_moments=curvature_moments,
                parameter_covariance=covariance,
            )

        next_population: _Population = jax.lax.cond(posterior, retain, adapt, updated)
        return next_population, {
            **record,
            "positions": updated.position,
            "paths": updated.latent_trajectory,
        }

    with inference_phase("mpgibbs", "compiling_sweeps", execution_context=execution_context):
        plan = compile_projected_sweeps(
            SweepProgram(
                sweep, excluded_history=excluded_particle_diagnostics(kernel.diagnostic_metrics)
            ),
            initial,
            execution_context=execution_context,
            sampler=sampler,
            target=target,
        )
    dimensions = (
        None
        if isinstance(kernel.path_update.method, AuxiliaryKalmanMethodSpec)
        else ParticleProfileDimensions(
            particles=kernel.path_update.method.num_particles, labels=kernel.num_parameter_particles
        )
    )

    def profile_state(current: _Population, phase: CompiledSweepPhase) -> FitProfileReport:
        full = measure_sweep(current, phase, clock)
        _, step_key = random.split(current.random_key)
        keys = random.split(step_key, num_chains)
        kernel_states = (
            current.chains._replace(
                latent_delta=current.chains.latent_delta[..., None] * current.latent_geometry
            )
            if curvature_limited
            else current.chains
        )
        parts = kernel.profile_parts(kernel_states, keys, compiler)
        parameter, _ = measure(parts.parameter_move, clock)
        path, _ = measure(parts.path_smoother, clock)
        rows = (
            FitComponentTiming(component="full_sweep", timing=TimingSpread.from_seconds(full)),
            FitComponentTiming(
                component="parameter_move", timing=TimingSpread.from_seconds(parameter)
            ),
            FitComponentTiming(component="path_smoother", timing=TimingSpread.from_seconds(path)),
        )
        split = (parameter, path)
        return FitProfileReport(
            phase="after_warmup" if num_warmup else "after_first_sweep",
            repetitions=REPETITIONS,
            chains=num_chains,
            time_steps=num_steps,
            particle_dimensions=dimensions,
            components=rows,
            split_total=sum_timings(split),
        )

    run = run_sweeps(
        plan,
        initial,
        execution_context=execution_context,
        profile=profile_state if execution_context.profile else None,
    )
    final = run.state
    draw_fields = frozenset(("positions", "paths", "observation_log_probs", "log_posterior"))
    warmup_extra = {
        name: value for name, value in run.history.warmup.items() if name not in draw_fields
    }
    posterior_extra = {
        name: value for name, value in run.history.posterior.items() if name not in draw_fields
    }
    # Combined display histories need no device allocation alongside the full
    # retained paths. Their consumers serialize or summarize host values.
    all_extra = {
        name: np.concatenate((np.asarray(warmup_extra[name]), np.asarray(value)), axis=1)
        for name, value in posterior_extra.items()
    }
    complete_lp = run.history.posterior["log_posterior"]
    return ParticleChainResult(
        starting_values={
            "positions": initial.chains.position,
            "paths": initial.chains.latent_trajectory,
            "log_joint": initial.chains.complete_log_posterior,
        },
        retained_draws=run.retained_draws,
        grouped_positions=run.history.posterior["positions"],
        observation_log_probs=run.history.posterior["observation_log_probs"],
        chain_extra_fields=posterior_extra,
        warmup_chain_extra_fields=warmup_extra,
        all_chain_extra_fields=all_extra,
        complete_log_posterior_history=complete_lp,
        warmup_complete_log_posterior_history=run.history.warmup["log_posterior"],
        all_complete_log_posterior_history=np.concatenate(
            (np.asarray(run.history.warmup["log_posterior"]), np.asarray(complete_lp)), axis=1
        ),
        latent_paths=run.history.posterior["paths"],
        initial_param_step_size=initial_param_step_size,
        final_param_step_size=final.chains.param_step_size,
        final_parameter_chol=final.chains.parameter_chol[0],
        initial_latent_delta=initial_latent_delta,
        final_latent_delta=final.chains.latent_delta,
        profile=run.profile
        if run.profile is not None
        else FitProfileReport(
            phase="compile_only",
            repetitions=0,
            chains=num_chains,
            time_steps=num_steps,
            particle_dimensions=dimensions,
        )
        if execution_context.profile
        else None,
        compiled_step=plan.compiled_step,
        first_step_seconds=run.first_step_seconds,
        sampling_loop_seconds=run.sampling_loop_seconds,
        final_state=retain_state(final),
        random_key=final.random_key,
        last_parameters=final.chains.position,
        last_paths=final.chains.latent_trajectory,
    )
