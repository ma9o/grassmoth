"""Dual-averaging NUTS on the exact joint parameter and latent-trajectory density."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.hmc import HMCState, hmc
from numpyro.infer.hmc_util import euclidean_kinetic_energy, velocity_verlet

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import JointTuning, SamplerCheckpoint
from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.reports.posterior_diagnostics import NutsDADiagnostics
from grassmoth.artifacts.reports.profiling import FitComponentTiming, FitProfileReport, TimingSpread
from grassmoth.artifacts.sampling import (
    FixedTrajectorySpec,
    JointSmootherSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.continuation import restore_state, retain_state
from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from grassmoth.models.ssm.inference.extraction import extract_grouped_public_samples
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.reparameterization import AutoReparam
from grassmoth.models.ssm.inference.samplers.nuts_blocks import (
    ConditionalNutsBlock,
    JointNutsBlock,
    NutsBlock,
    NutsBlockState,
)
from grassmoth.models.ssm.inference.samplers.particle.conditional import ConditionalPathKernel
from grassmoth.models.ssm.inference.samplers.profiling import REPETITIONS, measure, measure_sweep
from grassmoth.models.ssm.inference.samplers.progress import inference_phase
from grassmoth.models.ssm.inference.samplers.sweep_plan import SweepProgram
from grassmoth.models.ssm.inference.samplers.sweeps import (
    compile_projected_sweeps,
    run_sweeps,
)
from grassmoth.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
    UnretainedRun,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled
    from numpy.typing import NDArray

    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.models.ssm.inference.joint_coordinates import JointChart
    from grassmoth.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel


class NutsChainState(NamedTuple):
    """One Hamiltonian state and the coordinates held by its configured block."""

    hmc: HMCState
    block: NutsBlockState


@dataclass(frozen=True)
class NutsDARun:
    """Aligned retained chains and adaptation telemetry from joint NUTS."""

    positions: jax.Array
    paths: jax.Array | NDArray
    observation_log_probs: jax.Array
    phase_extra_fields: Mapping[str, Mapping[str, jax.Array]]
    final_step_size: jax.Array
    compiled_step: Compiled
    checkpoint: SamplerCheckpoint
    retained_draws: int
    profile: FitProfileReport | None = None

    def __post_init__(self) -> None:
        """Own immutable phase mappings without changing the numerical buffers."""
        freeze_fields(self)


def _extra_fields(
    state: HMCState, *, trajectory: NutsTrajectorySpec | FixedTrajectorySpec
) -> Mapping[str, jax.Array]:
    tree_depth: jax.Array = jnp.ceil(jnp.log2(state.num_steps + 1)).astype(jnp.int32)
    tree_fields: Mapping[str, jax.Array] = (
        {
            "tree_depth": tree_depth,
            "max_tree_depth_reached": jnp.asarray(state.num_steps)
            >= 2**trajectory.max_tree_depth - 1,
        }
        if isinstance(trajectory, NutsTrajectorySpec)
        else {}
    )
    return cast(
        "Mapping[str, jax.Array]",
        {
            **tree_fields,
            "diverging": jnp.asarray(state.diverging),
            "accept_prob": jnp.asarray(state.accept_prob),
            "num_steps": jnp.asarray(state.num_steps),
            "energy": jnp.asarray(state.energy),
            "potential_energy": jnp.asarray(state.potential_energy),
            "step_size": jnp.asarray(state.adapt_state.step_size),
        },
    )


def run_nuts_da(
    target: ParticleTarget,
    *,
    sampler: SamplerSpec,
    start: jax.Array | SamplerCheckpoint,
    execution_context: ModelExecutionContext,
) -> NutsDARun | InitializationFailure:
    """Advance independent chains together, then retain only post-adaptation joint draws.

    ``start`` is a fresh random key or the checkpoint these chains resume.
    NumPyro owns dual averaging, windowed diagonal mass adaptation and the NUTS
    transition. A single vectorized executable advances every chain on either CPU
    or GPU; no chain is serialized to accommodate a smaller memory allocation.
    Heuristic: fixed-step trajectories jitter the base step by uniform [0.8, 1.2];
    Neal (2011), section 5.3.2 motivates randomization but not this width.
    The unused HMC tree-depth argument keeps NumPyro's default of 10.
    """
    continuation = start if isinstance(start, SamplerCheckpoint) else None
    compiler = execution_context.compiler
    clock = execution_context.clock
    options = sampler.algorithm
    assert isinstance(options, NutsDASpec)
    coordinates = build_joint_coordinates(target, options.parameterization, compiler=compiler)
    block: NutsBlock = (
        JointNutsBlock(coordinates)
        if isinstance(options.smoother, JointSmootherSpec)
        else ConditionalNutsBlock(
            coordinates, ConditionalPathKernel(target, options.smoother, sampler.num_warmup)
        )
    )
    parameterization_record = coordinates.record
    if continuation is not None and sampler.num_warmup == 0:
        tuning = continuation.tuning
        if (
            not isinstance(tuning, JointTuning)
            or tuning.parameterization != parameterization_record
        ):
            return InitializationFailure.unavailable(
                "Changing joint parameterization or its compiled mask requires positive warmup"
            )
    # Preserve identical scientific starts across parameterizations, including the original
    # affine default. A fully fixed model retains one independent normal auxiliary.
    centred = build_joint_coordinates(target, compiler=compiler)
    scientific_start = centred.decode(
        jnp.zeros(max(1, centred.dimension), dtype=target.initial_position.dtype)
    )
    initial = coordinates.encode(*scientific_start)
    chart = coordinates.chart
    trajectory = options.trajectory
    num_warmup, num_chains = sampler.num_warmup, sampler.num_chains

    def potential(
        runtime_chart: JointChart, held: NutsBlockState
    ) -> Callable[[jax.Array], jax.Array]:
        return lambda point: -block.log_density(point, held, runtime_chart)

    initialize, raw_step = hmc(
        potential_fn_gen=potential,
        algo="HMC" if isinstance(trajectory, FixedTrajectorySpec) else "NUTS",
    )

    def step(state: HMCState, runtime_chart: JointChart, held: NutsBlockState) -> HMCState:
        if isinstance(trajectory, NutsTrajectorySpec):
            return raw_step(state, model_args=(runtime_chart, held))
        # A mixture of exact Metropolis-adjusted HMC kernels avoids resonances.
        # Neal (2011), section 5.3.2 motivates jitter; ±20% is a heuristic, not
        # a paper or benchmark recommendation. It never changes frozen base tuning.
        kernel_key, jitter_key = jax.random.split(state.rng_key)
        multiplier = jax.random.uniform(jitter_key, (), minval=0.8, maxval=1.2)
        varied = state._replace(
            rng_key=kernel_key,
            adapt_state=state.adapt_state._replace(
                step_size=state.adapt_state.step_size * multiplier
            ),
        )
        moved = raw_step(varied, model_args=(runtime_chart, held))
        adaptation = jax.lax.cond(
            state.i < num_warmup, lambda: moved.adapt_state, lambda: state.adapt_state
        )
        return moved._replace(adapt_state=adaptation)

    def nuts_chain_initializer(
        point: jax.Array, chain_key: jax.Array, runtime_chart: JointChart
    ) -> NutsChainState:
        point, held = block.initialize(point)
        state = initialize(
            point,
            num_warmup=sampler.num_warmup,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=options.target_accept_prob,
            max_tree_depth=trajectory.max_tree_depth
            if isinstance(trajectory, NutsTrajectorySpec)
            else 10,
            num_steps=trajectory.steps if isinstance(trajectory, FixedTrajectorySpec) else None,
            trajectory_length=None,  # pyright: ignore[reportArgumentType] - NumPyro documents None for fixed-step HMC; its unannotated default is a float.
            rng_key=chain_key,
            model_args=(runtime_chart, held),
        )
        return NutsChainState(state, held)

    with inference_phase("nuts_da", "initializing_chains", execution_context=execution_context):
        starts = (
            jnp.broadcast_to(initial, (sampler.num_chains, initial.size))
            if continuation is None
            else jax.vmap(coordinates.encode)(
                jnp.asarray(continuation.parameters.values), jnp.asarray(continuation.paths.values)
            )
        )
        key = (
            jax.random.wrap_key_data(jnp.asarray(start.random_key.values))
            if isinstance(start, SamplerCheckpoint)
            else start
        )
        chain_keys = jax.random.split(key, sampler.num_chains)
        states = compiler(
            jax.vmap(nuts_chain_initializer, in_axes=(0, 0, None)), starts, chain_keys, chart
        ).call(starts, chain_keys, chart)
        if continuation is not None and sampler.num_warmup == 0:
            restored = restore_state(states, continuation.native_state)
            if isinstance(restored, InitializationFailure):
                return restored
            states = restored

            # Persisted scores are caches of the current executable's density,
            # not transition identity. Rebind them at the restored native point
            # while retaining the same chart, random stream and frozen tuning.
            def restored_joint_gradient(
                state: NutsChainState, runtime_chart: JointChart
            ) -> NutsChainState:
                energy, gradient = jax.value_and_grad(potential(runtime_chart, state.block))(
                    state.hmc.z
                )
                return state._replace(
                    hmc=state.hmc._replace(potential_energy=energy, z_grad=gradient)
                )

            states = compiler(
                jax.vmap(restored_joint_gradient, in_axes=(0, None)),
                states,
                chart,
            ).call(states, chart)
        jax.block_until_ready(states)
    if not all(
        np.isfinite(np.asarray(value)).all()
        for value in (states.hmc.potential_energy, states.hmc.z_grad)
    ):
        execution_context.logger.error(
            "Non-finite NUTS start: energy=%s nonfinite_gradient_indices=%s position_finite=%s chart_finite=%s",
            np.asarray(states.hmc.potential_energy).tolist(),
            np.argwhere(~np.isfinite(np.asarray(states.hmc.z_grad)))[:16].tolist(),
            bool(np.isfinite(np.asarray(states.hmc.z)).all()),
            all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(chart)),
        )
        return InitializationFailure.unavailable(
            "NUTS-DA requires a finite exact joint density and gradient at its starting point"
        )

    def project(state: NutsChainState, runtime_chart: JointChart) -> dict[str, jax.Array]:
        parameters, path = coordinates.bind(runtime_chart).decode(
            block.join(state.hmc.z, state.block)
        )
        return {"positions": parameters, "paths": path}

    def smooth_chain(
        state: NutsChainState, runtime_chart: JointChart, iteration: jax.Array
    ) -> tuple[NutsChainState, dict[str, jax.Array]]:
        moved = state.hmc
        held = state.block
        info: dict[str, jax.Array] = {}
        if isinstance(block, ConditionalNutsBlock):
            next_key, path_key = jax.random.split(moved.rng_key)
            held, info = block.smooth(moved.z, held, runtime_chart, path_key, iteration)
            energy, gradient = jax.value_and_grad(potential(runtime_chart, held))(moved.z)
            moved = moved._replace(potential_energy=energy, z_grad=gradient, rng_key=next_key)
        return NutsChainState(moved, held), info

    def move_chain(
        state: NutsChainState, runtime_chart: JointChart
    ) -> tuple[NutsChainState, dict[str, jax.Array]]:
        moved = step(state.hmc, runtime_chart, state.block)
        updated, path_info = smooth_chain(
            NutsChainState(moved, state.block), runtime_chart, state.hmc.i
        )
        return updated, {**_extra_fields(moved, trajectory=trajectory), **path_info}

    def sweep(
        carry: tuple[NutsChainState, jax.Array, JointChart],
        posterior: jax.Array,
    ) -> tuple[tuple[NutsChainState, jax.Array, JointChart], dict[str, jax.Array]]:
        current, key, runtime_chart = carry
        next_key, step_key = jax.random.split(key)
        current = current._replace(
            hmc=current.hmc._replace(rng_key=jax.random.split(step_key, num_chains))
        )
        updated, record = jax.vmap(move_chain, in_axes=(0, None))(current, runtime_chart)
        projected: dict[str, jax.Array] = jax.lax.cond(
            posterior,
            lambda: jax.vmap(project, in_axes=(0, None))(updated, runtime_chart),
            lambda: {
                "positions": jnp.zeros(
                    (num_chains, target.initial_position.size), dtype=updated.hmc.z.dtype
                ),
                "paths": jnp.zeros(
                    (num_chains, *target.independent_mask.shape), dtype=updated.hmc.z.dtype
                ),
            },
        )
        return (updated, next_key, runtime_chart), {**record, **projected}

    with inference_phase("nuts_da", "compiling_sweeps", execution_context=execution_context):
        plan = compile_projected_sweeps(
            SweepProgram(sweep),
            (states, key, chart),
            execution_context=execution_context,
            sampler=sampler,
            target=target,
        )

    def profile_state(
        current: tuple[NutsChainState, jax.Array, JointChart], phase: CompiledSweepPhase
    ) -> FitProfileReport:
        durations = measure_sweep(current, phase, clock)
        frozen, random_key, runtime_chart = current
        _, step_key = jax.random.split(random_key)
        frozen = frozen._replace(
            hmc=frozen.hmc._replace(rng_key=jax.random.split(step_key, num_chains))
        )

        def profile_nuts_transition(state: NutsChainState, chart: JointChart) -> HMCState:
            return jax.vmap(step, in_axes=(0, None, 0))(state.hmc, chart, state.block)

        def profile_nuts_gradient(state: NutsChainState, chart: JointChart) -> object:
            return jax.vmap(lambda point, held: jax.value_and_grad(potential(chart, held))(point))(
                state.hmc.z, state.block
            )

        def profile_nuts_decode(state: NutsChainState, chart: JointChart) -> object:
            return jax.vmap(project, in_axes=(0, None))(state, chart)

        transition = compiler(profile_nuts_transition, frozen, runtime_chart)
        gradient = compiler(profile_nuts_gradient, frozen, runtime_chart)
        decode = compiler(profile_nuts_decode, frozen, runtime_chart)
        transition_times, moved = measure(lambda: transition.call(frozen, runtime_chart), clock)
        gradient_times, _ = measure(lambda: gradient.call(frozen, runtime_chart), clock)
        decode_times, _ = measure(lambda: decode.call(frozen, runtime_chart), clock)
        steps = np.asarray(moved.num_steps)

        def profile_nuts_integrator(
            state: NutsChainState, chart: JointChart, count: jax.Array
        ) -> object:
            """Time the same leapfrog primitive without tree building, on a frozen copy.

            Each chain uses the profile transition's maximum step count. Momentum
            is drawn from its adapted diagonal Gaussian. This probe measures
            integrator work, not an alternative posterior transition.
            """

            def integrate(chain: NutsChainState) -> object:
                hmc_state = chain.hmc
                initialize_integrator, leapfrog = velocity_verlet(
                    potential(chart, chain.block),
                    euclidean_kinetic_energy,
                )
                momentum = hmc_state.adapt_state.mass_matrix_sqrt * jax.random.normal(
                    hmc_state.rng_key, hmc_state.z.shape, hmc_state.z.dtype
                )
                initial = initialize_integrator(
                    hmc_state.z, momentum, hmc_state.potential_energy, hmc_state.z_grad
                )
                return jax.lax.fori_loop(
                    0,
                    count,
                    lambda _, carry: leapfrog(
                        hmc_state.adapt_state.step_size,
                        hmc_state.adapt_state.inverse_mass_matrix,
                        carry,
                    ),
                    initial,
                )

            return jax.vmap(integrate)(state)

        count = jnp.asarray(steps.max())
        integrator = compiler(profile_nuts_integrator, frozen, runtime_chart, count)
        integration_times, _ = measure(lambda: integrator.call(frozen, runtime_chart, count), clock)
        integration_spread = TimingSpread.from_seconds(integration_times)
        execution_context.logger.info(
            "NUTS profile integrator (%d steps): median=%.3f ms range=[%.3f, %.3f] ms",
            int(steps.max()),
            integration_spread.median_ms,
            integration_spread.min_ms,
            integration_spread.max_ms,
        )
        for name, values in (("density_gradient", gradient_times), ("path_decode", decode_times)):
            spread = TimingSpread.from_seconds(values)
            execution_context.logger.info(
                "NUTS profile %s: median=%.3f ms range=[%.3f, %.3f] ms",
                name,
                spread.median_ms,
                spread.min_ms,
                spread.max_ms,
            )
        execution_context.logger.info(
            "NUTS profile tree steps: mean=%.3f max=%d min=%d; active-work fraction=%.6f",
            float(steps.mean()),
            int(steps.max()),
            int(steps.min()),
            float(steps.mean() / max(1, steps.max())),
        )
        components = (
            FitComponentTiming(component="full_sweep", timing=TimingSpread.from_seconds(durations)),
            FitComponentTiming(
                component="parameter_move", timing=TimingSpread.from_seconds(transition_times)
            ),
        )
        split_total = None
        if isinstance(block, ConditionalNutsBlock):

            def profile_nuts_smoother(state: NutsChainState, chart: JointChart) -> object:
                return jax.vmap(smooth_chain, in_axes=(0, None, 0))(state, chart, frozen.hmc.i)

            after_parameters = NutsChainState(moved, frozen.block)
            smoother = compiler(profile_nuts_smoother, after_parameters, runtime_chart)
            path_times, _ = measure(lambda: smoother.call(after_parameters, runtime_chart), clock)
            components += (
                FitComponentTiming(
                    component="path_smoother", timing=TimingSpread.from_seconds(path_times)
                ),
            )
            split_total = TimingSpread.from_seconds(
                tuple(a + b for a, b in zip(transition_times, path_times, strict=True))
            )
        return FitProfileReport(
            phase="after_warmup" if sampler.num_warmup else "after_first_sweep",
            repetitions=REPETITIONS,
            chains=sampler.num_chains,
            time_steps=len(target.times),
            components=components,
            split_total=split_total,
        )

    run = run_sweeps(
        plan,
        (states, key, chart),
        execution_context=execution_context,
        profile=profile_state if execution_context.profile else None,
    )
    states, next_key, _ = run.state
    last = jax.vmap(project, in_axes=(0, None))(states, chart)
    last_parameters, last_paths = last["positions"], last["paths"]
    draw_fields = frozenset(("positions", "paths", "observation_log_probs"))
    posterior = {
        name: value for name, value in run.history.posterior.items() if name not in draw_fields
    }
    all_extra = {
        name: jnp.concatenate((run.history.warmup[name], value), axis=1)
        for name, value in posterior.items()
    }
    return NutsDARun(
        retained_draws=run.retained_draws,
        positions=run.history.posterior["positions"],
        paths=run.history.posterior["paths"],
        observation_log_probs=run.history.posterior["observation_log_probs"],
        phase_extra_fields={
            "warmup": run.history.warmup,
            "post_warmup": posterior,
            "all": all_extra,
        },
        final_step_size=states.hmc.adapt_state.step_size,
        profile=run.profile
        if run.profile is not None
        else FitProfileReport(
            phase="compile_only",
            repetitions=0,
            chains=sampler.num_chains,
            time_steps=len(target.times),
        )
        if execution_context.profile
        else None,
        compiled_step=plan.compiled_step,
        checkpoint=SamplerCheckpoint(
            settings=sampler,
            parameters=NumericalArray.from_numpy(np.asarray(last_parameters)),
            paths=NumericalArray.from_numpy(np.asarray(last_paths)),
            random_key=NumericalArray.from_numpy(np.asarray(jax.random.key_data(next_key))),
            native_state=retain_state(states),
            tuning=JointTuning(parameterization=parameterization_record),
        ),
    )


def fit_nuts_da(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    sampler: SamplerSpec,
    execution_context: ModelExecutionContext,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | InitializationFailure:
    """Fit the production exact target and expose the shared posterior/report contract."""
    compiler = execution_context.compiler
    clock = execution_context.clock
    trace_key, sampling_key = jax.random.split(jax.random.key(sampler.seed))
    with inference_phase("nuts_da", "building_target", execution_context=execution_context):
        bundle = build_particle_problem(
            priors,
            panel,
            trace_key=trace_key,
            # Fully non-centred; a strategy caches per-site choices, so each fit owns one.
            reparam=AutoReparam(centered=0.0),
            compiler=compiler,
        )
    started = clock()
    run = run_nuts_da(
        bundle.runtime,
        sampler=sampler,
        start=sampling_key if continuation is None else continuation,
        execution_context=execution_context,
    )
    if isinstance(run, InitializationFailure):
        return run
    assert isinstance(run.checkpoint.tuning, JointTuning)
    if sampler.num_samples_per_chain == 0:
        return UnretainedRun.completed(
            sampler,
            run.compiled_step,
            run.phase_extra_fields,
            profile=run.profile,
        )
    free_mask = bundle.runtime.state_density_mask
    if bundle.exact_constraints is not None:
        free_mask = free_mask & bundle.exact_constraints.free_mask
    with inference_phase("nuts_da", "latent_diagnostics", execution_context=execution_context):
        latent_chains = LatentChainEvidence.from_draws(
            paths=run.paths,
            free_mask=np.asarray(free_mask),
            states=tuple(
                (ConstructRef(id=s.id), s.name) for s in panel.compiled_dynamical_model.states
            ),
        )
    run = replace(run, paths=latent_chains.paths)
    extra = run.phase_extra_fields["post_warmup"]
    mcmc = TrajectoryMCMCResult(
        chain_samples=extract_grouped_public_samples(
            run.positions,
            bundle=bundle,
            num_chains=sampler.num_chains,
            num_samples=run.retained_draws,
            compiler=compiler,
        ),
        chain_extra_fields=extra,
        num_chains=sampler.num_chains,
        num_samples=run.retained_draws,
    )
    return MCMCPosterior.from_run(
        draws=JointPosteriorDraws(
            parameters=mcmc.get_samples(),
            latent_paths=run.paths.reshape((-1, *run.paths.shape[2:])),
            state_ids=tuple(state.id for state in panel.compiled_dynamical_model.states),
        ),
        diagnostics=ProductionDiagnostics(
            profile=run.profile,
            checkpoint=run.checkpoint,
            mcmc=mcmc,
            observation_log_probs=run.observation_log_probs,
            compiled_step=run.compiled_step,
            public_sites=tuple(sorted(bundle.public_sites)),
            latent_chains=latent_chains,
            exact_observation_rows=jnp.any(
                ~bundle.exact_constraints.free_mask & bundle.runtime.state_density_mask, axis=-1
            )
            if bundle.exact_constraints is not None
            else None,
            sampler_diagnostics=NutsDADiagnostics(
                settings=sampler,
                divergences=int(jnp.sum(extra["diverging"])),
                mean_accept_prob=float(jnp.mean(extra["accept_prob"])),
                mean_num_steps=float(jnp.mean(extra["num_steps"])),
                mcmc_phase_seconds=clock() - started,
                final_step_size=tuple(float(value) for value in run.final_step_size),
                max_tree_depth_fraction=float(jnp.mean(extra["max_tree_depth_reached"]))
                if "max_tree_depth_reached" in extra
                else None,
            ),
            phase_extra_fields=run.phase_extra_fields,
        ),
    )
