"""Dual-averaging NUTS on the exact joint parameter and latent-trajectory density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.hmc import HMCState, hmc

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import JointTuning, SamplerCheckpoint
from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.reports.posterior_diagnostics import MCMCEvidence, NutsDADiagnostics
from grassmoth.artifacts.sampling import (
    FixedTrajectorySpec,
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
from grassmoth.models.ssm.inference.samplers.progress import inference_phase
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
from grassmoth.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from jax.stages import Compiled

    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.compute.programs import ProgramCompiler, ProgramPreparer
    from grassmoth.models.ssm.inference.joint_coordinates import JointChart
    from grassmoth.models.ssm.inference.reparameterization import Strategy
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel


@dataclass(frozen=True)
class NutsDARun:
    """Aligned retained chains and adaptation telemetry from joint NUTS."""

    positions: jax.Array
    paths: jax.Array
    observation_log_probs: jax.Array
    phase_extra_fields: Mapping[str, Mapping[str, jax.Array]]
    final_step_size: jax.Array
    compiled_step: Compiled
    checkpoint: SamplerCheckpoint

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
    prepare_program: ProgramPreparer,
    compiler: ProgramCompiler,
    sampler: SamplerSpec,
    compute_budget: ComputeBudget,
    key: jax.Array,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    continuation: SamplerCheckpoint | None = None,
) -> NutsDARun | InitializationFailure:
    """Advance independent chains together, then retain only post-adaptation joint draws.

    NumPyro owns dual averaging, windowed diagonal mass adaptation and the NUTS
    transition. A single vectorized executable advances every chain on either CPU
    or GPU; no chain is serialized to accommodate a smaller memory allocation.
    """
    options = sampler.algorithm
    assert isinstance(options, NutsDASpec)
    coordinates = build_joint_coordinates(target, options.parameterization, compiler=compiler)
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

    def potential(runtime_chart: JointChart) -> Callable[[jax.Array], jax.Array]:
        bound = coordinates.bind(runtime_chart)
        return lambda point: -bound.log_density(point)

    initialize, raw_step = hmc(
        potential_fn_gen=potential,
        algo="HMC" if isinstance(trajectory, FixedTrajectorySpec) else "NUTS",
    )

    def step(state: HMCState, runtime_chart: JointChart) -> HMCState:
        if isinstance(trajectory, NutsTrajectorySpec):
            return raw_step(state, model_args=(runtime_chart,))
        # A mixture of exact Metropolis-adjusted HMC kernels avoids resonances.
        # The ±20% jitter belongs to this proposal, never to frozen base tuning.
        kernel_key, jitter_key = jax.random.split(state.rng_key)
        multiplier = jax.random.uniform(jitter_key, (), minval=0.8, maxval=1.2)
        varied = state._replace(
            rng_key=kernel_key,
            adapt_state=state.adapt_state._replace(
                step_size=state.adapt_state.step_size * multiplier
            ),
        )
        moved = raw_step(varied, model_args=(runtime_chart,))
        adaptation = jax.lax.cond(
            state.i < num_warmup, lambda: moved.adapt_state, lambda: state.adapt_state
        )
        return moved._replace(adapt_state=adaptation)

    def initialize_chain(
        point: jax.Array, chain_key: jax.Array, runtime_chart: JointChart
    ) -> HMCState:
        return initialize(
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
            model_args=(runtime_chart,),
        )

    with inference_phase(
        "nuts_da", "initializing_chains", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        starts = (
            jnp.broadcast_to(initial, (sampler.num_chains, initial.size))
            if continuation is None
            else jax.vmap(coordinates.encode)(
                jnp.asarray(continuation.parameters.values), jnp.asarray(continuation.paths.values)
            )
        )
        if continuation is not None:
            key = jax.random.wrap_key_data(jnp.asarray(continuation.random_key.values))
        chain_keys = jax.random.split(key, sampler.num_chains)
        states = compiler(
            jax.vmap(initialize_chain, in_axes=(0, 0, None)), starts, chain_keys, chart
        ).call(starts, chain_keys, chart)
        if continuation is not None and sampler.num_warmup == 0:
            states = restore_state(states, continuation.native_state)

            # Persisted scores are caches of the current executable's density,
            # not transition identity. Rebind them at the restored native point
            # while retaining the same chart, random stream and frozen tuning.
            def evaluate(point: jax.Array, runtime_chart: JointChart) -> jax.Array:
                return potential(runtime_chart)(point)

            energy, gradient = compiler(
                jax.vmap(jax.value_and_grad(evaluate), in_axes=(0, None)), states.z, chart
            ).call(states.z, chart)
            states = states._replace(potential_energy=energy, z_grad=gradient)
        jax.block_until_ready(states)
    if not all(
        np.isfinite(np.asarray(value)).all() for value in (states.potential_energy, states.z_grad)
    ):
        return InitializationFailure.unavailable(
            "NUTS-DA requires a finite exact joint density and gradient at its starting point"
        )

    def project(point: jax.Array, runtime_chart: JointChart) -> dict[str, jax.Array]:
        parameters, path = coordinates.bind(runtime_chart).decode(point)
        return {"positions": parameters, "paths": path}

    def step_factory(
        _batch_size: int,
    ) -> Callable[
        [tuple[HMCState, jax.Array, JointChart], jax.Array],
        tuple[tuple[HMCState, jax.Array, JointChart], dict[str, jax.Array]],
    ]:
        def sweep(
            carry: tuple[HMCState, jax.Array, JointChart],
            posterior: jax.Array,
        ) -> tuple[tuple[HMCState, jax.Array, JointChart], dict[str, jax.Array]]:
            current, key, runtime_chart = carry
            next_key, step_key = jax.random.split(key)
            current = current._replace(rng_key=jax.random.split(step_key, num_chains))
            updated = jax.vmap(step, in_axes=(0, None))(current, runtime_chart)
            record = dict(_extra_fields(updated, trajectory=trajectory))
            projected = jax.lax.cond(
                posterior,
                lambda: jax.vmap(project, in_axes=(0, None))(updated.z, runtime_chart),
                lambda: {
                    "positions": jnp.zeros(
                        (num_chains, target.initial_position.size), dtype=updated.z.dtype
                    ),
                    "paths": jnp.zeros(
                        (num_chains, *target.independent_mask.shape), dtype=updated.z.dtype
                    ),
                },
            )
            record = {**record, **projected}
            return (updated, next_key, runtime_chart), record

        return sweep

    with inference_phase(
        "nuts_da", "compiling_sweeps", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        plan = compile_projected_sweeps(
            step_factory,
            (states, key, chart),
            num_warmup=sampler.num_warmup,
            num_samples=sampler.num_samples_per_chain,
            max_batch_size=1,
            budget=compute_budget,
            target=target,
            path_dtype=states.z.dtype,
            num_chains=sampler.num_chains,
            clock=clock,
            logger=logger,
            wall_clock=wall_clock,
            label="nuts_da",
            prepare_program=prepare_program,
        )
    run = run_sweeps(
        plan,
        (states, key, chart),
        clock=clock,
        logger=logger,
        wall_clock=wall_clock,
        label="nuts_da",
    )
    states, next_key, _ = run.state
    last_parameters, last_paths = jax.vmap(coordinates.decode)(states.z)
    draw_fields = frozenset(("positions", "paths", "observation_log_probs"))
    posterior = {
        name: value for name, value in run.history.posterior.items() if name not in draw_fields
    }
    all_extra = {
        name: jnp.concatenate((run.history.warmup[name], value), axis=1)
        for name, value in posterior.items()
    }
    return NutsDARun(
        positions=run.history.posterior["positions"],
        paths=run.history.posterior["paths"],
        observation_log_probs=run.history.posterior["observation_log_probs"],
        phase_extra_fields={
            "warmup": run.history.warmup,
            "post_warmup": posterior,
            "all": all_extra,
        },
        final_step_size=states.adapt_state.step_size,
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
    prepare_program: ProgramPreparer,
    compiler: ProgramCompiler,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | None,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | InitializationFailure:
    """Fit the production exact target and expose the shared posterior/report contract."""
    trace_key, sampling_key = jax.random.split(jax.random.key(sampler.seed))
    with inference_phase(
        "nuts_da", "building_target", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        bundle = build_particle_problem(
            priors,
            panel,
            scheme=LATENT_TRANSITION_EULER_MARUYAMA,
            trace_key=trace_key,
            reparam=reparam,
            compiler=compiler,
        )
    started = clock()
    run = run_nuts_da(
        bundle.runtime,
        sampler=sampler,
        compute_budget=compute_budget,
        key=sampling_key,
        clock=clock,
        logger=logger,
        wall_clock=wall_clock,
        continuation=continuation,
        prepare_program=prepare_program,
        compiler=compiler,
    )
    if isinstance(run, InitializationFailure):
        return run
    assert isinstance(run.checkpoint.tuning, JointTuning)
    parameterization_record = run.checkpoint.tuning.parameterization
    if sampler.num_samples_per_chain == 0:
        return UnretainedRun.completed(
            sampler,
            MCMCEvidence(method="nuts_da"),
            run.compiled_step,
            run.phase_extra_fields,
            parameterization=parameterization_record,
        )
    extra = run.phase_extra_fields["post_warmup"]
    mcmc = TrajectoryMCMCResult(
        chain_samples=extract_grouped_public_samples(
            run.positions,
            bundle=bundle,
            num_chains=sampler.num_chains,
            num_samples=sampler.num_samples_per_chain,
            compiler=compiler,
        ),
        chain_extra_fields=extra,
        num_chains=sampler.num_chains,
        num_samples=sampler.num_samples_per_chain,
        backend="nuts_da",
    )
    free_mask = bundle.runtime.independent_mask
    if bundle.exact_constraints is not None:
        free_mask = free_mask & bundle.exact_constraints.free_mask
    return MCMCPosterior.from_run(
        draws=JointPosteriorDraws(
            parameters=mcmc.get_samples(),
            latent_paths=run.paths.reshape((-1, *run.paths.shape[2:])),
            state_ids=tuple(state.id for state in panel.compiled_dynamical_model.states),
        ),
        diagnostics=ProductionDiagnostics(
            checkpoint=run.checkpoint,
            parameterization=parameterization_record,
            mcmc=mcmc,
            observation_log_probs=run.observation_log_probs,
            compiled_step=run.compiled_step,
            public_sites=tuple(sorted(bundle.public_sites)),
            latent_chains=LatentChainEvidence(
                paths=np.asarray(run.paths),
                free_mask=np.asarray(free_mask),
                states=tuple(
                    (ConstructRef(id=s.id), s.name) for s in panel.compiled_dynamical_model.states
                ),
            ),
            exact_observation_rows=jnp.any(
                ~bundle.exact_constraints.free_mask & bundle.runtime.state_density_mask, axis=-1
            )
            if bundle.exact_constraints is not None
            else None,
            sampler_diagnostics=NutsDADiagnostics(
                settings=sampler,
                mcmc_phase_seconds=clock() - started,
                final_step_size=tuple(float(value) for value in run.final_step_size),
                max_tree_depth_fraction=float(jnp.mean(extra["max_tree_depth_reached"]))
                if "max_tree_depth_reached" in extra
                else None,
            ),
            phase_extra_fields=run.phase_extra_fields,
        ),
        evidence=MCMCEvidence(method="nuts_da"),
    )
