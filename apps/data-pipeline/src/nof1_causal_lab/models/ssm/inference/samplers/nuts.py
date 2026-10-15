"""Dual-averaging NUTS on the exact joint parameter and latent-trajectory density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.hmc import HMCState, hmc

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import MCMCEvidence, NutsDADiagnostics
from nof1_causal_lab.artifacts.sampling import NutsDASpec, SamplerSpec
from nof1_causal_lab.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from nof1_causal_lab.models.ssm.inference.extraction import extract_grouped_public_samples
from nof1_causal_lab.models.ssm.inference.joint_coordinates import build_joint_coordinates
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.inference.samplers.progress import inference_phase
from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import compile_sweeps
from nof1_causal_lab.models.ssm.inference.samplers.sweeps import run_sweeps
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.reparameterization import Strategy
    from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import SweepStage
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel


@dataclass(frozen=True)
class NutsDARun:
    """Aligned retained chains and adaptation telemetry from joint NUTS."""

    positions: jax.Array
    paths: jax.Array | None
    observation_log_probs: jax.Array
    phase_extra_fields: Mapping[str, Mapping[str, jax.Array]]
    final_step_size: jax.Array
    compiled_step: Compiled

    def __post_init__(self) -> None:
        """Own immutable phase mappings without changing the numerical buffers."""
        freeze_fields(self)


def _extra_fields(state: HMCState, *, max_tree_depth: int) -> Mapping[str, jax.Array]:
    return {
        "diverging": jnp.asarray(state.diverging),
        "accept_prob": jnp.asarray(state.accept_prob),
        "num_steps": jnp.asarray(state.num_steps),
        "tree_depth": jnp.ceil(jnp.log2(state.num_steps + 1)).astype(jnp.int32),
        "max_tree_depth_reached": jnp.asarray(state.num_steps) >= 2**max_tree_depth - 1,
        "energy": jnp.asarray(state.energy),
        "potential_energy": jnp.asarray(state.potential_energy),
        "step_size": jnp.asarray(state.adapt_state.step_size),
    }


def run_nuts_da(
    target: ParticleTarget,
    *,
    sampler: SamplerSpec,
    compute_budget: ComputeBudget,
    key: jax.Array,
    clock: Callable[[], float],
) -> NutsDARun | InitializationFailure:
    """Advance independent chains together, then retain only post-adaptation joint draws.

    NumPyro owns dual averaging, windowed diagonal mass adaptation and the NUTS
    transition. A single vectorized executable advances every chain on either CPU
    or GPU; no chain is serialized to accommodate a smaller memory allocation.
    """
    options = sampler.algorithm
    assert isinstance(options, NutsDASpec)
    coordinates = build_joint_coordinates(target)
    # A fully fixed model can still use one independent normal auxiliary.
    initial = jnp.zeros(max(1, coordinates.dimension), dtype=target.initial_position.dtype)
    initialize, step = hmc(lambda point: -coordinates.log_density(point), algo="NUTS")

    def initialize_chain(chain_key: jax.Array) -> HMCState:
        return initialize(
            initial,
            num_warmup=sampler.num_warmup,
            step_size=options.initial_step_size,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=options.target_accept_prob,
            max_tree_depth=options.max_tree_depth,
            rng_key=chain_key,
        )

    with inference_phase("nuts_da", "initializing_chains", clock=clock):
        states = jax.jit(jax.vmap(initialize_chain))(jax.random.split(key, sampler.num_chains))
        jax.block_until_ready(states)
    if not all(
        np.isfinite(np.asarray(value)).all() for value in (states.potential_energy, states.z_grad)
    ):
        return InitializationFailure.unavailable(
            "NUTS-DA requires a finite exact joint density and gradient at its starting point"
        )

    def project(point: jax.Array) -> dict[str, jax.Array]:
        parameters, path = coordinates.decode(point)
        context = target.context(parameters, target.times)
        record = {
            "positions": parameters,
            "observation_log_probs": jnp.asarray(
                target.observation_log_probs(context, path, target.observations)
            ),
        }
        if sampler.retain_latent_paths:
            record = {**record, "paths": path}
        return record

    def step_factory(
        _batch_size: int, stage: SweepStage
    ) -> Callable[[HMCState], tuple[HMCState, dict[str, jax.Array]]]:
        def sweep(current: HMCState) -> tuple[HMCState, dict[str, jax.Array]]:
            updated = jax.vmap(step)(current)
            record = dict(_extra_fields(updated, max_tree_depth=options.max_tree_depth))
            if stage == "posterior":
                record = {**record, **jax.vmap(project)(updated.z)}
            return updated, record

        return sweep

    with inference_phase("nuts_da", "compiling_sweeps", clock=clock):
        plan = compile_sweeps(
            step_factory,
            states,
            num_warmup=sampler.num_warmup,
            num_samples=sampler.num_samples_per_chain,
            max_batch_size=1,
            budget=compute_budget,
        )
    run = run_sweeps(plan, states, clock=clock, label="nuts_da")
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
        paths=run.history.posterior["paths"] if sampler.retain_latent_paths else None,
        observation_log_probs=run.history.posterior["observation_log_probs"],
        phase_extra_fields={
            "warmup": run.history.warmup,
            "post_warmup": posterior,
            "all": all_extra,
        },
        final_step_size=run.state.adapt_state.step_size,
        compiled_step=plan.compiled_step,
    )


def fit_nuts_da(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | None,
    clock: Callable[[], float],
) -> MCMCPosterior | InitializationFailure:
    """Fit the production exact target and expose the shared posterior/report contract."""
    trace_key, sampling_key = jax.random.split(jax.random.key(sampler.seed))
    with inference_phase("nuts_da", "building_target", clock=clock):
        bundle = build_particle_problem(
            priors,
            panel,
            scheme=LATENT_TRANSITION_EULER_MARUYAMA,
            trace_key=trace_key,
            reparam=reparam,
        )
    started = clock()
    run = run_nuts_da(
        bundle.runtime,
        sampler=sampler,
        compute_budget=compute_budget,
        key=sampling_key,
        clock=clock,
    )
    if isinstance(run, InitializationFailure):
        return run
    extra = run.phase_extra_fields["post_warmup"]
    mcmc = TrajectoryMCMCResult(
        chain_samples=extract_grouped_public_samples(
            run.positions,
            bundle=bundle,
            num_chains=sampler.num_chains,
            num_samples=sampler.num_samples_per_chain,
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
            latent_paths=run.paths.reshape((-1, *run.paths.shape[2:]))
            if run.paths is not None
            else None,
            state_ids=tuple(state.id for state in panel.compiled_dynamical_model.states),
        ),
        diagnostics=ProductionDiagnostics(
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
            )
            if run.paths is not None
            else None,
            exact_observation_rows=jnp.any(
                ~bundle.exact_constraints.free_mask & bundle.runtime.state_density_mask, axis=-1
            )
            if bundle.exact_constraints is not None
            else None,
            sampler_diagnostics=NutsDADiagnostics(
                settings=sampler,
                mcmc_phase_seconds=clock() - started,
                final_step_size=tuple(float(value) for value in run.final_step_size),
                max_tree_depth_fraction=float(jnp.mean(extra["max_tree_depth_reached"])),
            ),
            phase_extra_fields=run.phase_extra_fields,
        ),
        evidence=MCMCEvidence(method="nuts_da"),
    )
