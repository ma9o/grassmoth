"""Dual-averaging NUTS on the exact joint parameter and latent-trajectory density."""

from __future__ import annotations

import logging
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
from nof1_causal_lab.compute.batching import compiled_memory
from nof1_causal_lab.compute.budget import MemoryDemand
from nof1_causal_lab.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from nof1_causal_lab.models.ssm.inference.extraction import extract_grouped_public_samples
from nof1_causal_lab.models.ssm.inference.joint_coordinates import build_joint_coordinates
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
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
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel

logger = logging.getLogger(__name__)


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

    states = jax.jit(jax.vmap(initialize_chain))(jax.random.split(key, sampler.num_chains))
    if not all(
        np.isfinite(np.asarray(value)).all() for value in (states.potential_energy, states.z_grad)
    ):
        return InitializationFailure.unavailable(
            "NUTS-DA requires a finite exact joint density and gradient at its starting point"
        )

    compiled_step = jax.jit(jax.vmap(step)).lower(states).compile()
    memory = compiled_memory(compiled_step)
    path_size = coordinates.path_center.size if sampler.retain_latent_paths else 0
    retained_size = (
        sampler.num_chains
        * sampler.num_samples_per_chain
        * (coordinates.parameter_dimension + path_size + target.times.size)
    )
    telemetry_size = sampler.num_chains * (sampler.num_warmup + sampler.num_samples_per_chain) * 8
    # History, stacking and the returned device buffers coexist during projection.
    retained_bytes = (retained_size + telemetry_size) * initial.dtype.itemsize
    demand = MemoryDemand(
        host_bytes=memory.host_bytes + 2 * retained_bytes,
        device_bytes=memory.device_bytes + retained_bytes,
    )
    if compute_budget.capacity(demand, limit=1) == 0:
        raise MemoryError(
            "Parallel NUTS chains and retained draws exceed the compute memory budget"
        )

    def project(point: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        parameters, path = coordinates.decode(point)
        context = target.context(parameters, target.times)
        return (
            parameters,
            path,
            jnp.asarray(target.observation_log_probs(context, path, target.observations)),
        )

    project_chains = jax.jit(jax.vmap(project))
    telemetry = jax.jit(
        lambda current: _extra_fields(current, max_tree_depth=options.max_tree_depth)
    )
    positions: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    extras: list[Mapping[str, np.ndarray]] = []
    started = clock()
    total = sampler.num_warmup + sampler.num_samples_per_chain
    logger.info(
        "nuts_da: %d vectorized chains, %d joint coordinates, %d warmup, %d retained per chain",
        sampler.num_chains,
        coordinates.dimension,
        sampler.num_warmup,
        sampler.num_samples_per_chain,
    )
    for index in range(total):
        states = compiled_step(states)
        extras.append(jax.device_get(telemetry(states)))
        if index >= sampler.num_warmup:
            parameters, path, observation_log_probs = jax.device_get(project_chains(states.z))
            positions.append(parameters)
            log_probs.append(observation_log_probs)
            if sampler.retain_latent_paths:
                paths.append(path)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == total:
            logger.info("nuts_da: step %d/%d, %.1fs elapsed", index + 1, total, clock() - started)

    def stack(history: list[np.ndarray]) -> jax.Array:
        return jnp.asarray(np.stack(history, axis=1))

    all_extra = {name: stack([item[name] for item in extras]) for name in extras[0]}
    return NutsDARun(
        positions=stack(positions),
        paths=stack(paths) if sampler.retain_latent_paths else None,
        observation_log_probs=stack(log_probs),
        phase_extra_fields={
            "warmup": {name: value[:, : sampler.num_warmup] for name, value in all_extra.items()},
            "post_warmup": {
                name: value[:, sampler.num_warmup :] for name, value in all_extra.items()
            },
            "all": all_extra,
        },
        final_step_size=states.adapt_state.step_size,
        compiled_step=compiled_step,
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
