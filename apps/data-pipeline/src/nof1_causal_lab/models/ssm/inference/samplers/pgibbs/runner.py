"""Vectorized Particle Gibbs chains and aligned retained posterior draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.models.ssm.inference.samplers.progress import inference_phase
from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import compile_sweeps
from nof1_causal_lab.models.ssm.inference.samplers.sweeps import run_sweeps
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.kernel import (
        ParticleGibbsKernel,
        ParticleGibbsState,
    )
    from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import SweepStage
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.inference.warmup.laps import LapsInitializationResult


@dataclass(frozen=True)
class ParticleGibbsRun:
    """Retained joint draws, warmup telemetry, and frozen proposal scales."""

    positions: jax.Array
    paths: jax.Array | None
    observation_log_probs: jax.Array
    log_posterior: jax.Array
    phase_extra_fields: Mapping[str, Mapping[str, jax.Array]]
    initial_latent_delta: jax.Array
    final_latent_delta: jax.Array
    final_param_step_size: jax.Array
    compiled_step: Compiled

    def __post_init__(self) -> None:
        """Freeze telemetry mappings while retaining the native array buffers."""
        freeze_fields(self)


def run_particle_gibbs[State](
    target: ParticleTarget,
    kernel: ParticleGibbsKernel[State],
    *,
    sampler: SamplerSpec,
    initialization: LapsInitializationResult,
    compute_budget: ComputeBudget,
    key: jax.Array,
    clock: Callable[[], float],
) -> ParticleGibbsRun | InitializationFailure:
    """Advance chains together, retaining aligned draws only after warmup."""
    initial_key, key = jax.random.split(key)
    with inference_phase("pgibbs", "initializing_chains", clock=clock):
        states = jax.jit(jax.vmap(kernel.initialize))(
            initialization.positions,
            initialization.paths,
            jax.random.split(initial_key, sampler.num_chains),
        )
        jax.block_until_ready(states)
    finite_parameters = jax.vmap(kernel.parameters.is_finite)(states.parameters)
    if (
        not np.isfinite(np.asarray(states.log_posterior)).all()
        or not np.asarray(finite_parameters).all()
    ):
        return InitializationFailure.unavailable(
            "Particle Gibbs requires finite exact joint starting densities and parameter-kernel states"
        )
    initial_delta = states.adaptation.delta

    def observation_factors(position: jax.Array, path: jax.Array) -> jax.Array:
        context = target.context(position, target.times)
        return jnp.asarray(target.observation_log_probs(context, path, target.observations))

    def step_factory(
        batch_size: int, stage: SweepStage
    ) -> Callable[
        [tuple[ParticleGibbsState[State], jax.Array]],
        tuple[tuple[ParticleGibbsState[State], jax.Array], dict[str, jax.Array]],
    ]:
        def sweep(
            carry: tuple[ParticleGibbsState[State], jax.Array],
        ) -> tuple[tuple[ParticleGibbsState[State], jax.Array], dict[str, jax.Array]]:
            current, current_key = carry
            next_key: jax.Array
            step_key: jax.Array
            next_key, step_key = jax.random.split(current_key)
            updated, info = jax.vmap(
                lambda state, draw_key: kernel.step(state, draw_key, batch_size=batch_size)
            )(current, jax.random.split(step_key, sampler.num_chains))
            record = {**info, "log_posterior": updated.log_posterior}
            if stage == "posterior":
                position = jax.vmap(kernel.parameters.position)(updated.parameters)
                record = {
                    **record,
                    "positions": position,
                    "observation_log_probs": jax.vmap(observation_factors)(position, updated.path),
                }
                if sampler.retain_latent_paths:
                    record = {**record, "paths": updated.path}
            next_carry: tuple[ParticleGibbsState[State], jax.Array] = (updated, next_key)
            return next_carry, record

        return sweep

    initial = (states, key)
    with inference_phase("pgibbs", "compiling_sweeps", clock=clock):
        plan = compile_sweeps(
            step_factory,
            initial,
            num_warmup=sampler.num_warmup,
            num_samples=sampler.num_samples_per_chain,
            max_batch_size=max(
                kernel.num_particles, (1 << (len(target.times) - 1).bit_length()) // 2
            ),
            budget=compute_budget,
        )
    run = run_sweeps(plan, initial, clock=clock, label="pgibbs")
    states, _ = run.state
    draw_fields = frozenset(("positions", "paths", "observation_log_probs", "log_posterior"))
    warmup = {name: value for name, value in run.history.warmup.items() if name not in draw_fields}
    posterior = {
        name: value for name, value in run.history.posterior.items() if name not in draw_fields
    }
    extra = {
        name: jnp.concatenate((warmup[name], value), axis=1) for name, value in posterior.items()
    }
    return ParticleGibbsRun(
        positions=run.history.posterior["positions"],
        paths=run.history.posterior["paths"] if sampler.retain_latent_paths else None,
        observation_log_probs=run.history.posterior["observation_log_probs"],
        log_posterior=jnp.concatenate(
            (run.history.warmup["log_posterior"], run.history.posterior["log_posterior"]), axis=1
        ),
        phase_extra_fields={"warmup": warmup, "post_warmup": posterior, "all": extra},
        initial_latent_delta=initial_delta,
        final_latent_delta=states.adaptation.delta,
        final_param_step_size=jax.vmap(kernel.parameters.step_size)(states.parameters),
        compiled_step=plan.compiled_step,
    )
