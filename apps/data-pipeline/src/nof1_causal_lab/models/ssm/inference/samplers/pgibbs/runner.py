"""Vectorized Particle Gibbs chains and aligned retained posterior draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.sampling import FixedLatentStepSpec, ParticleGibbsSpec
from nof1_causal_lab.models.ssm.inference.continuation import restore_state, retain_state
from nof1_causal_lab.models.ssm.inference.samplers.progress import inference_phase
from nof1_causal_lab.models.ssm.inference.samplers.sweeps import (
    compile_projected_sweeps,
    run_sweeps,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled

    from nof1_causal_lab.artifacts.arrays import NumericalArray
    from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.kernel import (
        ParticleGibbsKernel,
        ParticleGibbsState,
    )
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.inference.warmup.initialization import ParticleInitialization


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
    final_state: tuple[NumericalArray, ...]
    random_key: jax.Array
    last_parameters: jax.Array
    last_paths: jax.Array

    def __post_init__(self) -> None:
        """Freeze telemetry mappings while retaining the native array buffers."""
        freeze_fields(self)


def run_particle_gibbs[State](
    target: ParticleTarget,
    kernel: ParticleGibbsKernel[State],
    *,
    sampler: SamplerSpec,
    initialization: ParticleInitialization,
    compute_budget: ComputeBudget,
    key: jax.Array,
    clock: Callable[[], float],
    continuation: SamplerCheckpoint | None = None,
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
    if continuation is not None:
        key = jax.random.wrap_key_data(jnp.asarray(continuation.random_key.values))
        if sampler.num_warmup == 0:
            states = restore_state(states, continuation.native_state)
            options = sampler.algorithm
            assert isinstance(options, ParticleGibbsSpec)
            scale = options.dsmc.latent_step
            if isinstance(scale, FixedLatentStepSpec):
                states = states._replace(
                    adaptation=states.adaptation._replace(
                        delta=jnp.full_like(states.adaptation.delta, scale.delta)
                    )
                )
    finite_parameters = jax.vmap(kernel.parameters.is_finite)(states.parameters)
    if (
        not np.isfinite(np.asarray(states.log_posterior)).all()
        or not np.asarray(finite_parameters).all()
    ):
        return InitializationFailure.unavailable(
            "Particle Gibbs requires finite exact joint starting densities and parameter-kernel states"
        )
    initial_delta = states.adaptation.delta
    starting_values = {
        "positions": jax.vmap(kernel.parameters.position)(states.parameters),
        "paths": states.path,
        "log_joint": states.log_posterior,
    }

    def step_factory(
        batch_size: int,
    ) -> Callable[
        [tuple[ParticleGibbsState[State], jax.Array], jax.Array],
        tuple[tuple[ParticleGibbsState[State], jax.Array], dict[str, jax.Array]],
    ]:
        def sweep(
            carry: tuple[ParticleGibbsState[State], jax.Array],
            _posterior: jax.Array,
        ) -> tuple[tuple[ParticleGibbsState[State], jax.Array], dict[str, jax.Array]]:
            current, current_key = carry
            next_key: jax.Array
            step_key: jax.Array
            next_key, step_key = jax.random.split(current_key)
            updated, info = jax.vmap(
                lambda state, draw_key: kernel.step(state, draw_key, batch_size=batch_size)
            )(current, jax.random.split(step_key, sampler.num_chains))
            record = {
                **info,
                "log_posterior": updated.log_posterior,
                "positions": jax.vmap(kernel.parameters.position)(updated.parameters),
                "paths": updated.path,
            }
            next_carry: tuple[ParticleGibbsState[State], jax.Array] = (updated, next_key)
            return next_carry, record

        return sweep

    initial = (states, key)
    with inference_phase("pgibbs", "compiling_sweeps", clock=clock):
        plan = compile_projected_sweeps(
            step_factory,
            initial,
            num_warmup=sampler.num_warmup,
            num_samples=sampler.num_samples_per_chain,
            max_batch_size=max(
                kernel.num_particles, (1 << (len(target.times) - 1).bit_length()) // 2
            ),
            budget=compute_budget,
            target=target,
            path_dtype=states.path.dtype,
            num_chains=sampler.num_chains,
            clock=clock,
            label="pgibbs",
        )
    run = run_sweeps(plan, initial, clock=clock, label="pgibbs")
    states, next_key = run.state
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
        phase_extra_fields={
            "warmup": warmup,
            "post_warmup": posterior,
            "all": extra,
            "initialization" if continuation is None else "continued_start": starting_values,
        },
        initial_latent_delta=initial_delta,
        final_latent_delta=states.adaptation.delta,
        final_param_step_size=jax.vmap(kernel.parameters.step_size)(states.parameters),
        compiled_step=plan.compiled_step,
        final_state=retain_state(states),
        random_key=next_key,
        last_parameters=jax.vmap(kernel.parameters.position)(states.parameters),
        last_paths=states.path,
    )
