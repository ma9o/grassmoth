"""Blocked parameter-conditional and dSMC trajectory updates.

The parameter-then-trajectory composition is Particle Gibbs (Andrieu, Doucet
and Holenstein, 2010, section 2.4.3). The conditional particle update uses the
dSMC tree from https://arxiv.org/abs/2202.02264, with the shared auxiliary laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    DsmcAdaptation,
    adapt_scales,
    initial_delta,
    initialize_adaptation,
)
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    resolve_particle_diagnostic_metrics,
)
from grassmoth.models.ssm.inference.samplers.particle.dsmc import step as dsmc_step
from grassmoth.models.ssm.inference.samplers.particle.math import _masked_mean
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from grassmoth.artifacts.sampling import DsmcSpec
    from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
    from grassmoth.models.ssm.inference.samplers.pgibbs.parameters import ParameterKernel
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


class ParticleGibbsState[State](NamedTuple):
    """A joint draw and separate parameter and trajectory adaptation states."""

    parameters: State
    path: jax.Array
    adaptation: DsmcAdaptation
    log_posterior: jax.Array


class ParticleGibbsStep[State](Protocol):
    """A blocked transition with a static dSMC concurrency limit."""

    def __call__(
        self, state: ParticleGibbsState[State], key: jax.Array, *, batch_size: int | None = None
    ) -> tuple[ParticleGibbsState[State], Mapping[str, jax.Array]]:
        """Advance once with all nodes parallel unless a batch limit is supplied."""
        ...


@dataclass(frozen=True)
class ParticleGibbsKernel[State]:
    """Pure blocked transitions sharing a fixed parameter-kernel implementation."""

    initialize: Callable[[jax.Array, jax.Array, jax.Array], ParticleGibbsState[State]]
    step: ParticleGibbsStep[State]
    parameters: ParameterKernel[State]
    num_particles: int


def build_particle_gibbs_kernel[State](
    target: ParticleTarget,
    spec: DsmcSpec,
    parameters: ParameterKernel[State],
    *,
    num_particles: int,
    num_warmup: int,
    paid_mix: PaidMixture | None,
    diagnostic_metrics_all: bool = False,
) -> ParticleGibbsKernel[State]:
    """Condition the common dSMC target on the newly updated parameter value."""
    free_mask = target.independent_mask
    if target.exact_constraints is not None:
        free_mask = free_mask & target.exact_constraints.free_mask
    setup = build_smoother_setup(
        target,
        spec,
        num_particles=num_particles,
        num_parameter_particles=1,
        free_mask=free_mask,
        paid_mix=paid_mix,
        diagnostic_metrics=resolve_particle_diagnostic_metrics(
            diagnostic_metrics_all=diagnostic_metrics_all, diagnostic_metrics=()
        ),
    )

    def initialize(
        position: jax.Array, path: jax.Array, key: jax.Array
    ) -> ParticleGibbsState[State]:
        context = target.context(position, target.times)
        path = target.replay_path(context, path)
        delta = jnp.full((len(path),), initial_delta(spec), dtype=path.dtype)
        return ParticleGibbsState(
            parameters.initialize(position, path, key),
            path,
            initialize_adaptation(delta),
            target.log_posterior(position, path, target.observations, target.times),
        )

    def step(
        state: ParticleGibbsState[State], key: jax.Array, *, batch_size: int | None = None
    ) -> tuple[ParticleGibbsState[State], Mapping[str, jax.Array]]:
        parameter_key, trajectory_key = jax.random.split(key)
        updated_parameters, parameter_info = parameters.step(
            state.parameters, state.path, state.log_posterior, parameter_key
        )
        position = parameters.position(updated_parameters)
        context = target.context(position, target.times)
        reference = target.replay_path(context, state.path)
        smoother_context = build_smoother_context(
            setup,
            position[None],
            jnp.zeros(1, dtype=position.dtype),
            reference=reference,
            delta=state.adaptation.delta,
            trajectory_dtype=state.log_posterior.dtype,
        )

        def correction(path: jax.Array, local_prior: jax.Array) -> jax.Array:
            return target.replay_correction(context, path, local_prior[0])[None]

        smooth = dsmc_step(
            smoother_context,
            trajectory_key,
            reference,
            correction if target.state_space.requires_replay else None,
            batch_size=batch_size,
        )
        path = target.replay_path(context, smooth.latent_path)
        proposed = jnp.any(smooth.proposal_mask, axis=-1)
        moved = jnp.any((path != reference) & smooth.proposal_mask, axis=-1)
        displacement = jnp.square(path - reference)
        tuning = adapt_scales(state.adaptation, spec, moved, proposed, num_warmup)
        log_posterior = smooth.label_scores.log_posteriors[0].astype(state.log_posterior.dtype)
        return ParticleGibbsState(updated_parameters, path, tuning, log_posterior), {
            **parameter_info,
            **smooth.diagnostics,
            "latent_accept_prob": _masked_mean(moved.astype(path.dtype), proposed),
            "latent_frozen_frac": _masked_mean((path == reference).astype(path.dtype), free_mask),
            "latent_move_rms": jnp.sqrt(_masked_mean(displacement, free_mask)),
            "latent_move_rms_per_t": jnp.sqrt(_masked_mean(displacement, free_mask, axis=-1)),
            "latent_delta": state.adaptation.delta,
        }

    return ParticleGibbsKernel(initialize, step, parameters, num_particles)
