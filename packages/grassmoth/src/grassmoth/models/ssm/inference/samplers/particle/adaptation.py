"""Warmup-only dSMC scale adaptation shared by both Particle Gibbs families."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.artifacts.sampling import DsmcSpec, FixedLatentStepSpec
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.inference.mcmc_state import _clip_scale

INITIAL_DELTA = 0.01
ADAPTATION_WINDOW = 100
ADAPTATION_TOLERANCE = 0.05
ADAPTATION_RHO = 0.5
ADAPTATION_RHO_MIN = 1e-3
ADAPTATION_GAMMA = -0.5
DELTA_MIN = 1e-5
DELTA_MAX = 10.0


def initial_delta(spec: DsmcSpec) -> float:
    """Resolve the assigned fixed scale or the adaptation starting value."""
    return (
        spec.latent_step.delta
        if isinstance(spec.latent_step, FixedLatentStepSpec)
        else INITIAL_DELTA
    )


class LatentAdaptation(NamedTuple):
    """Runtime scale-adaptation controls, independent of the proposal structure."""

    enabled: jax.Array
    target: jax.Array


def latent_adaptation(spec: DsmcSpec) -> LatentAdaptation:
    """Resolve the authored scale policy before tracing any transitions."""
    return LatentAdaptation(
        enabled=scalar_operand(value=not isinstance(spec.latent_step, FixedLatentStepSpec)),
        target=scalar_operand(
            value=0.5
            if isinstance(spec.latent_step, FixedLatentStepSpec)
            else spec.latent_step.target
        ),
    )


class DsmcAdaptation(NamedTuple):
    """Per-time scales and the recent accepted/proposed movement counts."""

    delta: jax.Array
    accepted: jax.Array
    proposed: jax.Array
    iteration: jax.Array
    accepted_total: jax.Array
    proposed_total: jax.Array


def initialize_adaptation(delta: jax.Array) -> DsmcAdaptation:
    """Initialize one scale vector or a batch of scale vectors."""
    shape = (*delta.shape[:-1], ADAPTATION_WINDOW, delta.shape[-1])
    window = jnp.zeros(shape, dtype=delta.dtype)
    return DsmcAdaptation(
        delta, window, window, jnp.asarray(0), jnp.zeros_like(delta), jnp.zeros_like(delta)
    )


@jax.jit
def adapt_scales(
    state: DsmcAdaptation,
    settings: LatentAdaptation,
    accepted: jax.Array,
    proposed: jax.Array,
    num_warmup: int | jax.Array,
) -> DsmcAdaptation:
    """Adapt only proposed coordinates; freeze scales and windows after warmup."""
    target = settings.target[0]

    def update() -> DsmcAdaptation:
        slot = state.iteration % ADAPTATION_WINDOW
        accept_window = state.accepted.at[..., slot, :].set(accepted)
        proposal_window = state.proposed.at[..., slot, :].set(proposed)
        accepted_total = state.accepted_total - state.accepted[..., slot, :] + accepted
        count = state.proposed_total - state.proposed[..., slot, :] + proposed
        rate = accepted_total / jnp.maximum(count, 1)
        iteration = state.iteration + 1
        learning_rate = jnp.maximum(
            iteration.astype(state.delta.dtype) ** ADAPTATION_GAMMA * ADAPTATION_RHO,
            ADAPTATION_RHO_MIN,
        )
        change = learning_rate * state.delta * (rate - target) / target
        active = (
            (jnp.abs(rate - target) >= ADAPTATION_TOLERANCE)
            & (iteration > ADAPTATION_WINDOW)
            & (count > 0)
        )
        delta = _clip_scale(
            jnp.where(active, state.delta + change, state.delta),
            min_scale=DELTA_MIN,
            max_scale=DELTA_MAX,
        )
        return DsmcAdaptation(
            delta, accept_window, proposal_window, iteration, accepted_total, count
        )

    adapted: DsmcAdaptation = jax.lax.cond(
        settings.enabled[0] & (state.iteration < num_warmup), update, lambda: state
    )
    return adapted
