"""Warmup-only dSMC scale adaptation shared by both Particle Gibbs families."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nof1_causal_lab.artifacts.sampling import DsmcSpec, FixedLatentStepSpec
from nof1_causal_lab.models.ssm.inference.mcmc_state import _clip_scale

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


@partial(jax.jit, static_argnames=("spec", "num_warmup"))
def adapt_scales(
    state: DsmcAdaptation,
    spec: DsmcSpec,
    accepted: jax.Array,
    proposed: jax.Array,
    num_warmup: int,
) -> DsmcAdaptation:
    """Adapt only proposed coordinates; freeze scales and windows after warmup."""
    if isinstance(spec.latent_step, FixedLatentStepSpec):
        return state
    target = spec.latent_step.target

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

    adapted: DsmcAdaptation = jax.lax.cond(state.iteration < num_warmup, update, lambda: state)
    return adapted
