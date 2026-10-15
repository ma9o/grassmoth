"""Warmup-only dSMC scale adaptation shared by both Particle Gibbs families."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.mcmc_state import _clip_scale

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.sampling import DsmcSpec


class DsmcAdaptation(NamedTuple):
    """Per-time scales and the recent accepted/proposed movement counts."""

    delta: jax.Array
    accepted: jax.Array
    proposed: jax.Array
    iteration: jax.Array
    accepted_total: jax.Array
    proposed_total: jax.Array


def initialize_adaptation(delta: jax.Array, spec: DsmcSpec) -> DsmcAdaptation:
    """Initialize one scale vector or a batch of scale vectors."""
    shape = (*delta.shape[:-1], spec.amala_adaptation_window, delta.shape[-1])
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

    def update() -> DsmcAdaptation:
        slot = state.iteration % spec.amala_adaptation_window
        accept_window = state.accepted.at[..., slot, :].set(accepted)
        proposal_window = state.proposed.at[..., slot, :].set(proposed)
        accepted_total = state.accepted_total - state.accepted[..., slot, :] + accepted
        count = state.proposed_total - state.proposed[..., slot, :] + proposed
        rate = accepted_total / jnp.maximum(count, 1)
        iteration = state.iteration + 1
        learning_rate = jnp.maximum(
            iteration.astype(state.delta.dtype) ** spec.amala_adaptation_gamma
            * spec.amala_adaptation_rho,
            spec.amala_adaptation_rho_min,
        )
        change = (
            learning_rate
            * state.delta
            * (rate - spec.amala_target_accept)
            / spec.amala_target_accept
        )
        active = (
            (jnp.abs(rate - spec.amala_target_accept) >= spec.amala_adaptation_tolerance)
            & (iteration > spec.amala_adaptation_window)
            & (count > 0)
        )
        delta = _clip_scale(
            jnp.where(active, state.delta + change, state.delta),
            min_scale=spec.amala_delta_min,
            max_scale=spec.amala_delta_max,
        )
        return DsmcAdaptation(
            delta, accept_window, proposal_window, iteration, accepted_total, count
        )

    adapted: DsmcAdaptation = jax.lax.cond(state.iteration < num_warmup, update, lambda: state)
    return adapted
