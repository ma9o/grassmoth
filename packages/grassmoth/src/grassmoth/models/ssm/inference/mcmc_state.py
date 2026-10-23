"""Offline joint-kernel state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.artifacts.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Mapping

    from blackjax.adaptation.step_size import DualAveragingAdaptationState
    from numpy.typing import NDArray

    from grassmoth.models.ssm.inference.targets.particle import ParticleContext


class TrajectoryMCMCState(NamedTuple):
    """Joint position and context carried between offline kernel calls."""

    position: jnp.ndarray
    # Fixed-pilot parameter oracle at this position and this replica's target.
    parameter_gradient: jnp.ndarray
    latent_context: ParticleContext  # noqa: F841, RUF100 - Retained state inspection in notebooks reads the bound context; Vulture checks src separately.
    latent_trajectory: jnp.ndarray
    trajectory_log_prob: jnp.ndarray
    complete_log_posterior: jnp.ndarray
    latent_delta: jnp.ndarray
    param_step_size: jnp.ndarray
    parameter_chol: jax.Array
    # BlackJAX dual-averaging state. Carried but not updated when
    # adaptation_scheme == "simple"; evolves only during warmup otherwise.
    param_da: DualAveragingAdaptationState


def _clip_scale(
    scale: jnp.ndarray,
    *,
    min_scale: float | None,
    max_scale: float | None,
) -> jnp.ndarray:
    clipped = scale
    if min_scale is not None:
        min_value = jnp.nextafter(
            jnp.asarray(min_scale, dtype=clipped.dtype),
            jnp.asarray(jnp.inf, dtype=clipped.dtype),
        )
        clipped = jnp.maximum(clipped, min_value)
    if max_scale is not None:
        clipped = jnp.minimum(clipped, jnp.asarray(max_scale, dtype=clipped.dtype))
    return clipped


@dataclass(frozen=True)
class TrajectoryMCMCResult:
    """MCMC-compatible view of particle posterior samples."""

    chain_samples: Mapping[str, jnp.ndarray]
    chain_extra_fields: Mapping[str, jax.Array | NDArray]
    num_chains: int
    num_samples: int
    backend: str = "marginal_particle_gibbs"

    def get_samples(self, *, group_by_chain: bool = False) -> Mapping[str, jnp.ndarray]:
        """Return parameter draws, preserving chain axes when requested and flattening them otherwise."""
        if group_by_chain:
            return self.chain_samples
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_samples.items()
        }

    def get_extra_fields(
        self, *, group_by_chain: bool = False
    ) -> Mapping[str, jax.Array | NDArray]:
        """Return sampler diagnostics with separate or flattened chain and draw axes."""
        if group_by_chain:
            return self.chain_extra_fields
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_extra_fields.items()
        }

    def __post_init__(self) -> None:
        """Own immutable copies of sample and diagnostic collections after construction."""
        freeze_fields(self)


def _adapt_scale(
    scale: jnp.ndarray,
    *,
    accepted: jnp.ndarray,
    target_accept: float,
    adaptation_rate: float,
    min_scale: float = 1e-6,
    max_scale: float = 1e3,
) -> jnp.ndarray:
    """Simple exponential step-size adaptation on per-step binary accept."""
    dtype = scale.dtype
    accepted_arr = jnp.asarray(accepted, dtype=dtype)
    target_accept_arr = jnp.asarray(target_accept, dtype=dtype)
    adaptation_rate_arr = jnp.asarray(adaptation_rate, dtype=dtype)
    min_scale_arr = jnp.asarray(min_scale, dtype=dtype)
    max_scale_arr = jnp.asarray(max_scale, dtype=dtype)
    factor = jnp.exp(adaptation_rate_arr * (accepted_arr - target_accept_arr))
    return jnp.clip(scale * factor, min_scale_arr, max_scale_arr)


def _latent_summary_from_chain_moments(
    chain_means: jnp.ndarray,
    chain_stds: jnp.ndarray,
) -> Mapping[str, jnp.ndarray]:
    pooled_mean = jnp.mean(chain_means, axis=0)
    pooled_second_moment = jnp.mean(chain_stds * chain_stds + chain_means * chain_means, axis=0)
    pooled_var = jnp.maximum(pooled_second_moment - pooled_mean * pooled_mean, 0.0)
    return {
        "chain_mean": chain_means,
        "chain_std": chain_stds,
        "mean": pooled_mean,
        "std": jnp.sqrt(pooled_var),
    }


def _clip_dual_averaging_state(
    da_state: DualAveragingAdaptationState,
    *,
    min_scale: float | None,
    max_scale: float | None,
) -> DualAveragingAdaptationState:
    if min_scale is None and max_scale is None:
        return da_state

    log_min = (
        None
        if min_scale is None
        else jnp.log(jnp.asarray(min_scale, dtype=jnp.asarray(da_state.mu).dtype))
    )
    log_max = (
        None
        if max_scale is None
        else jnp.log(jnp.asarray(max_scale, dtype=jnp.asarray(da_state.mu).dtype))
    )

    def _clip_log_value(value: float | jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.asarray(value)
        if log_min is not None:
            clipped = jnp.maximum(clipped, log_min)
        if log_max is not None:
            clipped = jnp.minimum(clipped, log_max)
        return clipped

    return da_state._replace(
        log_step_size=_clip_log_value(da_state.log_step_size),
        log_step_size_avg=_clip_log_value(da_state.log_step_size_avg),
        mu=_clip_log_value(da_state.mu),
    )
