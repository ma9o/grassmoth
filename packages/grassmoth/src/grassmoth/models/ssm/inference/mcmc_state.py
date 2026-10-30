"""Offline joint-kernel state."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import DualAveragingAdaptationState, dual_averaging_adaptation

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.models.ssm.inference.targets.coordinates import PathChart
from grassmoth.models.ssm.inference.targets.particle import ParticleContext
from grassmoth.models.ssm.shapes import Array, Float


class TrajectoryMCMCState(NamedTuple):
    """Joint position and context carried between offline kernel calls."""

    position: Float[Array, "*batch U"]
    # Fixed-pilot parameter oracle at this position and this replica's target.
    parameter_gradient: Float[Array, "*batch U"]
    latent_context: ParticleContext  # noqa: F841, RUF100 - Retained state inspection in notebooks reads the bound context; Vulture checks src separately.
    latent_trajectory: Float[Array, "*batch T D"]
    trajectory_log_prob: Float[Array, "*batch"]
    complete_log_posterior: Float[Array, "*batch"]
    latent_delta: Float[Array, "*batch T"] | Float[Array, "*batch T D"]
    param_step_size: Float[Array, "*batch"]
    parameter_chol: Float[Array, "*batch U U"]
    # BlackJAX dual-averaging state. Carried but not updated when
    # adaptation_scheme == "simple"; evolves only during warmup otherwise.
    param_da: DualAveragingAdaptationState
    # The physical path remains the durable state; the chart is fixed within a sweep.
    chart: PathChart | None = None


def trajectory_state(
    *,
    position: Float[Array, " U"],
    parameter_gradient: Float[Array, " U"],
    latent_context: ParticleContext,
    latent_trajectory: Float[Array, "T D"],
    trajectory_log_prob: Float[Array, ""],
    complete_log_posterior: Float[Array, ""],
    latent_delta: Float[Array, " T"] | Float[Array, "T D"],
    param_step_size: Float[Array, ""],
    parameter_chol: Float[Array, "U U"],
    param_da: DualAveragingAdaptationState,
    chart: PathChart | None = None,
) -> TrajectoryMCMCState:
    """Construct one chain, binding parameter, path and tuning axes together.

    The test import hook checks these arguments in one shape context. NamedTuple
    field annotations alone do not check their relationships. Under vmap this
    constructor sees one chain; JAX batches the completed record afterwards.
    """
    return TrajectoryMCMCState(
        position,
        parameter_gradient,
        latent_context,
        latent_trajectory,
        trajectory_log_prob,
        complete_log_posterior,
        latent_delta,
        param_step_size,
        parameter_chol,
        param_da,
        chart,
    )


def _clip_scale(
    scale: Float[Array, " *shape"],
    *,
    min_scale: float | Float[Array, " *#shape"] | None,
    max_scale: float | Float[Array, " *#shape"] | None,
) -> Float[Array, " *shape"]:
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

    chain_samples: Mapping[str, jnp.ndarray | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]
    chain_extra_fields: Mapping[str, jax.Array | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]
    num_chains: int
    num_samples: int
    backend: str = "marginal_particle_gibbs"

    def get_samples(
        self, *, group_by_chain: bool = False
    ) -> Mapping[str, jnp.ndarray | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Return parameter draws, preserving chain axes when requested and flattening them otherwise."""
        if group_by_chain:
            return self.chain_samples
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_samples.items()
        }

    def get_extra_fields(
        self, *, group_by_chain: bool = False
    ) -> Mapping[str, jax.Array | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
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
    scale: Float[Array, " *shape"],
    *,
    accepted: Float[Array, " *shape"],
    target_accept: float | Float[Array, ""],
    adaptation_rate: float | Float[Array, ""],
    min_scale: float | Float[Array, ""] = 1e-6,
    max_scale: float | Float[Array, ""] = 1e3,
) -> Float[Array, " *shape"]:
    """Simple exponential step-size adaptation on per-step binary accept."""
    dtype = scale.dtype
    accepted_arr = jnp.asarray(accepted, dtype=dtype)
    target_accept_arr = jnp.asarray(target_accept, dtype=dtype)
    adaptation_rate_arr = jnp.asarray(adaptation_rate, dtype=dtype)
    min_scale_arr = jnp.asarray(min_scale, dtype=dtype)
    max_scale_arr = jnp.asarray(max_scale, dtype=dtype)
    factor = jnp.exp(adaptation_rate_arr * (accepted_arr - target_accept_arr))
    return jnp.clip(scale * factor, min_scale_arr, max_scale_arr)


def restarted_dual_averaging(step_size: Float[Array, " *shape"]) -> DualAveragingAdaptationState:
    """Start dual averaging at each step scalar, whatever the batch shape."""
    initialize = dual_averaging_adaptation(target=0.0)[0]
    flat = jax.vmap(initialize)(step_size.reshape(-1))
    return jax.tree.map(lambda value: value.reshape(step_size.shape), flat)


def _clip_dual_averaging_state(
    da_state: DualAveragingAdaptationState,
    *,
    min_scale: float | Float[Array, " *shape"] | None,
    max_scale: float | Float[Array, " *shape"] | None,
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
