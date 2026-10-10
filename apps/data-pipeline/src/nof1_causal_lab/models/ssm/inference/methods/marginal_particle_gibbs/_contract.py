"""Latent-smoother contract types and per-step context for MPGibbs."""

# References: see kernel.py and smoothers/dsmc.py for algorithm provenance.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleContext

if TYPE_CHECKING:
    from jax.typing import DTypeLike
    from jaxtyping import Array, Float

    from nof1_causal_lab.models.ssm.inference.targets.intervals import ParticleIntervalObservations


type ParticleContextRuntimeFn = Callable[[jnp.ndarray, jnp.ndarray], ParticleContext]
type ObservationIncrementLogProbRuntimeFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]
type TrajectoryLogProbRuntimeFn = Callable[[ParticleContext, jnp.ndarray, jnp.ndarray], jnp.ndarray]
type TransitionLogProbFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]

# Paid mixture leaf: the amala_exact z-anchored component plus a FIXED pilot
# component (IEKS warmup moments) and a wide tail, all inside one paid proposal
# density. Any component that is useless on a given fit costs only its share of
# proposal mass — never correctness — so the mixture strictly generalizes
# amala_exact (its z-component alone).
_DSMC_LEAF_PROPOSAL_PAID_MIX = "paid_mix"


class MPGibbsLatentSmootherResult(NamedTuple):
    """Production result contract for MPGibbs latent smoothers."""

    latent_path: Float[Array, "T D"]
    final_label_log_probs: Float[Array, " K"]
    origin_path: jnp.ndarray
    proposal_mask: jnp.ndarray
    diagnostics: Mapping[str, jnp.ndarray]


class MPGibbsStatic(NamedTuple):
    """Build-time configuration and target callables for the smoother context."""

    latent_context_runtime_fn: ParticleContextRuntimeFn
    log_prior_unc_fn: Callable[[jnp.ndarray], jnp.ndarray]
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    trajectory_log_prob_fn: TrajectoryLogProbRuntimeFn
    runtime_observations: jnp.ndarray
    runtime_times: jnp.ndarray
    latent_free_mask: jnp.ndarray
    num_particles: int
    num_parameter_particles: int
    latent_delta: float
    amala_kappa: float
    amala_grad_clip: float
    # Number of latent coordinates proposed per sweep (None = all). Restricting the
    # per-sweep update to a random coordinate block sidesteps the joint-coherence
    # weight degeneracy of full-state proposals at higher latent dimension; the tree
    # seams still pay the full transition density, so this is exactly the
    # coordinate-conditional cSMC on the same target.
    latent_block_coords: int | None
    # paid_mix leaf: mixture weights (wide weight = 1 - z - pilot) and the FIXED
    # per-time pilot moments derived from the IEKS warmup paths.
    paid_mix_z_weight: float
    paid_mix_pilot_weight: float
    # Present only for paid_mix: means, pilot variances, wide-tail variances.
    pilot_moments: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None
    transition_initial_log_prob_fn: Callable[[ParticleContext, jnp.ndarray], jnp.ndarray]
    transition_log_prob_fn: TransitionLogProbFn
    transition_log_probs_for_pairs_fn: TransitionLogProbFn
    transition_pairwise_log_probs_fn: TransitionLogProbFn
    diagnostic_metrics: frozenset[str]
    intervals: ParticleIntervalObservations | None


class SmootherContext(NamedTuple):
    """Per-step latent-smoother inputs.

    Explicit interface replacing the implicit closure capture the smoothers
    previously relied on. Built once per joint step by
    :func:`build_smoother_context`; consumed by the smoother modules.
    """

    contexts: ParticleContext
    initial_label_log_probs: Float[Array, " K"]
    num_steps: int
    num_free_particles: int
    num_parameter_particles: int
    latent_dtype: DTypeLike
    traj_dtype: DTypeLike
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    runtime_observations: jnp.ndarray
    latent_free_mask: jnp.ndarray
    amala_delta: Float[Array, " D"]
    amala_kappa: float
    amala_grad_clip: float
    latent_block_coords: int | None
    paid_mix_z_weight: float
    paid_mix_pilot_weight: float
    # Present only for paid_mix: means, pilot variances, wide-tail variances.
    pilot_moments: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None
    initial_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_current_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_next_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    selected_transition_log_probs: Callable[..., jnp.ndarray]
    pairwise_transition_log_probs: Callable[..., jnp.ndarray]
    trajectory_label_log_probs: Callable[..., jnp.ndarray]
    trajectory_label_log_weights: Callable[[jnp.ndarray], jnp.ndarray]
    intervals: ParticleIntervalObservations | None
