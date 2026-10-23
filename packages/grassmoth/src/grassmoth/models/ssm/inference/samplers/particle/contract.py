"""Shared dSMC contract types and per-step conditional or marginalized targets."""

# References: see dsmc.py and ../mpgibbs/kernel.py for algorithm provenance.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax.numpy as jnp

from grassmoth.models.ssm.inference.targets.particle import ParticleContext

if TYPE_CHECKING:
    from jax.typing import DTypeLike
    from jaxtyping import Array, Float

    from grassmoth.artifacts.sampling import AuxiliaryDrift, LatentBlockSize
    from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations


type ParticleContextRuntimeFn = Callable[[jnp.ndarray, jnp.ndarray], ParticleContext]
type ObservationIncrementLogProbRuntimeFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]
type TrajectoryLogProbRuntimeFn = Callable[[ParticleContext, jnp.ndarray, jnp.ndarray], jnp.ndarray]
type TransitionLogProbFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]


class TransitionValueGradFn(Protocol):
    """Exact transition density and a score with an explicit differentiation direction."""

    def __call__(
        self,
        context: ParticleContext,
        previous: jnp.ndarray,
        current: jnp.ndarray,
        index: jnp.ndarray | int,
        *,
        current_gradient: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate the declared transition and its requested state gradient."""
        ...


class PaidMixture(NamedTuple):
    """Fixed initialization moments and the owned mixture specification."""

    mean: jnp.ndarray
    variance: jnp.ndarray
    enabled: jnp.ndarray


class TrajectoryLabelScores(NamedTuple):
    """Exact per-label densities and the corrected conditional label law."""

    log_probs: Float[Array, " K"]
    trajectory_log_probs: Float[Array, " K"]
    log_posteriors: Float[Array, " K"]


class ParticleSmootherResult(NamedTuple):
    """A trajectory draw and parameter-label weights from conditional dSMC."""

    latent_path: Float[Array, "T D"]
    label_scores: TrajectoryLabelScores
    origin_path: jnp.ndarray
    proposal_mask: jnp.ndarray
    diagnostics: Mapping[str, jnp.ndarray]


class SmootherSetup(NamedTuple):
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
    drift: AuxiliaryDrift
    # Number of latent coordinates proposed per sweep ("all" = all). Restricting the
    # per-sweep update to a random coordinate block sidesteps the joint-coherence
    # weight degeneracy of full-state proposals at higher latent dimension; the tree
    # seams still pay the full transition density, so this is exactly the
    # coordinate-conditional cSMC on the same target.
    latent_block_coords: LatentBlockSize
    paid_mix: PaidMixture | None
    transition_initial_log_prob_fn: Callable[[ParticleContext, jnp.ndarray], jnp.ndarray]
    transition_value_grad_fn: TransitionValueGradFn
    transition_pairwise_log_probs_fn: TransitionLogProbFn
    diagnostic_weights: jnp.ndarray
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
    amala_delta: Float[Array, " T"]
    drift: AuxiliaryDrift
    latent_block_coords: LatentBlockSize
    paid_mix: PaidMixture | None
    initial_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_current_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_next_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    pairwise_transition_log_probs: Callable[..., jnp.ndarray]
    score_labels: Callable[[jnp.ndarray], TrajectoryLabelScores]
    intervals: ParticleIntervalObservations | None
    diagnostic_weights: jnp.ndarray | bool
