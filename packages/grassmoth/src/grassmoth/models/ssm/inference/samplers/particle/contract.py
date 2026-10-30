"""Shared dSMC contract types and per-step conditional or marginalized targets."""

# References: see dsmc.py and ../mpgibbs/kernel.py for algorithm provenance.

from collections.abc import Callable, Mapping
from typing import NamedTuple, Protocol

import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.artifacts.sampling import AuxiliaryDrift, LatentBlockSize
from grassmoth.models.ssm.inference.targets.coordinates import PathChart, ScaleChart
from grassmoth.models.ssm.inference.targets.intervals import ParticleIntervalObservations
from grassmoth.models.ssm.inference.targets.particle import ParticleContext
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int

type ParticleContextRuntimeFn = Callable[[Float[Array, " U"], Float[Array, " T"]], ParticleContext]
type ObservationIncrementLogProbRuntimeFn = Callable[
    [ParticleContext, Float[Array, " D"], Int[Array, ""], Float[Array, "T M"]], Float[Array, ""]
]
type TrajectoryLogProbRuntimeFn = Callable[
    [ParticleContext, Float[Array, "T D"], Float[Array, "T M"]], Float[Array, ""]
]
type TransitionLogProbFn = Callable[
    [ParticleContext, Float[Array, "P D"], Float[Array, "Q D"], Int[Array, ""]], Float[Array, "P Q"]
]


class TransitionValueGradFn(Protocol):
    """Exact transition density and a score with an explicit differentiation direction."""

    def __call__(
        self,
        context: ParticleContext,
        previous: Float[Array, " D"],
        current: Float[Array, " D"],
        index: Int[Array, ""] | int,
        *,
        current_gradient: bool,
    ) -> tuple[Float[Array, ""], Float[Array, " D"]]:
        """Evaluate the declared transition and its requested state gradient."""
        ...


class PaidMixture(NamedTuple):
    """Fixed initialization moments and the owned mixture specification."""

    mean: jnp.ndarray
    variance: jnp.ndarray
    enabled: jnp.ndarray

    def in_coordinates(self, chart: PathChart, position: jnp.ndarray) -> "PaidMixture":
        """Express the physical pilot at one symmetric ensemble guide position."""
        return PaidMixture(
            chart.encode(position, self.mean),
            self.variance * jnp.exp(-2 * chart.log_scale(position))
            if isinstance(chart, ScaleChart)
            else self.variance,
            self.enabled,
        )


class TrajectoryLabelScores(NamedTuple):
    """Exact per-label densities and the corrected conditional label law."""

    log_probs: Float[Array, " K"]
    trajectory_log_probs: Float[Array, " K"]
    log_posteriors: Float[Array, " K"]


class ParticleSmootherResult(NamedTuple):
    """A trajectory draw and parameter-label weights from conditional dSMC."""

    latent_path: Float[Array, "T D"]
    label_scores: TrajectoryLabelScores
    origin_path: Int[Array, " T"]
    proposal_mask: Bool[Array, "T D"]
    diagnostics: Mapping[str, jnp.ndarray]


class SmootherSetup(NamedTuple):
    """Build-time configuration and target callables for the smoother context."""

    latent_context_runtime_fn: ParticleContextRuntimeFn
    log_prior_unc_fn: Callable[[jnp.ndarray], jnp.ndarray]
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    trajectory_log_prob_fn: TrajectoryLogProbRuntimeFn
    runtime_observations: Float[Array, "T M"]
    runtime_times: Float[Array, " T"]
    latent_free_mask: Bool[Array, "T D"]
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
    guide_position: Float[Array, " U"]
    initial_label_log_probs: Float[Array, " K"]
    num_steps: int
    num_free_particles: int
    num_parameter_particles: int
    latent_dtype: DTypeLike
    traj_dtype: DTypeLike
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    runtime_observations: Float[Array, "T M"]
    latent_free_mask: Bool[Array, "T D"]
    amala_delta: Float[Array, " T"] | Float[Array, "T D"]
    drift: AuxiliaryDrift
    latent_block_coords: LatentBlockSize
    paid_mix: PaidMixture | None
    initial_value_grad_by_param: Callable[
        [Float[Array, " D"]], tuple[Float[Array, " K"], Float[Array, "K D"]]
    ]
    transition_current_value_grad_by_param: Callable[
        [Float[Array, " D"], Float[Array, " D"], Int[Array, ""]],
        tuple[Float[Array, " K"], Float[Array, "K D"]],
    ]
    transition_next_value_grad_by_param: Callable[
        [Float[Array, " D"], Float[Array, " D"], Int[Array, ""]],
        tuple[Float[Array, " K"], Float[Array, "K D"]],
    ]
    pairwise_transition_log_probs: Callable[
        [Float[Array, "P D"], Float[Array, "Q D"], Int[Array, ""]], Float[Array, "P Q K"]
    ]
    score_labels: Callable[[Float[Array, "T D"]], TrajectoryLabelScores]
    label_scores_from_densities: Callable[[Float[Array, " K"]], TrajectoryLabelScores]
    intervals: ParticleIntervalObservations | None
    diagnostic_weights: jnp.ndarray | bool
