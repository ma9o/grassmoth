"""Shared smoother contract types and per-step conditional or marginalized targets."""

# References: see ../mpgibbs/kernel.py for algorithm provenance.

from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.models.ssm.inference.targets.particle import ParticleContext
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int

type ParticleContextRuntimeFn = Callable[[Float[Array, " U"], Float[Array, " T"]], ParticleContext]
type TrajectoryLogProbRuntimeFn = Callable[
    [ParticleContext, Float[Array, "T D"], Float[Array, "T M"]], Float[Array, ""]
]


class TrajectoryLabelScores(NamedTuple):
    """Exact per-label densities and the corrected conditional label law."""

    log_probs: Float[Array, " K"]
    trajectory_log_probs: Float[Array, " K"]
    log_posteriors: Float[Array, " K"]


class ParticleSmootherResult(NamedTuple):
    """A trajectory draw and parameter-label weights from a conditional path update."""

    latent_path: Float[Array, "T D"]
    label_scores: TrajectoryLabelScores
    origin_path: Int[Array, " T"]
    proposal_mask: Bool[Array, "T D"]
    diagnostics: Mapping[str, jnp.ndarray]


class SmootherSetup(NamedTuple):
    """Build-time configuration and target callables for the smoother context."""

    latent_context_runtime_fn: ParticleContextRuntimeFn
    log_prior_unc_fn: Callable[[jnp.ndarray], jnp.ndarray]
    trajectory_log_prob_fn: TrajectoryLogProbRuntimeFn
    runtime_observations: Float[Array, "T M"]
    runtime_times: Float[Array, " T"]
    latent_free_mask: Bool[Array, "T D"]
    num_parameter_particles: int
    diagnostic_weights: jnp.ndarray


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
    num_parameter_particles: int
    traj_dtype: DTypeLike
    runtime_observations: Float[Array, "T M"]
    latent_free_mask: Bool[Array, "T D"]
    amala_delta: Float[Array, " T"] | Float[Array, "T D"]
    score_labels: Callable[[Float[Array, "T D"]], TrajectoryLabelScores]
    label_scores_from_densities: Callable[[Float[Array, " K"]], TrajectoryLabelScores]
    diagnostic_weights: jnp.ndarray | bool
