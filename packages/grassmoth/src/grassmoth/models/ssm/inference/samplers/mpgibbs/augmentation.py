"""Discard conditional reading precisions before MPGibbs selects its parameter label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax

from grassmoth.models.ssm.inference.samplers.particle.augmentation import MarginalPathUpdate
from grassmoth.models.ssm.inference.samplers.particle.math import _normalize_log_probs

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import ParameterMoveInputs
    from grassmoth.models.ssm.inference.samplers.particle.contract import TrajectoryLabelScores
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True)
class ConditionalPathScores:
    """Scores paid by the path update, possibly conditional on precisions."""

    values: TrajectoryLabelScores


class MarginalLabelScores(NamedTuple):
    """Marginal target scores admitted to the final label draw and state caches."""

    log_probs: jax.Array
    trajectory_log_probs: jax.Array
    log_posteriors: jax.Array


def finish_path_update(
    prepared: ParameterMoveInputs,
    conditional: ConditionalPathScores,
    target: ParticleTarget,
    latent_path: jax.Array,
) -> MarginalLabelScores:
    """Discard lambda before selecting a label; only marginal scores may be cached."""
    boundary = prepared.boundary
    if isinstance(boundary, MarginalPathUpdate):
        return MarginalLabelScores(*conditional.values)
    trajectory = jax.vmap(
        lambda context: target.path_log_prob(context, latent_path, target.observations)
    )(boundary.context.contexts)
    posterior = jax.vmap(target.log_prior)(prepared.particles) + trajectory
    return MarginalLabelScores(
        _normalize_log_probs(posterior + prepared.correction), trajectory, posterior
    )
