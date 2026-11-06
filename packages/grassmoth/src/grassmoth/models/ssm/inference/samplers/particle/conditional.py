"""Exact single-parameter path updates and their warmup, independent of the parameter kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    FixedLatentStepSpec,
    GuidedMethodSpec,
    PathUpdateSpec,
)
from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    DELTA_MAX,
    INITIAL_MOVEMENT_MULTIPLIER,
    CurvatureMoments,
    ScaleAdaptationState,
    adapt_curvature_geometry,
    adapt_scales,
    guided_curvature,
    guided_delta_limit,
    initial_delta,
    initialize_adaptation,
    initialize_curvature_moments,
    latent_adaptation,
)
from grassmoth.models.ssm.inference.samplers.particle.augmentation import (
    PrecisionPathUpdate,
    begin_path_update,
)
from grassmoth.models.ssm.inference.samplers.particle.auxiliary_kalman import auxiliary_kalman_step
from grassmoth.models.ssm.inference.samplers.particle.guided import guided_step
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.warmup.parameter_adaptation import covariance_windows

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


class ConditionalPathTuning(NamedTuple):
    """Warmup state frozen in full before retained path updates."""

    adaptation: ScaleAdaptationState
    geometry: jax.Array
    curvature: CurvatureMoments


@dataclass(frozen=True)
class ConditionalPathKernel:
    """The existing exact smoother conditioned on one parameter, with no label proposal.

    Student-t reading precisions are drawn after the marginal parameter update,
    used only by the path kernel, and discarded before the next parameter update.
    Thus the composed transition preserves the marginal joint posterior.
    """

    target: ParticleTarget
    settings: PathUpdateSpec
    num_warmup: int

    @property
    def free_mask(self) -> jax.Array:
        """Coordinates the scientific target admits to a path proposal."""
        target = self.target
        return target.independent_mask & (
            True if target.exact_constraints is None else target.exact_constraints.free_mask
        )

    @property
    def curvature_limited(self) -> bool:
        """Whether adapted guided drift uses the shared frozen curvature metric."""
        settings = self.settings
        return (
            isinstance(settings.method, GuidedMethodSpec)
            and settings.method.kappa == 1
            and not isinstance(settings.latent_step, FixedLatentStepSpec)
        )

    def initialize(self, parameters: jax.Array, path: jax.Array) -> ConditionalPathTuning:
        """Match the shared smoother's existing scale and curvature initialization."""
        geometry = jnp.ones_like(path)
        delta = initial_delta(self.settings)
        if self.curvature_limited:
            context = self.target.context(parameters, self.target.times)
            geometry = guided_delta_limit(self.target, context, path, self.free_mask)
            delta = INITIAL_MOVEMENT_MULTIPLIER
        return ConditionalPathTuning(
            initialize_adaptation(jnp.full((len(path),), delta, dtype=path.dtype)),
            geometry,
            initialize_curvature_moments(geometry),
        )

    def step(
        self,
        parameters: jax.Array,
        reference: jax.Array,
        tuning: ConditionalPathTuning,
        key: jax.Array,
        iteration: jax.Array,
    ) -> tuple[jax.Array, ConditionalPathTuning, dict[str, jax.Array]]:
        """Update the physical path exactly and adapt only during warmup."""
        target = self.target
        context = target.context(parameters, target.times)
        static = build_smoother_setup(
            target, num_parameter_particles=1, free_mask=self.free_mask, diagnostic_metrics=()
        )
        delta = tuning.adaptation.delta
        boundary = begin_path_update(
            target,
            static,
            self.settings,
            context,
            reference,
            delta[:, None] * tuning.geometry if self.curvature_limited else delta,
            parameters[None],
            jnp.zeros((1,), dtype=parameters.dtype),
            key,
        )
        precisions = (
            boundary.precisions.log_values if isinstance(boundary, PrecisionPathUpdate) else None
        )
        if isinstance(self.settings.method, AuxiliaryKalmanMethodSpec):
            result = auxiliary_kalman_step(
                target,
                boundary.context,
                context,
                boundary.key,
                reference,
                block_length=self.settings.update.block_length
                if isinstance(self.settings.update, BlockedPathUpdateSpec)
                else None,
                log_precisions=precisions,
            )
        else:
            result = guided_step(
                target,
                boundary.context,
                context,
                self.settings,
                boundary.key,
                reference,
                log_precisions=precisions,
            )
        path = target.replay_path(context, result.latent_path)
        moved = jnp.any((path != reference) & result.proposal_mask, axis=-1)
        proposed = jnp.any(result.proposal_mask, axis=-1)

        def adapt() -> ConditionalPathTuning:
            geometry, curvature = tuning.geometry, tuning.curvature
            if self.curvature_limited:
                curvature, geometry = adapt_curvature_geometry(
                    curvature,
                    guided_curvature(target, context, path, self.free_mask),
                    geometry,
                    iteration,
                    covariance_windows(self.num_warmup),
                )
            scales = adapt_scales(
                tuning.adaptation,
                latent_adaptation(self.settings),
                moved.astype(delta.dtype),
                proposed,
                self.num_warmup,
                jnp.full_like(delta, 1.0 if self.curvature_limited else DELTA_MAX),
            )
            return ConditionalPathTuning(scales, geometry, curvature)

        updated_tuning: ConditionalPathTuning = jax.lax.cond(
            iteration < self.num_warmup, adapt, lambda: tuning
        )
        return (
            path,
            updated_tuning,
            {
                "latent_accept_prob": jnp.sum(moved) / jnp.maximum(1, jnp.sum(proposed)),
                "latent_delta": updated_tuning.adaptation.delta,
            },
        )
