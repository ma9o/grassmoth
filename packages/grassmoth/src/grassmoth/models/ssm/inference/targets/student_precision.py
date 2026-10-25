"""Transient StudentT precision augmentation for exact guided path updates.

The parameter ensemble is formed independently of these precisions. After the
conditional path move, discard them before selecting the marginal parameter label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from grassmoth.artifacts.model.likelihood import StudentTLawSpec
from grassmoth.models.ssm.execution.observation_model import grouped_response
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations

if TYPE_CHECKING:
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.observation_groups import LawGroup
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StudentTPrecisions:
    """One log precision per emitted reading; all other storage is inert."""

    log_values: jax.Array
    observed: jax.Array

    def path_log_prob(
        self, target: ParticleTarget, context: ParticleContext, path: jax.Array
    ) -> jax.Array:
        """Complete conditional path density, including exact replay and windows."""
        contexts = jax.tree.map(lambda value: value[None], context)
        return sequential_observations(target, contexts, context, self.log_values).path_log_probs(
            path
        )[0]

    def log_prior(self, target: ParticleTarget, context: ParticleContext) -> jax.Array:
        """Gamma(ν/2, rate ν/2), once per observed factor, including its normalizer."""
        model = cast("HeterogeneousObservation", target.dynamical_model(context).observation_model)
        total = jnp.zeros((), dtype=self.log_values.dtype)
        for group in model.density_groups():
            if isinstance(group.laws[0], StudentTLawSpec):
                indices = jnp.asarray(group.indices)
                law = cast(
                    "StudentTLawSpec[jax.Array]",
                    group.evaluate(
                        jnp.zeros(len(model.laws)),
                        jnp.sqrt(jnp.diag(model.measurement.manifest_cov)),
                    ),
                )
                shape = law.df / 2
                logs = self.log_values[:, indices]
                density = (
                    shape * jnp.log(shape)
                    - gammaln(shape)
                    + (shape - 1) * logs
                    - jnp.exp(jnp.log(shape) + logs)
                )
                total = total + jnp.sum(jnp.where(self.observed[:, indices], density, 0.0))
        return total

    def step_limit(
        self,
        target: ParticleTarget,
        contexts: ParticleContext,
        free: jax.Array,
        delta_max: float,
    ) -> jax.Array:
        """Symmetric ensemble bound for path-independent conditional curvature.

        General window summaries and deterministic replay retain the frozen base
        geometry. Never derive retained-sweep geometry from the reference path.
        """
        if target.intervals is not None or target.state_space.requires_replay:
            return jnp.full((target.times.size, 1), jnp.inf)

        def curvature(context: ParticleContext) -> jax.Array:
            model = cast(
                "HeterogeneousObservation", target.dynamical_model(context).observation_model
            )
            information = jnp.zeros(
                (target.times.size, target.state_space.size, target.state_space.size)
            )
            zero = jnp.zeros(target.state_space.size)
            scales = jnp.sqrt(jnp.diag(model.measurement.manifest_cov))
            for group in model.density_groups():
                if isinstance(group.laws[0], StudentTLawSpec):

                    def operands(
                        state: jax.Array, group: LawGroup = group
                    ) -> StudentTLawSpec[jax.Array]:
                        return cast(
                            "StudentTLawSpec[jax.Array]",
                            group.evaluate(
                                model.linear_predictor(target.project_state(context, state)), scales
                            ),
                        )

                    loading = jax.jacfwd(lambda state: operands(state).loc)(zero)
                    indices = jnp.asarray(group.indices)
                    precision = jnp.where(
                        self.observed[:, indices],
                        jnp.exp(self.log_values[:, indices] - 2 * jnp.log(operands(zero).scale)),
                        0.0,
                    )
                    information = information + jnp.einsum(
                        "tc,ci,cj->tij", precision, loading, loading
                    )
            information = jnp.where(free[:, :, None] & free[:, None, :], information, 0.0)
            return jnp.max(jnp.sum(jnp.abs(information), axis=-1), axis=-1)

        bound = jnp.max(jax.vmap(curvature)(contexts), axis=0)
        return (2 / jnp.maximum(bound, 2 / delta_max))[:, None]


def draw_student_precisions(
    target: ParticleTarget,
    context: ParticleContext,
    path: jax.Array,
    key: jax.Array,
) -> StudentTPrecisions | None:
    """Draw Γ((ν+1)/2, (ν+r²)/2) after exact recurrence and window aggregation."""
    model = cast("HeterogeneousObservation", target.dynamical_model(context).observation_model)
    groups = tuple(
        group for group in model.density_groups() if isinstance(group.laws[0], StudentTLawSpec)
    )
    if not groups:
        return None
    path = target.replay_path(context, path)
    predictors = jax.vmap(
        lambda state: model.linear_predictor(target.project_state(context, state))
    )(path)
    responses = jax.vmap(
        lambda eta: grouped_response(model.density_groups(), eta, model.measurement.manifest_cov)
    )(predictors)
    means, emitted = (
        (responses, jnp.ones_like(responses))
        if target.intervals is None
        else target.intervals.operator.project_response_trajectory(responses)
    )
    observed = jnp.isfinite(target.observations) & (emitted > 0)
    logs = jnp.zeros_like(target.observations)
    selected = jnp.zeros_like(observed)
    keys = jax.random.split(key, len(groups))
    for group, group_key in zip(groups, keys, strict=True):
        indices = jnp.asarray(group.indices)
        law = cast(
            "StudentTLawSpec[jax.Array]",
            group.evaluate(
                jnp.zeros(len(model.laws)), jnp.sqrt(jnp.diag(model.measurement.manifest_cov))
            ),
        )
        mask = observed[:, indices]
        residual = jnp.where(mask, target.observations[:, indices] - means[:, indices], 0.0)
        log_rate = jnp.logaddexp(
            jnp.log(law.df), 2 * (jnp.log(jnp.abs(residual)) - jnp.log(law.scale))
        ) - jnp.log(2.0)
        shapes = jnp.broadcast_to((law.df + 1) / 2, residual.shape)
        reading_keys = jax.random.split(group_key, residual.size)

        def draw(
            draw_key: jax.Array, shape: jax.Array, rate: jax.Array, present: jax.Array
        ) -> jax.Array:
            return jnp.asarray(
                jax.lax.cond(
                    present,
                    lambda: jax.random.loggamma(draw_key, shape, dtype=path.dtype) - rate,
                    lambda: jnp.zeros((), dtype=path.dtype),
                )
            )

        values = jax.vmap(draw)(
            reading_keys, shapes.reshape(-1), log_rate.reshape(-1), mask.reshape(-1)
        )
        logs = logs.at[:, indices].set(values.reshape(mask.shape))
        selected = selected.at[:, indices].set(mask)
    return StudentTPrecisions(logs, selected)
