"""Exact predictive moments from the same bound native law used by execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.model.likelihood import (
    CategoricalLawSpec,
    GammaLawSpec,
    NormalLawSpec,
    OrderedLogisticLawSpec,
)
from grassmoth.models.ssm.compile.observations import materialize_observation_laws
from grassmoth.models.ssm.execution.observation_distributions import (
    category_probabilities,
    evaluate_law,
    feasible_law,
    gaussian_distribution,
    law_moments,
    point_observation_scales,
    with_response,
)
from grassmoth.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.predictive.types import PredictiveDraws
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime


def observation_signal_and_variance(
    compiled_dynamical_model: CompiledDynamicalModel,
    prediction: PredictiveDraws,
    manifest_index: int,
    time_indices: np.ndarray,
    *,
    observation_support: ObservationSupportRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Scalar moments or true-width categorical vectors; undefined moments stay NaN/inf."""
    paths = prediction.trajectory
    operator = compile_observation_operator(observation_support)
    interval = operator is not None and manifest_index in operator.interval_summary_indices
    all_gaussian = all(
        isinstance(observation.law, NormalLawSpec)
        for observation in compiled_dynamical_model.observations
    )

    def per_draw(means, predictors, parameters):
        bound = materialize_observation_laws(compiled_dynamical_model, parameters)[manifest_index]
        covariance = parameters["manifest_cov"]
        scale = (
            jnp.sqrt(covariance[manifest_index, manifest_index])
            if interval
            else point_observation_scales(covariance)[manifest_index]
        )

        def at_time(predictor, response):
            categorical = isinstance(bound, (OrderedLogisticLawSpec, CategoricalLawSpec))
            baseline = (
                predictor
                if categorical
                else jnp.ones_like(predictor)
                if isinstance(bound, GammaLawSpec)
                else jnp.zeros_like(predictor)
            )
            law = evaluate_law(bound, baseline, scale)
            if not categorical:
                law = with_response(law, response)
            feasible, valid = feasible_law(law)
            if isinstance(feasible, (OrderedLogisticLawSpec, CategoricalLawSpec)):
                return category_probabilities(feasible), None
            if isinstance(law, NormalLawSpec) and (all_gaussian or interval):
                marginal = gaussian_distribution(
                    law.loc[None],
                    covariance[
                        manifest_index : manifest_index + 1, manifest_index : manifest_index + 1
                    ],
                )
                return marginal.mean[0], marginal.variance[0]
            mean, variance = law_moments(feasible)
            return jnp.where(valid, mean, jnp.nan), jnp.where(valid, variance, jnp.nan)

        return jax.vmap(at_time)(predictors, means)

    signal, variance = jax.vmap(per_draw)(
        paths.expected_observations[:, time_indices, manifest_index],
        paths.linear_predictors[:, time_indices, manifest_index],
        dict(prediction.parameters),
    )
    return np.asarray(signal), None if variance is None else np.asarray(variance)
