"""Native observation-only models for distribution and predictive-diagnostic tests."""

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from grassmoth.artifacts.model.likelihood import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    LinkFunction,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.predictive.observations import sample_model_observations
from tests.support.observation import observation_laws


def _zero_drift(x, u, t):
    return jnp.zeros_like(x)


def sample_observation_fixture(
    linear_predictors,
    samples,
    times,
    *,
    rng_key=None,
    manifest_dists=None,
    manifest_links=None,
    manifest_level_counts=None,
    observation_support=None,
    observation_mask=None,
    n_subsample=50,
    manifest_names=None,
):
    count, _, channels = linear_predictors.shape
    n_use = min(n_subsample, count)
    indices = jnp.linspace(0, count - 1, n_use).astype(int)

    def broadcast(value):
        value = jnp.asarray(value)
        if value.ndim == 0:
            return jnp.broadcast_to(value, (n_use,))
        if value.shape[0] == n_use:
            return value
        if value.shape[0] >= count:
            return value[indices]
        return jnp.broadcast_to(value, (n_use, *value.shape))

    families = (
        [DistributionFamily(value) for value in manifest_dists]
        if manifest_dists is not None
        else [DistributionFamily.GAUSSIAN] * channels
    )
    links = (
        [LinkFunction(OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[family][0]) for family in families]
        if manifest_links is None
        else [LinkFunction(value) for value in manifest_links]
    )
    if manifest_level_counts is None and any(
        family in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        for family in families
    ):
        raise ValueError("manifest_level_counts is required for discrete observation fixtures")
    extras = {name: broadcast(value) for name, value in samples.items() if name.startswith("obs_")}

    def build(covariance, parameters):
        if manifest_level_counts is not None:
            parameters = {**parameters, "obs_level_counts": jnp.asarray(manifest_level_counts)}
        return dsx.DynamicalModel(
            initial_condition=dist.Delta(jnp.zeros(channels), event_dim=1),
            state_evolution=dsx.DeterministicContinuousTimeStateEvolution(drift=_zero_drift),
            observation_model=HeterogeneousObservation(
                MeasurementParams(jnp.eye(channels), jnp.zeros(channels), covariance),
                observation_laws(families, links, parameters, level_counts=manifest_level_counts),
            ),
            control_dim=0,
            t0=times[0],
        )

    models = eqx.filter_vmap(build)(broadcast(samples["manifest_cov"]), extras)
    return sample_model_observations(
        models,
        linear_predictors[indices],
        times,
        rng_key=jax.random.PRNGKey(42) if rng_key is None else rng_key,
        observation_support=observation_support,
        observation_mask=observation_mask,
        manifest_names=manifest_names or [f"var_{index}" for index in range(channels)],
    )
