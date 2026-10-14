"""Constructor/layout sampling with the original row and channel random streams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.model.likelihood import (
    BernoulliLogitsLawSpec,
    CategoricalLawSpec,
    Law,
    NormalLawSpec,
    OrderedLogisticLawSpec,
)
from nof1_causal_lab.models.ssm.covariance import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    category_probabilities,
    feasible_law,
    gaussian_distribution,
    point_observation_scales,
    safe_native,
    to_native,
)
from nof1_causal_lab.models.ssm.execution.observation_groups import compile_law_groups

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.execution.observation_groups import LawGroup

type ObservationSampleFn = Callable[[jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class PointObservationSampler:
    """Point-observation sampling callable compiled for a particular ordered set of laws."""

    sample_point: ObservationSampleFn


@dataclass(frozen=True)
class MeanObservationSampler:
    """Sampling callable operating on response-mean trajectories over measurement support."""

    sample_mean_trajectory: ObservationSampleFn


def _trajectory_sampler(sample_vector: ObservationSampleFn) -> ObservationSampleFn:
    def sample(key: jax.Array, trajectory: jax.Array) -> jax.Array:
        return jax.vmap(sample_vector)(jax.random.split(key, trajectory.shape[0]), trajectory)

    return sample


def _padded_categorical(masses: jax.Array) -> dist.CategoricalLogits:
    """Draw from padded category masses with the original normalization arithmetic."""
    return dist.CategoricalLogits(logits=jnp.log(masses / jnp.sum(masses)))


def _sample_native(key: jax.Array, law: Law[jax.Array], sampling_size: int) -> jax.Array:
    feasible, valid = feasible_law(law)
    native = to_native(feasible)
    if isinstance(law, BernoulliLogitsLawSpec):
        native = dist.CategoricalLogits(
            logits=jnp.stack(
                (jax.nn.log_sigmoid(-law.logits), jax.nn.log_sigmoid(law.logits)), axis=-1
            )
        )
    elif isinstance(law, CategoricalLawSpec):
        # Preserve the padded Gumbel shape and the old probability arithmetic.
        logits = jnp.pad(
            law.logits, ((0, sampling_size - law.logits.shape[-1]),), constant_values=-1e30
        )
        native = _padded_categorical(
            jnp.where(jnp.arange(sampling_size) < law.logits.shape[-1], jax.nn.softmax(logits), 0.0)
        )
    elif isinstance(feasible, OrderedLogisticLawSpec):
        masses = category_probabilities(feasible)
        native = _padded_categorical(
            jnp.pad(jnp.maximum(masses, 0.0), ((0, sampling_size - masses.shape[-1]),))
        )
    draw = native.sample(key)
    return jnp.where(valid, draw, jnp.nan)


def build_point_observation_sampler(
    laws: ObservationLaws, manifest_cov: jax.Array, *, groups: tuple[LawGroup, ...] | None = None
) -> PointObservationSampler:
    """Compile point sampling with joint Gaussian covariance or grouped heterogeneous laws."""
    n_channels = len(laws)
    if all(isinstance(law, NormalLawSpec) for law in laws):
        factor = jnp.linalg.cholesky(symmetrize_with_jitter(manifest_cov))

        def sample_gaussian(key: jax.Array, predictors: jax.Array) -> jax.Array:
            return predictors + jnp.matmul(factor, jax.random.normal(key, predictors.shape))

        return PointObservationSampler(sample_gaussian)
    compiled_groups = compile_law_groups(laws) if groups is None else groups
    scales = point_observation_scales(manifest_cov)

    def sample_vector(key: jax.Array, predictors: jax.Array) -> jax.Array:
        channel_keys = jax.random.split(key, n_channels)
        sampled = jnp.zeros_like(predictors)
        for group in compiled_groups:
            indices = jnp.asarray(group.indices)
            law = group.evaluate(predictors, scales)
            sampling_size = 0
            if isinstance(group.laws[0], OrderedLogisticLawSpec):
                sampling_size = group.laws[0].cutpoints.sampling_size
            elif isinstance(group.laws[0], CategoricalLawSpec):
                sampling_size = group.laws[0].logits.sampling_size
            draws = jax.vmap(partial(_sample_native, sampling_size=sampling_size))(
                channel_keys[indices], law
            )
            sampled = sampled.at[indices].set(draws.astype(sampled.dtype))
        return sampled

    return PointObservationSampler(sample_vector)


def build_interval_summary_sampler(
    laws: ObservationLaws, manifest_cov: jax.Array, interval_summary_indices: Sequence[int]
) -> MeanObservationSampler:
    """Project responses first; native arguments are reconstructed per original group."""
    groups = compile_law_groups(laws, interval_summary_indices, interval=True)
    scales = jnp.sqrt(jnp.diag(manifest_cov))
    positions = jnp.asarray(interval_summary_indices)

    def sample_vector(key: jax.Array, responses: jax.Array) -> jax.Array:
        means = jnp.zeros((len(laws),), dtype=responses.dtype).at[positions].set(responses)
        sampled = jnp.zeros_like(means)
        keys = (key,) if len(groups) == 1 else jax.random.split(key, len(groups))
        for subkey, group in zip(keys, groups, strict=True):
            indices = jnp.asarray(group.indices)
            law = group.evaluate(jnp.zeros_like(means), scales, means)
            if isinstance(law, NormalLawSpec):
                # The jittered covariance block, not the scale operand, owns Gaussian noise.
                native = gaussian_distribution(law.loc, manifest_cov[jnp.ix_(indices, indices)])
                valid = jnp.isfinite(law.loc)
            else:
                native, valid = safe_native(law)
            sampled = sampled.at[indices].set(jnp.where(valid, native.sample(subkey), jnp.nan))
        return sampled[positions]

    return MeanObservationSampler(_trajectory_sampler(sample_vector))
