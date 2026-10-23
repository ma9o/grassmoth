"""Prepared exact noise agrees with native distributions under masks and nonlinear means."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from dynestyx.observation_missingness import masked_observation_log_prob

from grassmoth.artifacts.model.likelihood import NormalLawSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.execution.observation_distributions import (
    gaussian_distribution,
    masked_law_log_prob,
)
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("with_interval", [False, True])
def test_prepared_noise_matches_native_values_and_gradients(with_interval):
    times = np.array([0.0, 0.7, 1.8, 2.4])
    readings = np.array(
        [
            [0.2, -0.3, 2.0, 0.4],
            [np.nan, 0.7, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [-0.4, np.nan, 1.0, 0.8],
        ]
    )
    covariance = np.array([[0.5, 0.12, 0, 0], [0.12, 0.7, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.4]])
    summaries = ("last", "last", "last", "mean" if with_interval else "last")
    starts = np.broadcast_to(times[:, None], readings.shape).copy()
    if with_interval:
        starts[:, 3] = np.maximum(times - 0.5, 0)
        readings[0, 3] = np.nan
        starts[0, 3] = np.nan
    base = interval_target(
        times,
        readings,
        starts,
        summaries,
        state_dim=2,
        covariance=covariance,
        families=(
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
            DistributionFamily.POISSON,
            DistributionFamily.STUDENT_T,
        ),
    )

    def declared(bound):
        model = base.dynamical_model(bound)
        old = model.observation_model
        assert isinstance(old, HeterogeneousObservation)
        scale = jnp.exp(0.2 * old.measurement.manifest_means[0])
        observation = HeterogeneousObservation(
            MeasurementParams(
                old.measurement.lambda_mat,
                old.measurement.manifest_means,
                old.measurement.manifest_cov * scale,
            ),
            old.laws,
        )
        evolution = dsx.GaussianStateEvolution(
            lambda state, _u, t0, t1: state + (t1 - t0) * (0.2 * jnp.sin(state) - 0.4 * state**3),
            lambda _state, _u, t0, t1: jnp.array([[0.3, 0.06], [0.06, 0.4]]) * scale * (t1 - t0),
        )
        return eqx.tree_at(
            lambda value: (value.observation_model, value.state_evolution),
            model,
            (observation, evolution),
        )

    target = replace(
        base, dynamical_model=declared, state_projection=lambda _bound, state: jnp.tanh(state)
    )
    states = jnp.array([[0.2, -0.4], [-0.3, 0.7], [0.6, 0.1], [-0.2, -0.5]])

    def compare(theta):
        bound = target.context(theta, target.times)
        factors = target.prepare_noise_factors(bound, target.observations)
        observation = target.dynamical_model(bound).observation_model
        assert isinstance(observation, HeterogeneousObservation)
        assert len(observation.group_indices) == 3

        def native_transition(previous, current, index):
            distribution = target._transition_distribution(bound, previous, index)
            return dist.MultivariateNormal(
                distribution.mean, scale_tril=distribution.scale_tril
            ).log_prob(target.project_state(bound, current))

        def native_observation(state, index):
            predictor = observation.linear_predictor(target.project_state(bound, state))
            y = target.observations[index]
            if target.intervals is not None:
                y = jnp.where(
                    target.intervals.operator.point_like_mask(jnp.dtype(bool)), y, jnp.nan
                )
            mask = ~jnp.isnan(y)
            value = jnp.nan_to_num(y, nan=0.0)
            total = jnp.zeros(())
            for group in observation.density_groups():
                indices = jnp.asarray(group.indices)
                law = group.evaluate(
                    predictor, jnp.sqrt(jnp.diag(observation.measurement.manifest_cov)), mask=mask
                )
                if isinstance(law, NormalLawSpec):
                    native = gaussian_distribution(
                        law.loc, observation.measurement.manifest_cov[jnp.ix_(indices, indices)]
                    )
                    total = total + masked_observation_log_prob(
                        native,
                        y=jnp.where(mask[indices], value[indices], law.loc),
                        obs_mask=mask[indices],
                    )
                else:
                    total = total + masked_law_log_prob(law, value[indices], mask[indices])
            return total

        def prepared_transition(previous, current, index):
            return target.transition_log_prob_with_noise(bound, factors, previous, current, index)

        def prepared_observation(state, index):
            return target.observation_increment_with_noise(
                bound, factors, state, index, target.observations
            )

        transition_args = (states[:-1], states[1:], jnp.arange(1, len(times)))
        observed_args = (states, jnp.arange(len(times)))
        expected_transition = jax.vmap(jax.value_and_grad(native_transition, argnums=(0, 1)))(
            *transition_args
        )
        actual_transition = jax.vmap(jax.value_and_grad(prepared_transition, argnums=(0, 1)))(
            *transition_args
        )
        expected_observation = jax.vmap(jax.value_and_grad(native_observation))(*observed_args)
        actual_observation = jax.vmap(jax.value_and_grad(prepared_observation))(*observed_args)
        per_call_transition = jax.vmap(lambda a, b, t: target.transition_log_prob(bound, a, b, t))(
            *transition_args
        )
        per_call_observation = jax.vmap(
            lambda x, t: target.observation_increment(bound, x, t, target.observations)
        )(*observed_args)
        return (
            expected_transition,
            actual_transition,
            expected_observation,
            actual_observation,
            per_call_transition,
            per_call_observation,
        )

    expected_t, actual_t, expected_y, actual_y, per_call_t, per_call_y = jax.jit(jax.vmap(compare))(
        jnp.array([[-0.45], [0.55]])
    )
    for expected, actual in ((expected_t, actual_t), (expected_y, actual_y)):
        for reference, result in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(per_call_t, expected_t[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(per_call_y, expected_y[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual_y[0][:, 2], 0.0)
    np.testing.assert_array_equal(actual_y[1][:, 2], 0.0)
