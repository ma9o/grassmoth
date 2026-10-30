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

from grassmoth.artifacts.model.expressions import coefficient
from grassmoth.artifacts.model.likelihood import (
    GammaLawSpec,
    NegativeBinomial2LawSpec,
    NormalLawSpec,
    PoissonLawSpec,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.execution.observation_distributions import (
    gaussian_distribution,
    masked_law_log_prob,
)
from grassmoth.models.ssm.execution.observation_model import (
    grouped_density,
    grouped_density_with_noise,
    prepare_grouped_density,
)
from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth.models.ssm.inference.targets.gaussian_reference import GaussianReference
from grassmoth_test_support.interval_particle import interval_target
from grassmoth_test_support.observation import observation_laws

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(
    ("family", "state_concentration"),
    [
        (DistributionFamily.GAMMA, False),
        (DistributionFamily.GAMMA, True),
        (DistributionFamily.NEGATIVE_BINOMIAL, False),
        (DistributionFamily.NEGATIVE_BINOMIAL, True),
        (DistributionFamily.POISSON, False),
    ],
)
def test_prepared_point_terms_preserve_authored_operands_and_derivatives(
    family, state_concentration
):
    """Fixed event terms retain learned parameters, arbitrary responses and missingness."""
    values = jnp.array([[0.2, jnp.nan], [2.0, 1.0], [jnp.nan, jnp.nan]])
    if family != DistributionFamily.GAMMA:
        values = values.at[0, 0].set(0).at[1, 0].set(3)

    def scores(parameters, readings, prepared):
        laws = observation_laws(
            (family, family),
            parameters={"obs_shape": jnp.exp(parameters[0]), "obs_r": jnp.exp(parameters[0])},
        )
        authored = []
        for law in laws:
            assert isinstance(law, (GammaLawSpec, NegativeBinomial2LawSpec, PoissonLawSpec))
            if isinstance(law, (GammaLawSpec, NegativeBinomial2LawSpec)):
                concentration = replace(
                    law.concentration,
                    expression=law.concentration.expression
                    if state_concentration
                    else coefficient(None, "shape"),
                    evaluate_fn=(lambda eta, _scale, values: values[0] * jnp.exp(0.2 * eta))
                    if state_concentration
                    else law.concentration.evaluate_fn,
                )
                law = (
                    GammaLawSpec(concentration=concentration, rate=law.rate)
                    if isinstance(law, GammaLawSpec)
                    else NegativeBinomial2LawSpec(concentration=concentration, mean=law.mean)
                )
            # This response is invalid at the preparation placeholder, yet valid at
            # the scored states. Its arbitrary expression must not be replaced by a link.
            operand = law.mean if isinstance(law, NegativeBinomial2LawSpec) else law.rate
            operand = replace(
                operand,
                expression=laws[0].operands()[0][1].expression - 2,
                evaluate_fn=lambda eta, _scale, _values: eta - 2,
            )
            if isinstance(law, GammaLawSpec):
                authored.append(GammaLawSpec(concentration=law.concentration, rate=operand))
            elif isinstance(law, NegativeBinomial2LawSpec):
                authored.append(
                    NegativeBinomial2LawSpec(concentration=law.concentration, mean=operand)
                )
            else:
                authored.append(PoissonLawSpec(rate=operand))
        model = HeterogeneousObservation(
            MeasurementParams(jnp.eye(2), jnp.zeros(2), jnp.eye(2)), tuple(authored)
        )
        groups = model.density_groups()
        factors = prepare_grouped_density(groups, model.measurement.manifest_cov, readings)
        predictors = jnp.array(
            [
                [parameters[1], jnp.nan],
                [parameters[1] + 0.5, parameters[1] + 0.2],
                [jnp.nan, jnp.nan],
            ]
        )

        def density(index):
            y, eta = readings[index], predictors[index]
            if prepared:
                return grouped_density_with_noise(model, y, eta, factors.at(index))
            return grouped_density(
                groups,
                jnp.nan_to_num(y, nan=0.0),
                eta,
                model.measurement.manifest_cov,
                ~jnp.isnan(y),
            )

        return jax.vmap(density)(jnp.arange(3))

    parameters = jnp.array([0.4, 2.6])
    actual = jax.jit(jax.value_and_grad(lambda p: scores(p, values, True).sum()))(parameters)
    expected = jax.jit(jax.value_and_grad(lambda p: scores(p, values, False).sum()))(parameters)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        jax.hessian(lambda p: scores(p, values, True).sum())(parameters),
        jax.hessian(lambda p: scores(p, values, False).sum())(parameters),
        atol=3e-5,
        rtol=3e-5,
    )
    for readings, invalid_parameters in (
        (values.at[0, 0].set(-1), parameters),
        (values, parameters.at[1].set(1.0)),
    ):
        np.testing.assert_allclose(
            scores(invalid_parameters, readings, True),
            scores(invalid_parameters, readings, False),
            atol=2e-5,
            rtol=2e-5,
        )


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
            mean = target.prepare_transition_mean(bound, previous, index)
            return target.transition_log_prob_with_mean(
                bound, factors, previous, current, index, mean
            )

        def prepared_observation(state, index):
            return target.observation_increment_with_noise(bound, factors, state, index)

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


@pytest.mark.parametrize("translated", [False, True])
def test_prepared_transition_means_preserve_bridge_densities(translated):
    times = np.array([0.0, 0.7, 1.8])
    base = interval_target(times, np.full((3, 1), 0.2), times[:, None], ("last",), state_dim=2)
    reference = GaussianReference(
        parameter_mean=jnp.zeros(1),
        parameter_chol=jnp.eye(1),
        path_mean=jnp.zeros((3, 2)),
        autoregression=jnp.broadcast_to(0.4 * jnp.eye(2), (3, 2, 2)),
        innovation_chol=jnp.broadcast_to(0.7 * jnp.eye(2), (3, 2, 2)),
        free_mask=base.independent_mask,
    )
    predecessors = jnp.array([[0.2, -0.4], [-0.3, 0.7]])
    successors = jnp.array([[0.6, 0.1], [-0.2, -0.5]])
    indices = jnp.arange(1, len(times))
    position = jnp.array([0.45])
    chart = TranslationChart(jnp.array([[-0.8], [0.3]]), jnp.array([-0.2]))
    chart_previous = chart.encode(position, predecessors) if translated else predecessors
    chart_current = chart.encode(position, successors) if translated else successors
    physical_path = jnp.concatenate((predecessors[:1], successors))
    path = chart.encode(position, physical_path) if translated else physical_path

    def compare(beta):
        physical = base.at_temperature(reference, beta)
        target = physical.in_coordinates(chart) if translated else physical
        context = target.context(position, target.times)
        noise = target.prepare_noise_factors(context, target.observations)
        means = jax.vmap(
            lambda previous, index: target.prepare_transition_mean(context, previous, index)
        )(chart_previous, indices)
        prepared = jax.vmap(
            lambda previous, current, index, mean: target.transition_log_prob_with_mean(
                context, noise, previous, current, index, mean
            )
        )(chart_previous, chart_current, indices, means)
        declared = jax.vmap(
            lambda previous, current, index: target.transition_log_prob(
                context, previous, current, index
            )
        )(chart_previous, chart_current, indices)

        def expected(parameter, coordinates):
            decoded = chart.decode(parameter, coordinates) if translated else coordinates
            return physical.log_posterior(parameter, decoded, physical.observations, physical.times)

        def actual(parameter, coordinates):
            return target.log_posterior(parameter, coordinates, target.observations, target.times)

        return (
            prepared,
            declared,
            jax.value_and_grad(actual, argnums=(0, 1))(position, path),
            jax.value_and_grad(expected, argnums=(0, 1))(position, path),
        )

    prepared, declared, actual, expected = jax.jit(jax.vmap(compare))(jnp.array([0.0, 0.4, 1.0]))
    np.testing.assert_allclose(prepared, declared, rtol=1e-5, atol=1e-6)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)
