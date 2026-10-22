"""Parity tests for shared predictor- and mean-space observation draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.model.likelihood import (
    GammaLawSpec,
    LinkFunction,
    NegativeBinomial2LawSpec,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.observation_dispatch import _sample_native
from grassmoth.models.ssm.execution.observation_distributions import (
    evaluate_law,
    prepare_response_log_prob,
    to_native,
)
from grassmoth.models.ssm.execution.observation_model import compile_observation_model
from tests.support.observation import (
    mean_density,
    mean_sampler,
    observation_kernel,
    observation_laws,
)

if TYPE_CHECKING:
    from tests.support.observation import ObservationParameters


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("family", ["gamma", "negative_binomial"])
def test_prepared_interval_density_preserves_parameter_and_response_derivatives(family):
    """One bound factor scores many candidates without freezing parameter dependence."""
    values = jnp.array([0.2, jnp.nan]) if family == "gamma" else jnp.array([4.0, jnp.nan])
    mask = jnp.array([1.0, 0.0])
    offsets = jnp.array([-1.5, 0.0, 1.8])

    def score(parameters, prepared):
        concentration = jnp.exp(parameters[0]) * jnp.ones(2)
        responses = jnp.exp(parameters[1] + offsets)
        if prepared:
            law = (
                GammaLawSpec(concentration=concentration, rate=concentration)
                if family == "gamma"
                else NegativeBinomial2LawSpec(mean=jnp.ones(2), concentration=concentration)
            )
            bound = prepare_response_log_prob(law, values, mask)
            return jax.vmap(lambda response: bound(jnp.array([response, jnp.nan])))(responses).sum()
        if family == "gamma":
            return (
                dist.Gamma(concentration[0], concentration[0] / responses).log_prob(values[0]).sum()
            )
        return dist.NegativeBinomial2(responses, concentration[0]).log_prob(values[0]).sum()

    for parameters in (jnp.array([-2.0, -0.3]), jnp.array([1.1, 0.4])):
        actual = jax.jit(jax.value_and_grad(lambda p: score(p, True)))(parameters)
        expected = jax.value_and_grad(lambda p: score(p, False))(parameters)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            jax.hessian(lambda p: score(p, True))(parameters),
            jax.hessian(lambda p: score(p, False))(parameters),
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("family", "link", "predictor", "mean", "extra_params", "std"),
    [
        (DistributionFamily.DELTA, LinkFunction.IDENTITY, -0.4, -0.4, {}, 0.0),
        (DistributionFamily.STUDENT_T, LinkFunction.IDENTITY, 0.4, 0.4, {"obs_df": 4.0}, 0.7),
        (DistributionFamily.POISSON, LinkFunction.LOG, 0.4, float(jnp.exp(0.4)), {}, 1.0),
        (
            DistributionFamily.GAMMA,
            LinkFunction.LOG,
            0.4,
            float(jnp.exp(0.4)),
            {"obs_shape": 2.0},
            1.0,
        ),
        (DistributionFamily.GAMMA, LinkFunction.INVERSE, 2.0, 0.5, {"obs_shape": 2.0}, 1.0),
        (
            DistributionFamily.BERNOULLI,
            LinkFunction.LOGIT,
            0.4,
            float(jax.nn.sigmoid(0.4)),
            {},
            1.0,
        ),
        (
            DistributionFamily.BERNOULLI,
            LinkFunction.PROBIT,
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {},
            1.0,
        ),
        (
            DistributionFamily.NEGATIVE_BINOMIAL,
            LinkFunction.LOG,
            0.4,
            float(jnp.exp(0.4)),
            {"obs_r": 3.0},
            1.0,
        ),
        (
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            0.4,
            float(jax.nn.sigmoid(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
        (
            DistributionFamily.BETA,
            LinkFunction.PROBIT,
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
    ],
)
def test_predictor_and_mean_samplers_use_the_same_draw(
    family: DistributionFamily,
    link: LinkFunction,
    predictor: float,
    mean: float,
    extra_params: ObservationParameters,
    std: float,
) -> None:
    key = jax.random.PRNGKey(42)
    bound = observation_laws([family], [link], extra_params)[0]
    point_draw = _sample_native(
        key, evaluate_law(bound, jnp.asarray(predictor), jnp.asarray(std)), 0
    )
    mean_draw = mean_sampler(observation_laws([family], parameters=extra_params)[0])(
        key,
        jnp.asarray([mean]),
        jnp.asarray([[std**2]]),
    )[0]

    np.testing.assert_array_equal(point_draw, mean_draw)


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("family", "extra_params", "invalid_mean"),
    [
        (DistributionFamily.POISSON, {}, -1.0),
        (DistributionFamily.GAMMA, {"obs_shape": 2.0}, 0.0),
        (DistributionFamily.BERNOULLI, {}, 1.1),
        (DistributionFamily.NEGATIVE_BINOMIAL, {"obs_r": 3.0}, -1.0),
        (DistributionFamily.BETA, {"obs_concentration": 8.0}, 0.0),
    ],
)
def test_mean_samplers_surface_invalid_domains_as_nan(
    family: DistributionFamily,
    extra_params: ObservationParameters,
    invalid_mean: float,
) -> None:
    draw = mean_sampler(observation_laws([family], parameters=extra_params)[0])(
        jax.random.PRNGKey(0),
        jnp.asarray([invalid_mean]),
        jnp.eye(1),
    )

    assert jnp.isnan(draw[0])


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_zero_mean_negative_binomial_is_an_exact_point_mass():
    law = to_native(
        NegativeBinomial2LawSpec(mean=jnp.array([0.0, 2.0]), concentration=jnp.asarray(4.0))
    )
    assert law.sample(jax.random.PRNGKey(3))[0] == 0.0
    assert law.log_prob(jnp.array([0, 0]))[0] == 0.0
    assert jnp.isneginf(law.log_prob(jnp.array([1, 1]))[0])
    np.testing.assert_allclose(law.mean, [0.0, 2.0])
    np.testing.assert_allclose(law.variance, [0.0, 3.0])


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_beta_sampler_and_density_preserve_small_authored_shapes():
    mean = jnp.array([0.001])
    concentration = 0.1
    key = jax.random.PRNGKey(12)
    native = dist.Beta(mean * concentration, (1.0 - mean) * concentration)
    extras: ObservationParameters = {"obs_concentration": concentration}
    sample = mean_sampler(observation_laws([DistributionFamily.BETA], parameters=extras)[0])(
        key, mean, jnp.eye(1)
    )
    np.testing.assert_array_equal(sample, native.sample(key))
    density = mean_density(observation_laws([DistributionFamily.BETA], parameters=extras)[0])(
        jnp.array([0.2]), mean, jnp.eye(1), jnp.ones(1)
    )
    np.testing.assert_allclose(density, native.log_prob(0.2).sum(), atol=1e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_binary_boundaries_and_predictor_tails_remain_exact():
    likelihood = mean_density(observation_laws([DistributionFamily.BERNOULLI], parameters=None)[0])
    mean = jnp.array([0.0, 1.0])
    assert likelihood(mean, mean, jnp.eye(2), jnp.ones(2)) == 0.0
    assert jnp.isneginf(likelihood(1.0 - mean, mean, jnp.eye(2), jnp.ones(2)))
    np.testing.assert_array_equal(
        mean_sampler(observation_laws([DistributionFamily.BERNOULLI], parameters=None)[0])(
            jax.random.key(9), mean, jnp.eye(2)
        ),
        mean,
    )
    assert float(
        observation_kernel(
            [DistributionFamily.BERNOULLI] * (jnp.array([100.0])).shape[-1],
            [LinkFunction.LOGIT] * (jnp.array([100.0])).shape[-1],
            None,
        ).log_prob_fn(jnp.zeros(1), jnp.array([100.0]), jnp.eye(1), jnp.ones(1))
    ) == pytest.approx(-100.0)


@pytest.mark.inference(concern="sampling")
def test_negative_binomial_density_has_the_exact_mean_gradient():
    likelihood = mean_density(
        observation_laws([DistributionFamily.NEGATIVE_BINOMIAL], parameters={"obs_r": 3.0})[0]
    )
    y, mean = jnp.array([0.0, 4.0]), jnp.array([1.5, 8.0])

    def fn(mu):
        return likelihood(y, mu, jnp.eye(2), jnp.ones(2))

    expected = y / mean - (3.0 + y) / (3.0 + mean)
    np.testing.assert_allclose(jax.grad(fn)(mean), expected, atol=2e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_negative_binomial_tail_values_gradients_and_hessians_match_double_reference(dtype):
    """Finite log probabilities must retain finite derivatives through extreme log means."""
    points = np.array(
        [(eta, log_r) for eta in (-80, -60, -40, 0, 40, 60, 80) for log_r in (-2, 1.17, 4)]
    )

    def actual(z, count):
        return to_native(
            NegativeBinomial2LawSpec(mean=jnp.exp(z[0]), concentration=jnp.exp(z[1]))
        ).log_prob(jnp.asarray(count, dtype=z.dtype))

    def reference(z, count):
        mean, r = jnp.exp(z)
        # Counts zero and two have elementary normalizers, independent of the
        # density implementation and generic gammaln/betaln approximations.
        zero_count = -r * jnp.log1p(mean / r)
        if count == 0.0:
            return zero_count
        return (
            jnp.log(r) + jnp.log1p(r) - jnp.log(2.0) + zero_count + 2 * (z[0] - jnp.log(r + mean))
        )

    def derivatives(fn, coordinates, count):
        return jax.jit(
            jax.vmap(lambda z: (jax.value_and_grad(fn)(z, count), jax.hessian(fn)(z, count)))
        )(coordinates)

    with jax.enable_x64():
        for count in (0.0, 2.0):
            observed = derivatives(actual, jnp.asarray(points, dtype=dtype), count)
            expected = derivatives(reference, jnp.asarray(points, dtype=jnp.float64), count)
            tolerance = 8e-5 if dtype == "float32" else 2e-10
            for value, target in zip(
                jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True
            ):
                assert np.isfinite(value).all()
                np.testing.assert_allclose(value, target, rtol=tolerance, atol=tolerance)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("family", "extras"),
    [
        (DistributionFamily.GAMMA, {"obs_shape": 2.0}),
        (DistributionFamily.BETA, {"obs_concentration": 3.0}),
        (DistributionFamily.NEGATIVE_BINOMIAL, {"obs_r": 3.0}),
    ],
)
def test_missing_invalid_observation_has_zero_gradient(family: DistributionFamily, extras):
    likelihood = mean_density(observation_laws([family], parameters=extras)[0])
    value = 2.0 if family == DistributionFamily.NEGATIVE_BINOMIAL else 0.2

    def fn(mu):
        return likelihood(jnp.array([value, jnp.nan]), mu, jnp.eye(2), jnp.array([1.0, 0.0]))

    gradient = jax.grad(fn)(jnp.array([0.4, 0.6]))
    assert jnp.isfinite(gradient[0])
    assert gradient[1] == 0.0


@pytest.mark.inference(concern="sampling")
def test_ragged_category_density_and_all_operand_gradients_match_native_laws():
    """Real category widths determine density; missing rows and padded operands are inert."""
    families = (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL) * 3
    levels = (2, 2, 3, 3, 5, 5)
    with jax.enable_x64():
        predictors = jnp.linspace(-0.5, 0.5, 6)
        cutpoints = jnp.broadcast_to(jnp.linspace(-1.0, 1.0, 4), (6, 4))
        intercepts = jnp.broadcast_to(jnp.array([-0.3, 0.2, 0.1, -0.1]), (6, 4))
        slopes = jnp.broadcast_to(jnp.array([0.4, -0.2, 0.1, 0.3]), (6, 4))
        observations = jnp.array([0.0, 1.0, jnp.nan, jnp.nan, 3.0, 2.0])
        mask = jnp.isfinite(observations)

        def actual(eta, thresholds, offsets, weights):
            model = compile_observation_model(
                observation_laws(
                    families,
                    None,
                    {
                        "obs_level_counts": jnp.array(levels),
                        "obs_ordered_cutpoints": thresholds,
                        "obs_cat_intercepts": offsets,
                        "obs_cat_slopes": weights,
                    },
                    level_counts=levels,
                ),
                manifest_cov=jnp.eye(6),
            )
            return model.kernel.log_prob_fn(observations, eta, jnp.eye(6), mask)

        def expected(eta, thresholds, offsets, weights):
            total = jnp.asarray(0.0)
            for column in (0, 1, 4, 5):
                width = levels[column] - 1
                if families[column] == DistributionFamily.ORDERED_LOGISTIC:
                    native = dist.OrderedLogistic(eta[column], thresholds[column, :width])
                else:
                    logits = jnp.concatenate(
                        (
                            jnp.zeros(1),
                            offsets[column, :width] + weights[column, :width] * eta[column],
                        )
                    )
                    native = dist.CategoricalLogits(logits=logits)
                total = total + native.log_prob(observations[column].astype(jnp.int32))
            return total

        operands = (predictors, cutpoints, intercepts, slopes)
        value, gradients = jax.jit(jax.value_and_grad(actual, argnums=(0, 1, 2, 3)))(*operands)
        reference, reference_gradients = jax.value_and_grad(expected, argnums=(0, 1, 2, 3))(
            *operands
        )
        np.testing.assert_allclose(value, reference, rtol=0, atol=1e-12)
        for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
            assert jnp.isfinite(gradient).all()
            np.testing.assert_allclose(gradient, reference_gradient, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(gradients[0][2:4], 0.0)


@pytest.mark.inference(concern="sampling")
def test_zero_mean_negative_binomial_has_the_exact_boundary_gradient():
    """The right derivative of log P(Y=0) at mean zero is -1, without an epsilon mean."""
    likelihood = mean_density(
        observation_laws([DistributionFamily.NEGATIVE_BINOMIAL], parameters={"obs_r": 3.0})[0]
    )

    def score(mean):
        return likelihood(jnp.zeros(1), jnp.atleast_1d(mean), jnp.eye(1), jnp.ones(1))

    assert score(jnp.asarray(0.0)) == 0.0
    assert jax.grad(score)(jnp.asarray(0.0)) == -1.0
    assert float(jax.grad(jax.grad(score))(jnp.asarray(0.0))) == pytest.approx(1.0 / 3.0)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("family", "extras", "invalid"),
    [
        (DistributionFamily.GAMMA, {"obs_shape": 0.01}, (0.0, -0.1, jnp.inf)),
        (DistributionFamily.BETA, {"obs_concentration": 0.1}, (0.0, 1.0, jnp.inf)),
        (DistributionFamily.POISSON, {}, (-1.0, 0.5, jnp.inf)),
        (DistributionFamily.NEGATIVE_BINOMIAL, {"obs_r": 0.01}, (-1.0, 0.5, jnp.inf)),
        (DistributionFamily.BERNOULLI, {}, (-1.0, 0.5, jnp.inf)),
    ],
)
def test_invalid_observation_support_is_rejected_and_missing_derivatives_are_zero(
    family, extras, invalid
):
    density = mean_density(observation_laws([family], parameters=extras)[0])
    for value in invalid:
        assert jnp.isneginf(density(jnp.array([value]), jnp.array([0.4]), jnp.eye(1), jnp.ones(1)))

    def absent(mean):
        return density(jnp.array([jnp.nan]), jnp.atleast_1d(mean), jnp.eye(1), jnp.zeros(1))

    value, derivative = jax.value_and_grad(absent)(jnp.asarray(0.4))
    assert value == 0.0
    assert derivative == 0.0
