"""Prior-coordinate preparation preserves conditional laws and initialization rejection."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.models.ssm.inference.parameter_transform import prepare_parameter_transform

pytestmark = pytest.mark.inference(concern="sampling")


def test_prepared_prior_keeps_conditional_support_factors_and_deterministics():
    def prior():
        scale = numpyro.sample("scale", dist.LogNormal(-0.2, 0.4))
        location = numpyro.sample("location", dist.Normal(scale, 0.8))
        bounded = numpyro.sample("bounded", dist.Uniform(location - 2.0, location + 3.0))
        numpyro.factor("coupled", -0.5 * (bounded - scale) ** 2)
        numpyro.deterministic("summary", bounded + location)

    values = {"scale": jnp.array(1.2), "location": jnp.array(0.4), "bounded": jnp.array(0.9)}
    parameters = prepare_parameter_transform(
        prior, jax.random.key(19), model_args=(), initial_values=values
    )
    constrained = parameters.constrain(parameters.initial_position)
    for name, value in values.items():
        np.testing.assert_allclose(constrained[name], value, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(constrained["summary"], 1.3, rtol=1e-5)

    def analytic(position):
        sites = parameters.unravel(position)
        scale = jnp.exp(sites["scale"])
        location = sites["location"]
        logit = sites["bounded"]
        bounded = location - 2.0 + 5.0 * jax.nn.sigmoid(logit)
        return (
            dist.Normal(-0.2, 0.4).log_prob(sites["scale"])
            + dist.Normal(scale, 0.8).log_prob(location)
            - jax.nn.softplus(-logit)
            - jax.nn.softplus(logit)
            - 0.5 * (bounded - scale) ** 2
        )

    for offset in (-0.3, 0.0, 0.4):
        position = parameters.initial_position + offset
        actual = jax.value_and_grad(parameters.log_prior)(position)
        expected = jax.value_and_grad(analytic)(position)
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)


def test_prepared_prior_rejects_a_finite_start_with_a_nonfinite_gradient():
    def prior():
        value = numpyro.sample("value", dist.Normal(0.0, 1.0))
        numpyro.factor("singular_gradient", jnp.sqrt(jnp.abs(value)))

    with pytest.raises(RuntimeError, match="Cannot find valid initial parameters"):
        prepare_parameter_transform(
            prior,
            jax.random.key(19),
            model_args=(),
            initial_values={"value": jnp.array(0.0)},
        )
