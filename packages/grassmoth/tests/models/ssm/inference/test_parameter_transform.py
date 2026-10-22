"""Prior-coordinate preparation preserves conditional laws and initialization rejection."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from grassmoth.models.ssm.inference.parameter_transform import prepare_parameter_transform
from grassmoth.models.ssm.inference.prior_program import numpyro_model
from grassmoth.models.ssm.inference.reparameterization import AutoReparam
from tests.support.inference import bind_panel_fixture, compile_fit_fixture
from tests.support.model import two_state_gaussian_model

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


@pytest.mark.parametrize("centered", [None, 0.0, 0.5])
def test_prior_replay_omits_unused_likelihood_without_changing_density(centered):
    """Structural replay retains the zero-backend sites, transforms, factors and gradients."""

    class ZeroLikelihood:
        def compute_log_likelihood(self, *_args, **_kwargs):
            return jnp.array(0.0)

    compiled = compile_fit_fixture(two_state_gaussian_model())
    panel = bind_panel_fixture(
        compiled.compiled_dynamical_model,
        jnp.zeros((5, 2), dtype=jnp.float32),
        jnp.arange(5, dtype=jnp.float32),
    )
    models = tuple(
        partial(numpyro_model, panel, compiled.prior_runtime_bundle, likelihood_backend=backend)
        for backend in (ZeroLikelihood(), None)
    )
    if centered is not None:
        models = tuple(
            numpyro.handlers.reparam(model, config=AutoReparam(centered=centered))
            for model in models
        )
    traces = tuple(
        numpyro.handlers.trace(numpyro.handlers.seed(model, rng_seed=19)).get_trace()
        for model in models
    )
    assert tuple(traces[0]) == tuple(traces[1])
    for name, site in traces[0].items():
        if site["type"] in ("sample", "deterministic"):
            np.testing.assert_allclose(
                traces[1][name]["value"], site["value"], rtol=1e-5, atol=1e-6
            )
    values = {
        name: site["value"]
        for name, site in traces[0].items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    transforms = tuple(
        prepare_parameter_transform(model, jax.random.key(19), model_args=(), initial_values=values)
        for model in models
    )
    np.testing.assert_allclose(
        transforms[1].initial_position, transforms[0].initial_position, rtol=1e-5, atol=1e-6
    )
    for offset in (-0.1, 0.0, 0.1):
        position = transforms[0].initial_position + offset
        reference = jax.value_and_grad(transforms[0].log_prior)(position)
        actual = jax.value_and_grad(transforms[1].log_prior)(position)
        for value, expected in zip(actual, reference, strict=True):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-6)
