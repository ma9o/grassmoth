"""Small parameter-contract checks without fitting or forward simulation."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from grassmoth.models.ssm.inference.prior_program import sample_parameters
from grassmoth_test_support.inference import compile_fit_fixture
from grassmoth_test_support.model import two_state_gaussian_model


@pytest.mark.inference(concern="sampling")
def test_parameter_trace_preserves_site_order_shapes_and_public_deterministics():
    dynamical_model_spec = two_state_gaussian_model()
    model = compile_fit_fixture(dynamical_model_spec)
    values = {
        "diffusion_diag_free": jnp.array([0.4, 0.6]),
        "diffusion_lower_free": jnp.array([0.25]),
        "manifest_var_diag_free": jnp.array([0.7, 0.8]),
        "t0_means_free": jnp.array([1.0, -1.0]),
        "t0_var_diag_free": jnp.array([2.0, 3.0]),
        "t0_var_lower_free": jnp.array([0.25]),
    }
    with handlers.substitute(data=values):
        trace = handlers.trace(sample_parameters).get_trace(
            model.compiled_dynamical_model, model.prior_runtime_bundle
        )
    sampled_names = [name for name in trace if name in values]
    assert sampled_names == list(values)
    for name, value in values.items():
        np.testing.assert_array_equal(trace[name]["value"], value)
    assert {name for name, site in trace.items() if site["type"] == "deterministic"} == {
        "diffusion",
        "lambda",
        "manifest_means",
        "manifest_cov",
        "t0_means",
        "t0_cov",
    }
    np.testing.assert_allclose(trace["t0_cov"]["value"], [[4.0, 1.5], [1.5, 9.0]], atol=1e-6)
    np.testing.assert_allclose(trace["diffusion"]["value"], [[0.4, 0.0], [0.25, 0.6]])
    np.testing.assert_allclose(trace["manifest_cov"]["value"], np.diag([0.49, 0.64]), atol=1e-6)
