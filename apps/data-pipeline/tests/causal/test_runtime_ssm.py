"""Tests for structural tracing and component-native dynamics specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest
from dynestyx import StochasticContinuousTimeStateEvolution

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.models.ssm.dynamics import (
    DynamicsSpec,
    Intervention,
    VectorFieldArgs,
    compile_dynamics,
)
from nof1_causal_lab.models.ssm.model import numpyro_model
from tests.dynamics_fixtures import decay_term, intercept_term, linear_term
from tests.inference_fixtures import bind_panel_fixture, compile_fit_fixture
from tests.model_fixtures import load_model_fixture


def _nonlinear_dynamics_uses_vector_field_backend_method_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "runtime_ssm/testssmmodeldynamicsdispatch_test_nonlinear_dynamics_uses_vector_field_backend_method_model_fixture.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec


@pytest.mark.inference(concern="simulation")
class TestCompiledModelDynamicsDispatch:
    """The NumPyro model samples dynamics and delegates at the backend boundary."""

    def test_nonlinear_dynamics_uses_vector_field_backend_method(self):
        import numpyro
        from numpyro import handlers

        class DynamicsAwareBackend:
            def compute_log_likelihood(
                self,
                dynamics,
                _measurement_params,
                _initial_state,
                _observations,
                time_intervals,
                **_kwargs,
            ):
                assert isinstance(dynamics, StochasticContinuousTimeStateEvolution)
                numpyro.deterministic(
                    "backend_n_vf_components",
                    jnp.asarray(len(dynamics.drift.args.params)),
                )
                numpyro.deterministic(
                    "backend_drift", dynamics.drift(jnp.ones(2), jnp.empty(0), 0.0)
                )
                return jnp.zeros_like(time_intervals)

        dynamical_model_spec = _nonlinear_dynamics_uses_vector_field_backend_method_model_fixture()
        model = compile_fit_fixture(dynamical_model_spec)
        tr = handlers.trace(handlers.seed(numpyro_model, rng_seed=0)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((4, 2)), jnp.arange(4, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=DynamicsAwareBackend(),
        )

        assert "vf_0_p0" not in tr
        assert int(tr["backend_n_vf_components"]["value"]) == 2
        np.testing.assert_allclose(
            tr["backend_drift"]["value"],
            np.array([-0.3, -0.5]),
        )


class TestComponentNativeLinearDynamics:
    """Numerical pins for linear dynamics expressed as first-class components."""

    @pytest.mark.inference(concern="simulation")
    def test_state_decay_and_edges_match_expected_vector_field(self):
        from numpyro.handlers import condition, seed

        spec = DynamicsSpec(
            n_latent=3,
            components=(
                decay_term(target=0),
                decay_term(target=1),
                decay_term(target=2),
                linear_term(source=1, target=0),
                linear_term(source=2, target=1),
                linear_term(source=0, target=2),
            ),
        )
        compiled = compile_dynamics(spec)
        sample_values = {
            "vf_0_p0": jnp.asarray(0.3),
            "vf_1_p0": jnp.asarray(0.5),
            "vf_2_p0": jnp.asarray(0.7),
            "vf_3_p0": jnp.asarray(0.2),
            "vf_4_p0": jnp.asarray(-0.1),
            "vf_5_p0": jnp.asarray(0.4),
        }
        with (
            seed(rng_seed=0),
            condition(data=sample_values),
        ):
            params = compiled.sample_params(lambda site_name: ndist.Delta(sample_values[site_name]))

        eta = jnp.array([1.0, 2.0, 3.0])
        actual = compiled.vector_field(
            jnp.asarray(0.0),
            eta,
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        expected = jnp.array(
            [
                -0.3 * 1.0 + 0.2 * 2.0,
                -0.5 * 2.0 - 0.1 * 3.0,
                -0.7 * 3.0 + 0.4 * 1.0,
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    @pytest.mark.inference(concern="sampling")
    def test_delta_state_decay_samples_component_param(self):
        from numpyro.handlers import seed

        spec = DynamicsSpec(
            n_latent=2,
            components=(decay_term(target=1),),
        )
        compiled = compile_dynamics(spec)
        with seed(rng_seed=0):
            params = compiled.sample_params(lambda _: ndist.Delta(jnp.asarray(0.4)))

        decay = params[0][scientific_id("parameter", "decay")]
        assert decay.shape == ()
        np.testing.assert_allclose(decay, 0.4, atol=1e-12)

    @pytest.mark.inference(concern="simulation")
    def test_state_intercepts_add_to_selected_targets(self):
        from numpyro.handlers import condition, seed

        spec = DynamicsSpec(
            n_latent=3,
            components=(
                intercept_term(target=0),
                intercept_term(target=2),
            ),
        )
        compiled = compile_dynamics(spec)
        sample_values = {
            "vf_0_p0": jnp.asarray(0.1),
            "vf_1_p0": jnp.asarray(-0.2),
        }
        with (
            seed(rng_seed=0),
            condition(data=sample_values),
        ):
            params = compiled.sample_params(lambda site_name: ndist.Delta(sample_values[site_name]))

        actual = compiled.vector_field(
            jnp.asarray(0.0),
            jnp.zeros(3),
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        np.testing.assert_allclose(actual, jnp.array([0.1, 0.0, -0.2]), atol=1e-12)
