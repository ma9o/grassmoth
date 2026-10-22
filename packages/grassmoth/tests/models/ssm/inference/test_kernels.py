"""Tests for the observation-kernel layer.

Covers exact predictor operations and build_observation_kernel.
"""

import jax
import jax.numpy as jnp
import pytest

from grassmoth.artifacts.model.likelihood import LinkFunction
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.observation_model import compile_observation_model
from tests.support.observation import observation_kernel, observation_laws


class TestBuildObservationKernel:
    @pytest.mark.contract
    @pytest.mark.parametrize(
        ("family", "link", "parameter"),
        [
            (DistributionFamily.STUDENT_T, LinkFunction.IDENTITY, "obs_df"),
            (DistributionFamily.GAMMA, LinkFunction.LOG, "obs_shape"),
            (DistributionFamily.NEGATIVE_BINOMIAL, LinkFunction.LOG, "obs_r"),
            (DistributionFamily.BETA, LinkFunction.LOGIT, "obs_concentration"),
        ],
    )
    def test_active_observation_hyperparameter_is_required(self, family, link, parameter):
        with pytest.raises(KeyError, match=parameter):
            compile_observation_model(
                observation_laws([family], [link], None), manifest_cov=jnp.eye(1)
            )

    @pytest.mark.inference(concern="sampling")
    @pytest.mark.inference(concern="predictive")
    def test_compiled_model_shares_predictor_semantics_for_likelihood_and_sampling(self):
        manifest_cov = jnp.eye(1)
        model = compile_observation_model(
            observation_laws([DistributionFamily.POISSON], [LinkFunction.LOG], None),
            manifest_cov=manifest_cov,
        )
        eta = jnp.array([jnp.log(3.0)])
        log_prob = model.kernel.log_prob_fn(
            jnp.array([2.0]),
            eta,
            manifest_cov,
            jnp.ones(1),
        )
        expected = jax.scipy.stats.poisson.logpmf(2.0, 3.0)
        assert jnp.isclose(log_prob, expected)

        draws = jax.vmap(model.point_sampler.sample_point)(
            jax.random.split(jax.random.PRNGKey(0), 1),
            eta[None, :],
        )
        assert draws.shape == (1, 1)
        assert draws[0, 0] >= 0
        assert jnp.isclose(draws[0, 0], jnp.rint(draws[0, 0]))

    @pytest.mark.inference(concern="predictive")
    def test_beta_kernel_accepts_traced_positive_site(self):
        @jax.jit
        def _build_response(obs_concentration):
            kernel = observation_kernel(
                [DistributionFamily.BETA],
                [LinkFunction.LOGIT],
                {"obs_concentration": obs_concentration},
            )
            return kernel.response_fn(jnp.array([0.0]))

        response = _build_response(jnp.array(9.0))
        assert jnp.isclose(response[0], 0.5)

    @pytest.mark.inference(concern="sampling")
    def test_beta_density_rejects_nonpositive_native_concentrations(self):
        kernel = observation_kernel(
            [DistributionFamily.BETA], [LinkFunction.LOGIT], {"obs_concentration": 0.0}
        )
        assert jnp.isneginf(
            kernel.log_prob_fn(jnp.array([0.5]), jnp.zeros(1), jnp.eye(1), jnp.ones(1))
        )

    @pytest.mark.contract
    def test_gamma_inverse_response_marks_invalid_eta_as_nan(self):
        kernel = observation_kernel(
            [DistributionFamily.GAMMA] * 2, [LinkFunction.INVERSE] * 2, {"obs_shape": 2.0}
        )
        response = kernel.response_fn(jnp.array([2.0, -0.5]))
        assert jnp.isclose(response[0], 0.5)
        assert jnp.isnan(response[1])

    @pytest.mark.contract
    def test_recognized_but_invalid_family_link_pair_raises(self):
        with pytest.raises(ValueError, match="invalid for observation family 'gaussian'"):
            observation_kernel([DistributionFamily.GAUSSIAN], [LinkFunction.LOG], None)

    @pytest.mark.contract
    def test_gaussian_response_is_identity(self):
        kernel = observation_kernel([DistributionFamily.GAUSSIAN] * 2, [LinkFunction.IDENTITY] * 2)
        x = jnp.array([1.0, -2.0])
        assert jnp.allclose(kernel.response_fn(x), x)
