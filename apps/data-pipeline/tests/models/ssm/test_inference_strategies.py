"""Contracts and numerical references for observation likelihoods and inference routing."""

from __future__ import annotations

import time

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    compile_observation_operator,
    expected_observation_mean,
    trajectory_observation_log_probs,
)
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.sampler_config import SamplerSpec
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.inference_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    make_observation_support_runtime,
    particle_posterior,
)
from tests.model_fixtures import one_state_gaussian_model
from tests.observation_fixtures import mean_density, observation_kernel, observation_laws


@pytest.mark.contract
def test_compile_observation_operator_keeps_point_support_without_interval_summary_mode():
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["pulse"],
        support_kinds=["point"],
        observation_windows=[None],
        support_start_times=np.array([[0.0], [1.0]]),
        support_end_times=np.array([[0.0], [1.0]]),
    )

    operator = compile_observation_operator(support)

    assert operator.support_kind_codes is not None
    assert operator.summary_operator_codes is not None
    assert operator.requires_interval_summary_handling is False
    assert operator.interval_summary_indices == ()
    np.testing.assert_array_equal(
        np.asarray(operator.point_like_mask(jnp.float32)),
        np.array([1.0], dtype=np.float32),
    )


@pytest.mark.inference(concern="predictive")
def test_held_windows_preserve_the_shared_plan_and_emission_schedule():
    times = np.array([0.0, 1.0, 3.0, 4.0, 6.0])
    support = make_observation_support_runtime(
        anchor_times=times,
        manifest_names=("mean",),
        support_kinds=("interval",),
        observation_windows=("2d",),
        support_start_times=np.array([[np.nan], [np.nan], [0.5], [2.0], [4.0]]),
        support_end_times=np.array([[np.nan], [np.nan], [3.0], [4.0], [6.0]]),
    )
    linear = compile_observation_operator(support)
    held = compile_observation_operator(support, held_channels=(0,))
    response = jnp.asarray(2 * times[:, None] + 1)
    linear_means, linear_mask = linear.project_response_trajectory(response)
    held_means, held_mask = held.project_response_trajectory(response)

    assert linear.window_plan is support.window_plan
    assert held.window_plan is not support.window_plan
    np.testing.assert_allclose(linear_means[2:, 0], [4.5, 7.0, 11.0])
    np.testing.assert_allclose(held_means[2:, 0], [2.6, 5.0, 9.0])
    np.testing.assert_array_equal(linear_mask[:, 0], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(held_mask, linear_mask)
    assert linear.scan.max_active_windows == held.scan.max_active_windows == 2
    np.testing.assert_array_equal(held.scan.emission_slots[2:, 0], [0, 1, 0])


@pytest.mark.contract
def test_expected_observation_mean_dispatches_by_summary_operator():
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0]),
        manifest_names=["y_sum", "y_count", "y_mean", "y_std", "y_last"],
        support_kinds=["interval", "interval", "interval", "interval", "point"],
        summary_operators=["sum", "count", "mean", "std", "last"],
        observation_windows=["1d", "1d", "1d", "1d", None],
        support_start_times=np.zeros((1, 5)),
        support_end_times=np.zeros((1, 5)),
    )
    operator = compile_observation_operator(support)
    assert operator.summary_operator_codes is not None

    expected = expected_observation_mean(
        response_t=jnp.array([7.0, 8.0, 9.0, 10.0, 11.0]),
        obs_sum=jnp.array([6.0, 4.0, 9.0, 8.0, 0.0]),
        obs_sumsq=jnp.array([36.0, 16.0, 41.0, 34.0, 0.0]),
        obs_weight=jnp.array([1.0, 1.0, 3.0, 2.0, 1.0]),
        summary_operator_codes=operator.summary_operator_codes,
    )

    np.testing.assert_allclose(expected, np.array([6.0, 4.0, 3.0, 1.0, 11.0]))


# =============================================================================
# Exact observation likelihoods
# =============================================================================


@pytest.mark.inference(concern="sampling")
class TestSupportAwareTrajectoryObservationLogProb:
    def test_window_average_matches_manual_gaussian_average(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
        )
        latent = jnp.array([[1.0], [3.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [2.0]], dtype=jnp.float32)
        obs_mask = ~jnp.isnan(observations)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d_meas = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)
        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
        )
        mean_log_prob_fn = mean_density(
            observation_laws([DistributionFamily.GAUSSIAN], parameters=None)[0]
        )

        ll = trajectory_observation_log_probs(
            latent,
            observations,
            obs_mask,
            H,
            d_meas,
            R,
            obs_kernel,
            mean_log_prob_fn,
            support,
        )

        manual = mean_log_prob_fn(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert ll[0] == pytest.approx(0.0)
        assert ll[1] == pytest.approx(float(manual))

    def test_overlapping_window_averages_match_manual_gaussian_means(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([-2.0, -1.0, 0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [-2.0], [-1.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [0.0], [1.0]]),
        )
        latent = jnp.array([[1.0], [3.0], [5.0], [7.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [jnp.nan], [3.0], [5.0]], dtype=jnp.float32)
        obs_mask = ~jnp.isnan(observations)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d_meas = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)
        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
        )
        mean_log_prob_fn = mean_density(
            observation_laws([DistributionFamily.GAUSSIAN], parameters=None)[0]
        )

        ll = trajectory_observation_log_probs(
            latent,
            observations,
            obs_mask,
            H,
            d_meas,
            R,
            obs_kernel,
            mean_log_prob_fn,
            support,
        )

        manual_0 = mean_log_prob_fn(
            jnp.array([3.0], dtype=jnp.float32),
            jnp.array([3.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )
        manual_1 = mean_log_prob_fn(
            jnp.array([5.0], dtype=jnp.float32),
            jnp.array([5.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )

        assert ll[0] == pytest.approx(0.0)
        assert ll[1] == pytest.approx(0.0)
        assert ll[2] == pytest.approx(float(manual_0))
        assert ll[3] == pytest.approx(float(manual_1))


@pytest.mark.inference(concern="sampling")
class TestObservationKernelMissingData:
    """Tests for Gaussian observation masking in shared emission kernels."""

    def test_missing_dimension_not_penalized(self):
        """Missing dims should not incur huge log-det penalties."""
        n_latent, n_manifest = 1, 2
        kernel = observation_kernel([DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None)
        x = jnp.array([0.0])
        y = jnp.array([1.0, -2.0])
        obs_mask = jnp.array([True, False])

        linear_predictor = jnp.ones((n_manifest, n_latent)) @ x + jnp.zeros(n_manifest)
        ll = kernel.log_prob_fn(
            y,
            linear_predictor,
            jnp.diag(jnp.array([0.5, 0.5])),
            obs_mask,
        )

        # Manual univariate logpdf for observed dimension
        sigma2 = 0.5
        resid = y[0]
        manual = -0.5 * (jnp.log(2 * jnp.pi * sigma2) + (resid**2) / sigma2)

        assert jnp.isfinite(ll)
        assert jnp.allclose(ll, manual, atol=1e-5), f"{ll} vs {manual}"


# =============================================================================
# fit() Integration Tests
# =============================================================================


class TestInferenceTracing:
    """Tracing behavior for resolved inference inputs."""

    @pytest.mark.inference(concern="sampling")
    def test_discover_sites_uses_dummy_backend_for_structural_trace(self):
        dynamical_model_spec = one_state_gaussian_model()
        model = compile_fit_fixture(dynamical_model_spec)
        observations = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        class _ExplodingBackend:
            def compute_log_likelihood(self, *_args, **_kwargs):
                raise AssertionError("site discovery should not evaluate the real likelihood")

        site_info = _discover_sites(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, observations, times),
            random.PRNGKey(0),
            _ExplodingBackend(),
        )

        assert "vf_0_p0" in site_info
        assert "manifest_var_diag_free" in site_info


@pytest.mark.contract
class TestDefaultMethodRouting:
    """Regression tests for default inference routing."""

    def test_fit_without_method_dispatches_to_marginal_particle_gibbs(self, monkeypatch):
        dynamical_model_spec = one_state_gaussian_model()
        model = compile_fit_fixture(dynamical_model_spec)
        observations = jnp.zeros((2, 1), dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_marginal_particle_gibbs(_priors, _panel, **kwargs):
            del kwargs
            return particle_posterior(
                JointPosteriorDraws(parameters={"vf_0_p0": jnp.zeros((1,), dtype=jnp.float32)})
            )

        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs"
            ".fit_marginal_particle_gibbs",
            fake_fit_marginal_particle_gibbs,
        )

        result = fit(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, observations, times),
            sampler=SamplerSpec(),
            clock=time.monotonic,
            compute_budget=COMPUTE_BUDGET,
        )
        assert not isinstance(result, ObservationPreflightFailure)

        assert isinstance(result, ParticleMCMCPosterior)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
