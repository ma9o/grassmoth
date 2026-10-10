"""Contract and numerical reference tests for inference.

Structural contracts run by default. Select Laplace initialization with ``warmup``
and particle sampling with ``inference``. Recovery checks live in
``test_parameter_recovery.py``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import coefficient
from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField
from nof1_causal_lab.models.ssm.execution.contracts import (
    LaplaceEvaluationResult,
    LaplaceSolverState,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    compile_observation_operator,
    expected_observation_mean,
    get_summary_operator_codes,
    trajectory_observation_log_probs,
)
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
from nof1_causal_lab.models.ssm.inference.targets.laplace import (
    LaplaceLikelihood,
    _assemble_support_aware_observation_system,
    _block_banded_logdet,
    _build_ieks_system_from_prior,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_banded_cholesky,
    _factor_block_profile_cholesky,
    _ieks_smooth,
    _infer_support_groups,
    _make_support_window_derivatives,
    _predictive_latent_init,
    _should_use_dense_support_laplace,
    _solve_block_banded_from_cholesky,
    block_profile_logdet_packed_cotangent,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    _solve_block_profile_from_cholesky,
    precision_logdet,
)
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites
from nof1_causal_lab.models.ssm.inference.warmup.map import _build_map_laplace_bundle
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.sampler_config import SamplerSpec
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.inference_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    make_observation_support_runtime,
    particle_posterior,
)
from tests.model_fixtures import (
    construct_named,
    one_state_gaussian_model,
    parameter_for,
    without_parameters,
)
from tests.observation_fixtures import mean_density, observation_kernel, observation_laws


def _map_bundle_reuses_runtime_objectives_across_same_shape_datasets__make_aux_kalman_mcmc_smoke_spec() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = one_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    latent_0_diffusion_diag = parameter_for(
        dynamical_model_spec, SiteKind.DIFFUSION_DIAG, "latent_0"
    )
    latent_0_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_0")
    latent_0_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_0")
    latent_0_revised = latent_0.revised(
        coefficients=(
            coefficient(latent_0_diffusion_diag.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(latent_0_t0_var_diag.id, "initial_scale"),
        )
    )
    parameters, distributions = without_parameters(dynamical_model_spec, latent_0_t0_means)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec


def _runtime_dynamics(
    *,
    drift: jnp.ndarray,
    diffusion: jnp.ndarray,
    cint: jnp.ndarray | None = None,
) -> StochasticContinuousTimeStateEvolution:
    params = {"drift": drift}
    if cint is not None:
        params["cint"] = cint
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=int(drift.shape[0]),
            components=(DenseLinear(),),
        ),
        vf_params=(params,),
        diffusion=diffusion,
    )


@pytest.mark.contract
def test_compile_observation_operator_keeps_point_support_without_interval_summary_mode():
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["pulse"],
        support_kinds=["point"],
        observation_windows=[None],
        support_start_times=np.array([[0.0], [1.0]]),
        support_end_times=np.array([[0.0], [1.0]]),
        interval_prev_coeffs=np.zeros((2, 1)),
        interval_curr_coeffs=np.zeros((2, 1)),
        interval_weights=np.zeros((2, 1)),
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
        interval_prev_coeffs=np.zeros((1, 5)),
        interval_curr_coeffs=np.zeros((1, 5)),
        interval_weights=np.ones((1, 5)),
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
# Marginal likelihood backend functionality
# =============================================================================


@pytest.mark.inference(concern="warmup")
class TestLaplaceEMBlockSolver:
    """Numerical checks for the block-tridiagonal IEKS rewrite."""

    @staticmethod
    def _dense_block_matrix(
        lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray
    ) -> jnp.ndarray:
        n_blocks, block_dim = diag.shape[:2]
        mat = np.zeros((n_blocks * block_dim, n_blocks * block_dim), dtype=np.float64)
        lower_np = np.asarray(lower)
        diag_np = np.asarray(diag)
        upper_np = np.asarray(upper)
        for i in range(n_blocks):
            row = slice(i * block_dim, (i + 1) * block_dim)
            mat[row, row] = diag_np[i]
            if i > 0:
                prev = slice((i - 1) * block_dim, i * block_dim)
                mat[row, prev] = lower_np[i]
            if i + 1 < n_blocks:
                nxt = slice((i + 1) * block_dim, (i + 2) * block_dim)
                mat[row, nxt] = upper_np[i]
        return jnp.asarray(mat)

    def test_gaussian_ieks_mode_matches_dense_system(self):
        """For Gaussian observations, one IEKS step should equal the exact mode solve."""
        key = random.PRNGKey(11)
        T = 5
        D = 2
        M = 2

        observations = random.normal(key, (T, M)) * 0.2
        obs_mask = jnp.ones((T, M), dtype=bool)
        Ad = jnp.broadcast_to(jnp.array([[0.92, 0.05], [0.02, 0.88]]), (T, D, D))
        Qd = jnp.broadcast_to(jnp.array([[0.15, 0.01], [0.01, 0.12]]), (T, D, D))
        cd = jnp.zeros((T, D))
        H = jnp.array([[1.0, 0.1], [0.2, 1.0]])
        d = jnp.array([0.0, 0.1])
        R = jnp.array([[0.2, 0.02], [0.02, 0.25]])
        init_mean = jnp.array([0.05, -0.1])
        init_cov = jnp.array([[0.8, 0.05], [0.05, 0.7]])

        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN] * M, [LinkFunction.IDENTITY] * M, None
        )

        H_rows = jnp.broadcast_to(H[None, :, :], (T, *H.shape))
        d_rows = jnp.broadcast_to(d[None, :], (T, *d.shape))
        z_smooth, log_lik, _inner_eval_aux = _ieks_smooth(
            observations,
            obs_mask,
            Ad,
            Qd,
            cd,
            H_rows,
            d_rows,
            R,
            init_mean,
            init_cov,
            obs_kernel,
            n_ieks_iters=1,
        )

        z_init = jnp.broadcast_to(init_mean, (T, D))
        grads, J_t = jax.vmap(
            lambda y_t, z_t, mask_t: obs_kernel.latent_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
        )(observations, z_init, obs_mask.astype(jnp.float32))
        tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_init, grads)
        prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )
        lower, diag, upper, rhs = _build_ieks_system_from_prior(
            prior_lower,
            prior_diag,
            prior_upper,
            prior_rhs,
            J_t,
            tilde_y,
        )
        dense = self._dense_block_matrix(lower, diag, upper)
        z_dense = jnp.linalg.solve(dense, rhs.reshape(-1)).reshape(T, D)

        np.testing.assert_allclose(z_smooth, z_dense, atol=1e-4, rtol=1e-4)
        assert jnp.isfinite(log_lik).all()


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
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
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
            interval_prev_coeffs=np.array(
                [
                    [[0.0, 0.0]],
                    [[0.5, 0.0]],
                    [[0.5, 0.5]],
                    [[0.0, 0.5]],
                ]
            ),
            interval_curr_coeffs=np.array(
                [
                    [[0.0, 0.0]],
                    [[0.5, 0.0]],
                    [[0.5, 0.5]],
                    [[0.0, 0.5]],
                ]
            ),
            interval_weights=np.array(
                [
                    [[0.0, 0.0]],
                    [[1.0, 0.0]],
                    [[1.0, 1.0]],
                    [[0.0, 1.0]],
                ]
            ),
            emission_slot_indices=np.array([[-1], [-1], [0], [1]], dtype=np.int64),
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


class TestLaplaceSupportAware:
    @pytest.mark.contract
    def test_infer_support_groups_ignores_reused_slot_history(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0], [np.nan], [2.0], [np.nan], [4.0]]),
            support_end_times=np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan], [5.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [-1], [0], [-1], [0]], dtype=np.int64),
        )

        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)
        assert len(window_batches) == 1
        windows = window_batches[0]

        np.testing.assert_array_equal(np.asarray(windows.anchor_indices), np.array([1, 3, 5]))
        np.testing.assert_array_equal(np.asarray(windows.start_indices), np.array([0, 2, 4]))
        np.testing.assert_array_equal(np.asarray(windows.state_lens), np.array([2, 2, 2]))
        np.testing.assert_array_equal(
            np.asarray(row_upper_bandwidths), np.array([1, 0, 1, 0, 1, 0])
        )
        assert windows.max_state_len == 2
        assert bandwidth == 1

    @pytest.mark.contract
    def test_infer_support_groups_buckets_by_state_length(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["3d"],
            support_start_times=np.array([[np.nan], [0.0], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0], [np.nan], [3.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [0.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [-1], [0]], dtype=np.int64),
        )

        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)

        assert [batch.max_state_len for batch in window_batches] == [2, 4]
        np.testing.assert_array_equal(np.asarray(window_batches[0].anchor_indices), np.array([1]))
        np.testing.assert_array_equal(np.asarray(window_batches[1].anchor_indices), np.array([3]))
        np.testing.assert_array_equal(np.asarray(row_upper_bandwidths), np.array([3, 2, 1, 0]))
        assert bandwidth == 3

    @pytest.mark.inference(concern="warmup")
    def test_profile_masked_banded_cholesky_matches_full_banded_solver(self):
        row_upper_bandwidths = jnp.array([3, 2, 1, 1, 0], dtype=jnp.int32)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        diag = jnp.asarray(np.array([4.0, 4.5, 4.2, 4.3, 3.8], dtype=np.float32)[:, None, None])
        upper = jnp.zeros((3, 5, 1, 1), dtype=jnp.float32)
        upper = upper.at[0, 0, 0, 0].set(0.20)
        upper = upper.at[0, 1, 0, 0].set(0.15)
        upper = upper.at[0, 2, 0, 0].set(0.10)
        upper = upper.at[0, 3, 0, 0].set(0.08)
        upper = upper.at[1, 0, 0, 0].set(0.05)
        upper = upper.at[1, 1, 0, 0].set(0.04)
        upper = upper.at[2, 0, 0, 0].set(0.02)
        rhs = jnp.asarray(np.array([1.0, -0.5, 0.25, 0.75, -1.25], dtype=np.float32)[:, None])

        chol_full, lower_full = _factor_block_banded_cholesky(diag, upper)
        sol_full = _solve_block_banded_from_cholesky(chol_full, lower_full, rhs)

        chol_profile, lower_profile = _factor_block_banded_cholesky(
            diag,
            upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        sol_profile = _solve_block_banded_from_cholesky(
            chol_profile,
            lower_profile,
            rhs,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

        np.testing.assert_allclose(
            np.asarray(chol_profile), np.asarray(chol_full), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(lower_profile),
            np.asarray(lower_full),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(sol_profile), np.asarray(sol_full), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(_block_banded_logdet(chol_profile)),
            np.asarray(_block_banded_logdet(chol_full)),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.inference(concern="warmup")
    def test_predictive_latent_init_rolls_forward_mean_dynamics(self):
        Ad = jnp.array(
            [
                [[1.0]],
                [[2.0]],
                [[3.0]],
            ],
            dtype=jnp.float32,
        )
        cd = jnp.array([[0.5], [1.0], [-2.0]], dtype=jnp.float32)
        init_mean = jnp.array([2.0], dtype=jnp.float32)

        z_init = _predictive_latent_init(Ad, cd, init_mean)

        np.testing.assert_allclose(
            np.asarray(z_init[:, 0]),
            np.array([2.5, 6.0, 16.0], dtype=np.float32),
        )

    @pytest.mark.inference(concern="warmup")
    def test_support_window_gauss_newton_matches_linear_gaussian_exact_blocks(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )
        window_batches, bandwidth, _row_upper_bandwidths = _infer_support_groups(support)
        assert len(window_batches) == 1
        windows = window_batches[0]
        observation_operator = compile_observation_operator(support)
        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
        )
        mean_log_prob_fn = mean_density(
            observation_laws([DistributionFamily.GAUSSIAN], parameters=None)[0]
        )
        window_derivatives = (
            _make_support_window_derivatives(
                max_state_len=windows.max_state_len,
                n_latent=1,
                n_manifest=1,
                summary_operator_codes=get_summary_operator_codes(support),
                obs_kernel=obs_kernel,
                mean_log_prob_fn=mean_log_prob_fn,
            ),
        )

        z_est = jnp.array([[0.4], [1.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [1.3]], dtype=jnp.float32)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)

        @jax.jit
        def _assemble(states, obs):
            return _assemble_support_aware_observation_system(
                states,
                obs,
                ~jnp.isnan(obs),
                H,
                d,
                R,
                obs_kernel,
                window_batches,
                observation_operator.point_like_mask(states.dtype),
                window_derivatives,
                bandwidth,
            )

        diag, upper, rhs = _assemble(z_est, observations)

        # y ~ Normal((z0 + z1)/2, variance=0.2): information is w wᵀ / R,
        # and the Gaussian information vector is w y / R, independent of z_est.
        # Compute this from the declared observation, without calling emission
        # or support code again to construct the reference.
        weights = np.array([0.5, 0.5])
        information = np.outer(weights, weights) / 0.2
        expected_rhs = weights * 1.3 / 0.2

        np.testing.assert_allclose(diag[:, 0, 0], np.diag(information), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            upper[0, 0, 0, 0],
            information[0, 1],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(rhs[:, 0], expected_rhs, rtol=1e-5, atol=1e-5)


class TestLaplaceSolverState:
    """Explicit solver initialization and stateless support-window derivatives."""

    @pytest.mark.inference(concern="warmup")
    @pytest.mark.parametrize("profile", [(2, 2, 1, 0), (1, 0, 1, 0)])
    def test_block_profile_logdet_cotangent_matches_direct_autodiff(self, profile):
        row_upper_bandwidths = jnp.array(profile, dtype=jnp.int32)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        diag = jnp.array(
            [
                [[4.0, 0.2], [0.2, 3.5]],
                [[3.8, -0.1], [-0.1, 3.2]],
                [[3.4, 0.05], [0.05, 2.9]],
                [[3.1, 0.0], [0.0, 2.7]],
            ],
            dtype=jnp.float32,
        )
        upper = jnp.zeros((2, 4, 2, 2), dtype=jnp.float32)
        upper = upper.at[0, 0].set(jnp.array([[0.12, -0.03], [0.05, 0.08]], dtype=jnp.float32))
        upper = upper.at[1, 0].set(jnp.array([[0.04, 0.01], [-0.02, 0.03]], dtype=jnp.float32))
        upper = upper.at[0, 1].set(jnp.array([[0.09, 0.02], [0.01, 0.07]], dtype=jnp.float32))
        upper = upper.at[0, 2].set(jnp.array([[0.06, -0.01], [0.02, 0.05]], dtype=jnp.float32))
        upper = upper[: max(profile)]

        def _packed_logdet(diag_blocks, upper_blocks):
            chol_diag, _lower = _factor_block_banded_cholesky(
                diag_blocks,
                upper_blocks,
                row_upper_bandwidths,
                row_lower_bandwidths,
            )
            return _block_banded_logdet(chol_diag)

        direct_diag_bar, direct_upper_bar = jax.grad(_packed_logdet, argnums=(0, 1))(diag, upper)
        value, (compiled_diag_bar, compiled_upper_bar) = jax.jit(
            jax.value_and_grad(
                lambda diagonal, off_diagonal: precision_logdet(
                    diagonal, off_diagonal, row_upper_bandwidths, row_lower_bandwidths
                ),
                argnums=(0, 1),
            )
        )(diag, upper)
        np.testing.assert_allclose(value, _packed_logdet(diag, upper), rtol=1e-5)
        np.testing.assert_allclose(compiled_diag_bar, direct_diag_bar, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(compiled_upper_bar, direct_upper_bar, rtol=1e-4, atol=1e-4)
        chol_diag, lower = _factor_block_profile_cholesky(
            diag,
            upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        reference_chol, reference_lower = _factor_block_banded_cholesky(
            diag, upper, row_upper_bandwidths, row_lower_bandwidths
        )
        rhs = jnp.arange(8, dtype=diag.dtype).reshape(4, 2) / 10
        np.testing.assert_allclose(
            _solve_block_profile_from_cholesky(
                chol_diag, lower, rhs, row_upper_bandwidths, row_lower_bandwidths
            ),
            _solve_block_banded_from_cholesky(
                reference_chol, reference_lower, rhs, row_upper_bandwidths, row_lower_bandwidths
            ),
            rtol=1e-5,
            atol=1e-6,
        )
        diag_bar, upper_bar = block_profile_logdet_packed_cotangent(
            chol_diag,
            lower,
            row_upper_bandwidths,
            row_lower_bandwidths,
            scale=jnp.array(1.0, dtype=diag.dtype),
        )

        np.testing.assert_allclose(
            np.asarray(diag_bar), np.asarray(direct_diag_bar), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(upper_bar),
            np.asarray(direct_upper_bar),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.contract
    def test_laplace_backend_threads_only_explicit_solver_state(self, monkeypatch):
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            n_ieks_iters=2,
        )
        ct_params = _runtime_dynamics(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion=jnp.linalg.cholesky(jnp.array([[0.1]], dtype=jnp.float32)),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = MultivariateNormal(
            loc=jnp.array([0.0], dtype=jnp.float32),
            covariance_matrix=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[0.0], [0.25], [0.5]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        seen_inits: list[np.ndarray | None] = []
        returned_mode = jnp.array([[0.1], [0.2], [0.3]], dtype=jnp.float32)

        def _fake_ieks(*_args, z_init=None, **_kwargs):
            seen_inits.append(None if z_init is None else np.asarray(z_init))
            return returned_mode, jnp.array(-1.0, dtype=jnp.float32), {"latent_mode": returned_mode}

        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.inference.targets.laplace._ieks_smooth",
            _fake_ieks,
        )
        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.inference.targets.laplace.build_discrete_transitions",
            lambda *_args, **_kwargs: SimpleNamespace(
                A=jnp.ones((3, 1, 1)),
                cov=jnp.ones((3, 1, 1)),
                bias=jnp.zeros((3, 1)),
            ),
        )

        first = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
        )
        second = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            solver_state=first.state,
            observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
        )

        assert float(first.log_likelihood) == pytest.approx(-1.0)
        assert float(second.log_likelihood) == pytest.approx(-1.0)
        assert seen_inits[0] is None
        warm_start = seen_inits[1]
        assert warm_start is not None
        np.testing.assert_allclose(warm_start, np.asarray(returned_mode))
        backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
        )
        assert seen_inits[2] is None
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            backend.n_ieks_iters = 99  # ty: ignore[invalid-assignment] -- Exercise runtime rejection of a frozen field.

    @pytest.mark.contract
    def test_support_derivative_builders_use_current_measurement_objects(self, monkeypatch):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            summary_operators=["std"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            n_ieks_iters=2,
            observation_support=support,
        )
        compiled = compile_observation_model(
            observation_laws([DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
            observation_support=support,
        )
        built = []

        def _build(**_kwargs):
            result = object()
            built.append(result)
            return result

        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.inference.targets.laplace._make_support_window_derivatives",
            _build,
        )
        first = backend._build_support_window_derivatives(compiled)
        assert backend._build_support_window_derivatives(compiled) != first
        assert len(built) == 2

        changed = compile_observation_model(
            observation_laws([DistributionFamily.POISSON], [LinkFunction.LOG], None),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
            observation_support=support,
        )
        assert backend._build_support_window_derivatives(changed) != first
        assert len(built) == 3


@pytest.mark.contract
def test_support_solver_routes_small_and_large_latent_paths():
    assert _should_use_dense_support_laplace(n_time=3, n_latent=1)
    assert not _should_use_dense_support_laplace(n_time=18, n_latent=10)


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


@pytest.mark.contract
def test_map_objectives_receive_each_bound_dataset(monkeypatch):
    observations_a = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    observations_b = jnp.array([[2.0], [3.0]], dtype=jnp.float32)
    times_a = jnp.array([0.0, 1.0], dtype=jnp.float32)
    times_b = jnp.array([0.0, 2.0], dtype=jnp.float32)
    counters = {"discover": 0, "build_eval_fns": 0}

    def fake_prepare_parameters(_priors, _panel, _trace_key, reparam):
        del reparam
        counters["discover"] += 1
        values = jnp.array([0.5, -0.25], dtype=jnp.float32)
        parameters = ParameterTransform(
            values, lambda z: {"theta": z}, lambda z: {"theta": z}, lambda z: -jnp.sum(z**2)
        )
        return parameters, {"theta": {"value": values}}, {"theta"}

    def fake_build_eval_fns(
        _panel,
        _parameters,
        likelihood_backend,
    ):
        del likelihood_backend
        counters["build_eval_fns"] += 1

        def log_lik_fn(z, runtime_observations, runtime_times, solver_state=None):
            del solver_state
            return jnp.sum(z) + jnp.sum(runtime_observations) + jnp.sum(runtime_times)

        def log_prior_unc_fn(z):
            return -0.5 * jnp.sum(z**2)

        def log_lik_with_aux_fn(z, runtime_observations, runtime_times, solver_state=None):
            del solver_state
            ll = log_lik_fn(z, runtime_observations, runtime_times)
            return ll, LaplaceEvaluationResult(
                ll,
                LaplaceSolverState(),
                {
                    "solver_kind": jnp.asarray(0, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(0, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(0, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(0.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(0.0, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(0.0, dtype=jnp.float32),
                    "final_damping": jnp.asarray(0.0, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(0.0, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.0, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(0.0, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.0, dtype=jnp.float32),
                },
            )

        return log_lik_fn, log_prior_unc_fn, log_lik_with_aux_fn

    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.inference.warmup.map.prepare_model_parameters",
        fake_prepare_parameters,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.inference.warmup.map._build_eval_fns",
        fake_build_eval_fns,
    )

    model = compile_fit_fixture(
        _map_bundle_reuses_runtime_objectives_across_same_shape_datasets__make_aux_kalman_mcmc_smoke_spec()
    )
    backend = Mock(
        spec=LaplaceLikelihood,
    )
    bundle_a = _build_map_laplace_bundle(
        model.prior_runtime_bundle,
        bind_panel_fixture(model.compiled_dynamical_model, observations_a, times_a),
        random.PRNGKey(0),
        backend,
        None,
    )
    bundle_b = _build_map_laplace_bundle(
        model.prior_runtime_bundle,
        bind_panel_fixture(model.compiled_dynamical_model, observations_b, times_b),
        random.PRNGKey(1),
        backend,
        None,
    )

    assert counters["discover"] == 2
    assert counters["build_eval_fns"] == 2

    z = jnp.array([0.2, -0.1], dtype=jnp.float32)
    log_post_a = bundle_a["log_posterior_fn"](z, observations_a, times_a)
    log_post_b = bundle_b["log_posterior_fn"](z, observations_b, times_b)
    assert float(log_post_a) == pytest.approx(2.075)
    assert float(log_post_b) == pytest.approx(7.075)


# =============================================================================
# Builder Noise Family Wiring Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if TYPE_CHECKING:
    from dynestyx import StochasticContinuousTimeStateEvolution
