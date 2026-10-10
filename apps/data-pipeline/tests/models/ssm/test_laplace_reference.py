"""Independent likelihood and gradient references for Laplace initialization."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx import StochasticContinuousTimeStateEvolution
from numpyro.distributions import MultivariateNormal
from scipy.optimize import brentq
from scipy.stats import norm, t

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.inference.targets.laplace import (
    LaplaceLikelihood,
    _dense_support_laplace_log_lik,
)
from tests.inference_fixtures import make_observation_support_runtime
from tests.observation_fixtures import mean_density, observation_kernel, observation_laws

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("rank", [0, 3])
def test_interval_precision_solve_logdet_and_gradient_match_dense(rank):
    from nof1_causal_lab.models.ssm.inference.targets.laplace.interval_precision import (
        IntervalPrecision,
        factor_interval_precision,
        solve_interval_precision,
    )

    with jax.enable_x64():
        diagonal = jnp.array([[[4.0, 0.2], [0.2, 3.0]]] * 3)
        upper = jnp.array([[[[0.1, 0.03], [-0.04, 0.2]]] * 2 + [[[0.0, 0.0], [0.0, 0.0]]]])
        jacobian = jnp.arange(18, dtype=jnp.float64).reshape(3, 2, 3)[..., :rank] / 30
        information = jnp.diag(jnp.array([0.6, -0.1, 0.0])[:rank])
        rhs = jnp.arange(6, dtype=jnp.float64).reshape(3, 2) / 5

        def objective(scale, dense):
            scaled_jacobian = scale * jacobian
            if dense:
                matrix = jnp.zeros((6, 6))
                for i in range(3):
                    matrix = matrix.at[2 * i : 2 * i + 2, 2 * i : 2 * i + 2].set(diagonal[i])
                for i in range(2):
                    matrix = matrix.at[2 * i : 2 * i + 2, 2 * i + 2 : 2 * i + 4].set(upper[0, i])
                    matrix = matrix.at[2 * i + 2 : 2 * i + 4, 2 * i : 2 * i + 2].set(upper[0, i].T)
                columns = scaled_jacobian.reshape(6, rank)
                matrix = matrix + columns @ information @ columns.T + 1e-6 * jnp.eye(6)
                solution = jnp.linalg.solve(matrix, rhs.reshape(-1)).reshape(3, 2)
                logdet = jnp.linalg.slogdet(matrix)[1]
            else:
                factor = factor_interval_precision(
                    IntervalPrecision(diagonal, upper, scaled_jacobian, information)
                )
                solution = solve_interval_precision(factor, rhs)
                logdet = factor.logdet
            return logdet + jnp.sum(solution**2), (solution, logdet)

        actual = jax.jit(jax.value_and_grad(lambda scale: objective(scale, False), has_aux=True))(
            0.8
        )
        expected = jax.value_and_grad(lambda scale: objective(scale, True), has_aux=True)(0.8)
        for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, e, rtol=1e-9, atol=1e-9)


def test_interval_precision_rejects_indefinite_matrix_with_positive_determinant():
    from nof1_causal_lab.models.ssm.inference.targets.laplace.interval_precision import (
        IntervalPrecision,
        factor_interval_precision,
    )

    factor = factor_interval_precision(
        IntervalPrecision(
            jnp.eye(2)[None], jnp.zeros((1, 1, 2, 2)), jnp.eye(2)[None], -2 * jnp.eye(2)
        )
    )
    assert jnp.isnan(factor.logdet)


@pytest.mark.parametrize("observed", [True, False])
def test_interval_curvature_ignores_missing_readings(observed):
    from nof1_causal_lab.models.ssm.inference.targets.laplace.support import (
        _assemble_support_aware_observation_system,
        _make_support_window_derivatives,
    )

    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["observed_signal", "missing_signal"],
        support_kinds=["interval", "interval"],
        observation_windows=["1d", "1d"],
        support_start_times=np.array([[np.nan, np.nan], [0.0, 0.0]]),
        support_end_times=np.array([[np.nan, np.nan], [1.0, 1.0]]),
    )
    backend = LaplaceLikelihood(
        n_latent=1,
        n_manifest=2,
        observation_support=support,
        observed_mask=np.ones((2, 2), dtype=bool),
    )
    kernel = observation_kernel([DistributionFamily.GAUSSIAN] * 2)
    derivatives = tuple(
        _make_support_window_derivatives(
            max_state_len=batch.max_state_len,
            n_latent=1,
            n_manifest=2,
            summary_operator_codes=backend._summary_operator_codes,
            obs_kernel=kernel,
            mean_log_prob_fn=mean_density(observation_laws([DistributionFamily.GAUSSIAN])[0]),
        )
        for batch in backend._support_window_batches
    )
    observations = jnp.array([[jnp.nan, jnp.nan], [1.0 if observed else jnp.nan, jnp.nan]])
    diag, jacobian, information, rhs = _assemble_support_aware_observation_system(
        jnp.array([[0.2], [0.4]]),
        observations,
        jnp.isfinite(observations),
        jnp.array([[1.0], [3.0]]),
        jnp.zeros(2),
        jnp.array([[0.25, 0.125], [0.125, 1.0]]),
        kernel,
        backend._support_window_batches,
        jnp.zeros(2),
        derivatives,
    )
    # A present interval mean has design (1/2, 1/2) and marginal variance 1/4.
    precision = float(observed)
    dense = np.diag(diag[:, 0, 0]) + jacobian[:, 0] @ information @ jacobian[:, 0].T
    np.testing.assert_allclose(dense, np.full((2, 2), precision), atol=1e-6)
    np.testing.assert_allclose(rhs[:, 0], 2.0 * precision, atol=1e-6)


def test_mixed_precision_point_and_interval_window_derivatives_match_gaussian_reference():
    from nof1_causal_lab.models.ssm.inference.targets.laplace.support import (
        _make_support_window_derivatives,
    )

    with jax.enable_x64():
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["mean_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
        )
        from nof1_causal_lab.models.ssm.execution.observation_operator import (
            get_summary_operator_codes,
        )

        derivatives = _make_support_window_derivatives(
            max_state_len=2,
            n_latent=1,
            n_manifest=1,
            summary_operator_codes=get_summary_operator_codes(support),
            obs_kernel=observation_kernel(
                [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
            ),
            mean_log_prob_fn=mean_density(observation_laws([DistributionFamily.GAUSSIAN])[0]),
        )
        gradient, jacobian, information = jax.jit(derivatives)(
            jnp.array([[0.2, 0.0], [0.4, 0.8]], dtype=jnp.float32),
            jnp.array([1, 2]),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.full((2, 1, 1), 0.5, dtype=jnp.float32),
            jnp.full((2, 1, 1), 0.5, dtype=jnp.float32),
            jnp.ones((2, 1, 1), dtype=jnp.float32),
            jnp.array([[0.3], [0.9]], dtype=jnp.float32),
            jnp.array([[1.5]], dtype=jnp.float64),
            jnp.array([-0.2], dtype=jnp.float64),
            jnp.array([[0.25]], dtype=jnp.float64),
        )
        expected_jacobian = np.array([[1.5, 0.0], [0.75, 0.75]])
        expected_gradient = (np.array([0.3, 0.9]) - np.array([0.1, 0.7]))[:, None]
        expected_gradient = expected_gradient / 0.25 * expected_jacobian
        np.testing.assert_allclose(jacobian[:, 0, :, 0], expected_jacobian, atol=1e-6)
        np.testing.assert_allclose(gradient[:, :, 0], expected_gradient, atol=1e-6)
        np.testing.assert_allclose(information, 4.0, atol=1e-6)


def test_structured_newton_preserves_coordinate_precision_with_double_curvature():
    from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import _prepare_linearized_path
    from nof1_causal_lab.models.ssm.inference.targets.laplace.solvers import (
        solve_fixed_point_mode,
        solve_latent_mode,
    )

    with jax.enable_x64():
        mean = jnp.array([[0.2], [-0.4]], dtype=jnp.float64)
        precision = jnp.array([[2.0], [3.0]], dtype=jnp.float64)

        def log_joint(path, center):
            return -0.5 * jnp.sum(precision * (path - center) ** 2)

        def system(_path, center):
            return (
                precision[:, :, None],
                jnp.zeros((1, 2, 1, 1), dtype=jnp.float64),
                precision * center,
            )

        def solve(center):
            mode, _ = solve_latent_mode(
                log_joint,
                system,
                jnp.zeros((2, 1), dtype=jnp.float32),
                center,
                bandwidth=1,
                row_upper=jnp.array([1, 1]),
                row_lower=jnp.array([0, 0]),
                max_steps=4,
            )
            return mode

        mode = solve(mean)
        assert mode.dtype == jnp.float32
        np.testing.assert_allclose(mode, mean, atol=1e-5)
        np.testing.assert_allclose(
            jax.grad(lambda center: jnp.sum(solve(center)))(mean), 1, atol=1e-5
        )
        fixed, _ = solve_fixed_point_mode(
            lambda path, _args: (path + mode) / 2,
            jnp.zeros_like(mode),
            max_steps=30,
        )
        assert fixed.dtype == jnp.float32
        np.testing.assert_allclose(fixed, mode, atol=2e-3)
        _, initial = _prepare_linearized_path(
            _runtime_dynamics(dynamics=jnp.array([[-0.1]]), diffusion=jnp.array([[0.2]])),
            jnp.ones(2, dtype=jnp.float64),
            jnp.zeros(1, dtype=jnp.float32),
            z_init=None,
            dtype=jnp.dtype(jnp.float32),
        )
        assert initial.dtype == jnp.float32


@pytest.mark.parametrize("solver", ["point", "interval", "dense", "interval_blocks"])
def test_student_t_laplace_value_and_gradient_match_scalar_reference(monkeypatch, solver):
    """A single observation of a Gaussian path reduces to a scalar Laplace integral.

    The reference integrates all orthogonal latent directions analytically and
    solves the remaining scalar mode with SciPy. This checks all three
    Optimistix paths against an independent reference.
    """
    from nof1_causal_lab.models.ssm.inference.targets import laplace

    monkeypatch.setattr(laplace, "_should_use_dense_support_laplace", lambda **_kwargs: False)
    n_time, n_latent = (4, 2) if solver == "interval_blocks" else (2, 1)
    state_mean = np.array([0.05, -0.03])[:n_latent]
    state_variance = np.array([0.6, 0.45])[:n_latent]
    process_variance = np.array([0.07, 0.04])[:n_latent]
    loadings = np.array([1.0, 0.4])[:n_latent]
    anchor_times = np.arange(n_time, dtype=float)
    observations = jnp.full((n_time, 1), jnp.nan).at[-1, 0].set(0.25)
    support = None
    weights = np.array([0.0, 1.0])
    if solver != "point":
        weights = np.ones(n_time) / (n_time - 1)
        weights[[0, -1]] /= 2
        start_times, end_times = np.full((2, n_time, 1), np.nan)
        start_times[-1, 0], end_times[-1, 0] = 0.0, n_time - 1
        support = make_observation_support_runtime(
            anchor_times=anchor_times,
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=[f"{n_time - 1}d"],
            support_start_times=start_times,
            support_end_times=end_times,
        )
    backend = LaplaceLikelihood(
        n_latent=n_latent,
        n_manifest=1,
        observed_mask=np.ones((n_time, 1), dtype=bool),
        n_ieks_iters=4,
        observation_support=support,
    )
    dynamics = _runtime_dynamics(
        dynamics=-0.09 * jnp.eye(n_latent, dtype=jnp.float32),
        diffusion=jnp.diag(jnp.sqrt(jnp.asarray(process_variance, dtype=jnp.float32))),
    )
    initial = MultivariateNormal(
        loc=jnp.asarray(state_mean, dtype=jnp.float32),
        covariance_matrix=jnp.diag(jnp.asarray(state_variance, dtype=jnp.float32)),
    )

    def _objective(raw):
        measurement = MeasurementParams(
            lambda_mat=jnp.asarray(loadings[None], dtype=jnp.float32),
            manifest_means=jnp.zeros(1, dtype=jnp.float32),
            manifest_cov=(jnp.exp(raw[1]) + 0.1).reshape(1, 1),
        )
        if solver == "dense":
            parameters = {"obs_df": jnp.exp(raw[0]) + 2.5}
            law = observation_laws([DistributionFamily.STUDENT_T], parameters=parameters)[0]
            value, _aux = _dense_support_laplace_log_lik(
                jnp.array([[0.0], [0.25]], dtype=jnp.float32),
                jnp.array([[False], [True]]),
                jnp.full((2, 1, 1), np.exp(-0.09), dtype=jnp.float32),
                jnp.full((2, 1, 1), -0.07 * np.expm1(-0.18) / 0.18, dtype=jnp.float32),
                jnp.zeros((2, 1), dtype=jnp.float32),
                measurement.lambda_mat,
                measurement.manifest_means,
                measurement.manifest_cov,
                initial.mean,
                initial.covariance_matrix,
                observation_kernel(
                    [DistributionFamily.STUDENT_T], [LinkFunction.IDENTITY], parameters
                ),
                mean_density(law),
                support,
                8,
            )
            return value
        return backend.compute_log_likelihood(
            dynamics,
            measurement,
            initial,
            observations,
            jnp.ones(n_time, dtype=jnp.float32),
            observation_laws=observation_laws(
                [DistributionFamily.STUDENT_T], parameters={"obs_df": jnp.exp(raw[0]) + 2.5}
            ),
        )

    anchors = anchor_times + 1
    stationary_var = np.dot(loadings**2, process_variance) / (2 * 0.09)
    initial_variance = np.dot(loadings**2, state_variance)
    covariance = stationary_var * np.exp(-0.09 * np.abs(anchors[:, None] - anchors[None, :])) + (
        initial_variance - stationary_var
    ) * np.exp(-0.09 * (anchors[:, None] + anchors[None, :]))
    prior_mean = weights @ (np.dot(loadings, state_mean) * np.exp(-0.09 * anchors))
    prior_var = weights @ covariance @ weights

    def _reference(raw):
        df, variance = np.exp(raw) + np.array([2.5, 0.1])
        observed = 0.25
        mode = brentq(
            lambda mean: (
                (mean - prior_mean) / prior_var
                - (df + 1) * (observed - mean) / (df * variance + (observed - mean) ** 2)
            ),
            prior_mean,
            observed,
        )
        residual_sq = (observed - mode) ** 2
        curvature = (
            1 / prior_var
            + (df + 1) * (df * variance - residual_sq) / (df * variance + residual_sq) ** 2
        )
        return (
            norm.logpdf(mode, loc=prior_mean, scale=np.sqrt(prior_var))
            + t.logpdf(observed, df, loc=mode, scale=np.sqrt(variance))
            + 0.5 * np.log(2 * np.pi / curvature)
        )

    raw = np.array([0.35, -1.2])
    value, gradient = jax.jit(jax.value_and_grad(_objective))(jnp.asarray(raw, dtype=jnp.float32))
    eps = 1e-4
    reference_gradient = np.array(
        [(_reference(raw + step) - _reference(raw - step)) / (2 * eps) for step in eps * np.eye(2)]
    )
    np.testing.assert_allclose(value, _reference(raw), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=2e-3, atol=2e-4)


def _runtime_dynamics(
    *,
    dynamics: jnp.ndarray,
    diffusion: jnp.ndarray,
    cint: jnp.ndarray | None = None,
) -> StochasticContinuousTimeStateEvolution:
    params = {"drift": dynamics}
    if cint is not None:
        params["cint"] = cint
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=int(dynamics.shape[0]),
            components=(DenseLinear(),),
        ),
        vf_params=(params,),
        diffusion=diffusion,
    )


class TestLaplaceSupportAware:
    def test_laplace_interval_matches_dense_reference(self, monkeypatch):
        from nof1_causal_lab.models.ssm.inference.targets import laplace

        # Force the numerical branch on a small path. Float64 support metadata
        # with float32 states also covers the former oversized window smoke.
        interval_solver = Mock(wraps=laplace._support_aware_ieks_laplace)
        monkeypatch.setattr(laplace, "_should_use_dense_support_laplace", lambda **_kwargs: False)
        monkeypatch.setattr(laplace, "_support_aware_ieks_laplace", interval_solver)
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            n_ieks_iters=1,
            observation_support=support,
            observed_mask=np.ones((3, 1), dtype=bool),
        )
        ct_params = _runtime_dynamics(
            dynamics=jnp.array([[-0.4]], dtype=jnp.float32),
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
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        # Scalar OU transitions provide an independent input to the dense solver.
        Ad = jnp.full((3, 1, 1), np.exp(-0.4), dtype=jnp.float32)
        Qd = jnp.full((3, 1, 1), -0.1 * np.expm1(-0.8) / 0.8, dtype=jnp.float32)
        cd = jnp.zeros((3, 1), dtype=jnp.float32)
        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
        )
        mean_log_prob_fn = mean_density(
            observation_laws([DistributionFamily.GAUSSIAN], parameters=None)[0]
        )

        @jax.jit
        def _banded(obs):
            return backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                obs,
                time_intervals,
                observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
            )

        @jax.jit
        def _dense(obs):
            value, _aux = _dense_support_laplace_log_lik(
                jnp.nan_to_num(obs, nan=0.0),
                ~jnp.isnan(obs),
                Ad,
                Qd,
                cd,
                meas_params.lambda_mat,
                meas_params.manifest_means,
                meas_params.manifest_cov,
                init.mean,
                init.covariance_matrix,
                obs_kernel,
                mean_log_prob_fn,
                support,
                1,
            )
            return value

        banded = _banded(observations)
        dense = _dense(observations)

        interval_solver.assert_called_once()
        # Independent OU covariance for the three latent anchors after the
        # initial prediction. The interval mean has weights (1/4, 1/2, 1/4).
        anchors = np.arange(1.0, 4.0)
        stationary_var = 0.1 / (2 * 0.4)
        covariance = stationary_var * np.exp(-0.4 * np.abs(anchors[:, None] - anchors[None, :])) + (
            1.0 - stationary_var
        ) * np.exp(-0.4 * (anchors[:, None] + anchors[None, :]))
        weights = np.array([0.25, 0.5, 0.25])
        variance = weights @ covariance @ weights + 0.2
        exact = -0.5 * (np.log(2 * np.pi * variance) + 0.25**2 / variance)
        np.testing.assert_allclose([banded, dense], exact, rtol=1e-4, atol=1e-4)
