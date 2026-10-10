"""Tests for vector-field transition construction."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx import (
    AffineDrift,
    StochasticContinuousTimeStateEvolution,
)
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.covariance_utils import CHOL_JITTER
from nof1_causal_lab.models.ssm.dynamics.edges import (
    DenseLinear,
    DiagonalDecay,
    Intercept,
    LinearEdge,
    StateDecay,
    StateIntercept,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import (
    StructuralDrift,
    VectorField,
)
from nof1_causal_lab.models.ssm.execution.contracts import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.execution.dynamical_model import (
    continuous_state_evolution,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    _transition_start_linearization_states,
)
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from tests.dynamics_fixtures import hill_term
from tests.inference_fixtures import make_observation_support_runtime
from tests.observation_fixtures import observation_laws


def _structural_drift(dynamics) -> StructuralDrift:
    assert isinstance(dynamics.drift, StructuralDrift)
    return dynamics.drift


def _expected_affine_drift():
    return AffineDrift(
        A=jnp.array([[-0.50, 0.05, 0.0], [-0.10, -0.55, 0.02], [0.0, 0.0, -0.65]]),
        b=jnp.array([0.10, -0.04, 0.02]),
    )


def _constant_runtime_dynamics() -> StochasticContinuousTimeStateEvolution:
    drift = jnp.array(
        [
            [-0.30, 0.05, 0.00],
            [0.00, -0.40, 0.02],
            [0.00, 0.00, -0.50],
        ],
        dtype=jnp.float32,
    )
    cint = jnp.array([0.03, -0.04, 0.02], dtype=jnp.float32)
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=3,
            components=(
                DenseLinear(),
                DiagonalDecay(),
                StateDecay(target=2),
                Intercept(),
                StateIntercept(target=0),
                LinearEdge(source=0, target=1),
            ),
        ),
        vf_params=(
            {"drift": drift},
            {"decay": jnp.array([0.20, 0.15, 0.10], dtype=jnp.float32)},
            {"decay": jnp.array(0.05, dtype=jnp.float32)},
            {"cint": cint},
            {"cint": jnp.array(0.07, dtype=jnp.float32)},
            {"weight": jnp.array(-0.10, dtype=jnp.float32)},
        ),
        diffusion=jnp.linalg.cholesky(jnp.diag(jnp.array([0.08, 0.06, 0.04], dtype=jnp.float32))),
    )


def _trajectory_runtime_dynamics() -> StochasticContinuousTimeStateEvolution:
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                hill_term(source=0, target=1).build(),
            ),
        ),
        vf_params=(
            {"decay": jnp.array([0.35, 0.45], dtype=jnp.float32)},
            {
                scientific_id("parameter", "emax"): jnp.array(0.80, dtype=jnp.float32),
                scientific_id("parameter", "ec50"): jnp.array(1.20, dtype=jnp.float32),
                scientific_id("parameter", "exponent"): jnp.array(2.0, dtype=jnp.float32),
            },
        ),
        diffusion=jnp.linalg.cholesky(jnp.diag(jnp.array([0.05, 0.07], dtype=jnp.float32))),
    )


def _interval_mean_support_runtime() -> ObservationSupportRuntime:
    return make_observation_support_runtime(
        anchor_times=np.array([0.0, 0.50, 1.25], dtype=np.float64),
        manifest_names=("mean_signal",),
        support_kinds=("interval",),
        summary_operators=("mean",),
        anchor_policies=("support_end",),
        observation_windows=("previous_interval",),
        support_start_times=np.array([[np.nan], [0.0], [0.50]], dtype=np.float64),
        support_end_times=np.array([[np.nan], [0.50], [1.25]], dtype=np.float64),
    )


@pytest.mark.inference(concern="warmup")
def test_constant_vector_field_local_linearization_matches_affine_view():
    dynamics = _constant_runtime_dynamics()
    affine = _expected_affine_drift()
    x_lin = jnp.array([0.70, -0.20, 0.40], dtype=jnp.float32)

    args = _structural_drift(dynamics).args
    field = _structural_drift(dynamics).vector_field
    drift = jax.jacfwd(lambda x: field(jnp.array(0.0), x, args))(x_lin)
    cint = field(jnp.array(0.0), x_lin, args) - drift @ x_lin

    np.testing.assert_allclose(drift, affine.A, rtol=1e-6, atol=1e-6)
    assert affine.b is not None
    np.testing.assert_allclose(cint, affine.b, rtol=1e-6, atol=1e-6)


@pytest.mark.contract
def test_transition_builder_requires_states_for_trajectory_dependent_dynamics():
    dynamics = _trajectory_runtime_dynamics()
    time_intervals = jnp.array([0.20, 0.50, 1.10], dtype=jnp.float32)

    with pytest.raises(ValueError, match="linearization_states"):
        build_discrete_transitions(dynamics, time_intervals)


@pytest.mark.inference(concern="warmup")
def test_initialization_covariance_handles_large_diffusion_candidates():
    """Large, low-density Pathfinder draws must remain scoreable in a mixed batch."""
    intervals = jnp.array([0.01, 0.5, 1.0], dtype=jnp.float32)
    decay = 0.08705525

    def transitions(sigma):
        dynamics = continuous_state_evolution(
            vector_field=VectorField(n_latent=1, components=(DiagonalDecay(),)),
            vf_params=({"decay": jnp.array([decay])},),
            diffusion=jnp.linalg.cholesky(jnp.reshape(sigma**2, (1, 1))),
        )
        return build_discrete_transitions(dynamics, intervals)

    scales = jnp.array([0.3, 1617.1584], dtype=jnp.float32)
    result = jax.jit(jax.vmap(transitions))(scales)
    factor = -np.expm1(-2 * decay * np.asarray(intervals)) / (2 * decay)
    np.testing.assert_allclose(
        result.cov[:, :, 0, 0],
        np.asarray(scales[:, None]) ** 2 * factor,
        rtol=2e-5,
    )
    np.testing.assert_allclose(result.A[1, :, 0, 0], np.exp(-decay * intervals), rtol=2e-5)
    derivative = jax.jit(jax.grad(lambda sigma: transitions(sigma).cov.sum()))(scales[1])
    np.testing.assert_allclose(derivative, 2 * scales[1] * factor.sum(), rtol=2e-5)


@pytest.mark.contract
def test_point_dynamic_linearization_states_use_interval_starts():
    init_mean = jnp.array([0.20, -0.10], dtype=jnp.float32)
    latent_trajectory = jnp.array(
        [
            [0.30, 0.10],
            [1.00, -0.20],
            [1.60, 0.50],
        ],
        dtype=jnp.float32,
    )

    states = _transition_start_linearization_states(latent_trajectory, init_mean)

    expected = jnp.array(
        [
            [0.20, -0.10],
            [0.30, 0.10],
            [1.00, -0.20],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(states, expected)


@pytest.mark.inference(concern="warmup")
def test_accounting_transition_scaling_preserves_integrated_brownian_law_and_gradient():
    """Large unit conversion is not stiffness: compare to the analytic integral law."""
    with jax.enable_x64():
        interval, sigma, rate = 365.0, 0.3, 0.2

        def transitions(gain):
            dynamics = continuous_state_evolution(
                vector_field=VectorField(n_latent=3, components=(DenseLinear(), Intercept())),
                vf_params=(
                    {"drift": jnp.array([[0.0, 0.0, 0.0], [gain, 0.0, 0.0], [0.0, 0.0, 0.0]])},
                    {"cint": jnp.array([rate, 0.0, 0.0])},
                ),
                diffusion=jnp.diag(jnp.array([sigma, 0.0, 0.0])),
            )
            return build_discrete_transitions(dynamics, jnp.array([interval]))

        gain = jnp.asarray(1e6)
        result = jax.jit(transitions)(gain)
        expected_covariance = sigma**2 * np.array(
            [
                [interval, gain * interval**2 / 2, 0.0],
                [gain * interval**2 / 2, gain**2 * interval**3 / 3, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(
            result.A[0], [[1, 0, 0], [gain * interval, 1, 0], [0, 0, 1]], rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            result.bias[0], [rate * interval, rate * gain * interval**2 / 2, 0], rtol=1e-8
        )
        np.testing.assert_allclose(
            result.cov[0], expected_covariance + CHOL_JITTER * np.eye(3), rtol=1e-7, atol=1e-9
        )
        gradient = jax.jit(jax.grad(lambda value: transitions(value).cov[0, 1, 1]))(gain)
        np.testing.assert_allclose(gradient, 2 * gain * sigma**2 * interval**3 / 3, rtol=1e-7)


@pytest.mark.inference(concern="warmup")
@pytest.mark.parametrize("solver", ["point", "dense", "banded"])
def test_nonlinear_laplace_backends_match_finite_difference(monkeypatch, solver):
    """Check nonlinear parameter gradients and diagnostics at each solver boundary."""
    from nof1_causal_lab.models.ssm.inference.targets import laplace

    # Size-based routing has a separate contract; exercise banded algebra on the
    # same small nonlinear path instead of allocating 81 anchors to reach it.
    monkeypatch.setattr(
        laplace, "_should_use_dense_support_laplace", lambda **_kwargs: solver == "dense"
    )
    base_dynamics = _trajectory_runtime_dynamics()
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        n_ieks_iters=2,
        observed_mask=np.ones((3, 1), dtype=bool),
        observation_support=None if solver == "point" else _interval_mean_support_runtime(),
    )
    measurement = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial = MultivariateNormal(
        loc=jnp.array([0.20, -0.10], dtype=jnp.float32),
        covariance_matrix=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.array([[0.05], [0.20], [0.12]], dtype=jnp.float32)
    if solver != "point":
        observations = observations.at[0, 0].set(jnp.nan)
    intervals = jnp.array([0.01, 0.50, 0.75], dtype=jnp.float32)

    def _objective(emax):
        hill_params = dict(_structural_drift(base_dynamics).args.params[1])
        hill_params[scientific_id("parameter", "emax")] = emax
        dynamics = eqx.tree_at(
            lambda model: _structural_drift(model).args.params,
            base_dynamics,
            (_structural_drift(base_dynamics).args.params[0], hill_params),
        )
        result = backend.compute_log_likelihood_with_aux(
            dynamics,
            measurement,
            initial,
            observations,
            intervals,
            observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
        )
        return result.log_likelihood, result

    evaluate = jax.jit(jax.value_and_grad(_objective, has_aux=True))
    emax = jnp.array(0.8, dtype=jnp.float32)
    (value, aux), gradient = evaluate(emax)
    eps = 1e-2
    (plus, _), _ = evaluate(emax + eps)
    (minus, _), _ = evaluate(emax - eps)
    assert np.isfinite(value)
    assert aux.state.latent_mode.shape == (3, 2)
    assert (
        int(aux.diagnostics["solver_kind"])
        == {
            "point": LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
            "dense": LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
            "banded": LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
        }[solver]
    )
    assert abs(float(gradient)) > 1e-4
    np.testing.assert_allclose(gradient, (plus - minus) / (2 * eps), rtol=1e-2, atol=1e-4)
