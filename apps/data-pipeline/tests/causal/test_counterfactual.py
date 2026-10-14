"""Numerical contracts for exact nonlinear simulation and posterior summaries."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.counterfactual import (
    summarize_draws,
)
from nof1_causal_lab.models.ssm.dynamics import (
    ConstantValueFn,
    EdgeInputOverride,
    Intervention,
    LinearRampValueFn,
    SimulationConfig,
    VariableOverride,
    VectorField,
    VectorFieldArgs,
    simulate,
)
from tests.causal.steady_state_reference import compute_steady_state
from tests.dynamics_fixtures import (
    DenseLinear,
)


def resolve_action_value(baseline_value, *, mode, value=None, amount=None):
    baseline = jnp.asarray(baseline_value)
    if mode == "set":
        if value is None:
            raise ValueError("mode='set' requires value")
        return jnp.asarray(value, dtype=baseline.dtype)
    if mode == "shift":
        if amount is None:
            raise ValueError("mode='shift' requires amount")
        return baseline + jnp.asarray(amount, dtype=baseline.dtype)
    raise ValueError(f"Unsupported action mode: {mode}")


def vmap_steady_state_effect_dynamics(
    vector_field,
    param_samples,
    treat_idx,
    outcome_idx,
    *,
    mode,
    value=None,
    amount=None,
):
    import jax

    if not param_samples:
        return jnp.zeros((0,))
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *param_samples)

    def per_draw(params):
        baseline = compute_steady_state(vector_field, params, Intervention.none())
        do_value = resolve_action_value(baseline[treat_idx], mode=mode, value=value, amount=amount)
        intervention = Intervention(
            overrides=(VariableOverride(index=treat_idx, value_fn=ConstantValueFn(do_value)),)
        )
        intervened = compute_steady_state(
            vector_field, params, intervention, initial_guess=baseline
        )
        return intervened[outcome_idx] - baseline[outcome_idx]

    return jax.vmap(per_draw)(stacked)


def _dense_matrix_vector_field(n_latent: int) -> VectorField:
    return VectorField(n_latent=n_latent, components=(DenseLinear(),))


# =============================================================================
# dense-linear VectorField + Intervention DSL
# =============================================================================


@pytest.mark.inference(concern="simulation")
class TestDenseLinearRuntime:
    def test_dynamics_matches_matrix_form(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.5], [0.0, -2.0]]), "cint": jnp.array([0.3, -0.1])},)
        eta = jnp.array([1.0, 2.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        out = vf(jnp.asarray(0.0), eta, args)
        expected = params[0]["drift"] @ eta + params[0]["cint"]
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_variable_override_forces_zero_dynamics_for_constant(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.5], [0.0, -2.0]]), "cint": jnp.zeros(2)},)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=ConstantValueFn(jnp.asarray(5.0))),)
        )
        args = VectorFieldArgs(params=params, intervention=intervention)
        out = vf(jnp.asarray(0.0), jnp.array([5.0, 2.0]), args)
        assert jnp.isclose(out[0], 0.0, atol=1e-6)

    def test_edge_input_override_changes_target_only(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        params = ({"drift": A, "cint": jnp.zeros(2)},)
        eta = jnp.array([1.0, 0.0])
        baseline_args = VectorFieldArgs(params=params, intervention=Intervention.none())
        baseline_dynamics = vf(jnp.asarray(0.0), eta, baseline_args)
        intervention = Intervention(
            overrides=(
                EdgeInputOverride(source=0, target=1, value_fn=ConstantValueFn(jnp.asarray(10.0))),
            )
        )
        intervened_dynamics = vf(
            jnp.asarray(0.0), eta, VectorFieldArgs(params=params, intervention=intervention)
        )
        assert jnp.isclose(intervened_dynamics[0], baseline_dynamics[0], atol=1e-6)
        assert intervened_dynamics[1] > baseline_dynamics[1]

    def test_initial_condition_applies_variable_overrides(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        intervention = Intervention(
            overrides=(VariableOverride(index=1, value_fn=ConstantValueFn(jnp.asarray(3.0))),)
        )
        args = VectorFieldArgs(
            params=({"drift": -jnp.eye(2), "cint": jnp.zeros(2)},), intervention=intervention
        )
        y0 = vf.initial_condition(jnp.array([1.0, 2.0]), args)
        assert jnp.isclose(y0[0], 1.0)
        assert jnp.isclose(y0[1], 3.0)


# =============================================================================
# compute_steady_state (numerical, replaces closed-form -A^{-1} c)
# =============================================================================


@pytest.mark.inference(concern="simulation")
class TestComputeSteadyState:
    def test_intervention_propagates_downstream(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        params = (
            {
                "drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]),
                "cint": jnp.array([2.0, 1.0]),
            },
        )
        baseline = compute_steady_state(vf, params, Intervention.none())
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=ConstantValueFn(jnp.asarray(5.0))),)
        )
        intervened = compute_steady_state(vf, params, intervention, initial_guess=baseline)
        assert jnp.isclose(intervened[0], 5.0, atol=1e-4)
        assert not jnp.isclose(intervened[1], baseline[1], atol=0.1)


# =============================================================================
# simulate
# =============================================================================


@pytest.mark.inference(concern="simulation")
class TestSimulate:
    def test_no_coupling_no_propagation(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        params = ({"drift": -jnp.eye(2), "cint": jnp.array([1.0, 2.0])},)
        time_grid = jnp.linspace(0.0, 5.0, 11)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=ConstantValueFn(jnp.asarray(0.0))),)
        )
        baseline = compute_steady_state(vf, params, Intervention.none())
        reference = simulate(vf, params, Intervention.none(), baseline, time_grid)
        action = simulate(vf, params, intervention, baseline, time_grid)
        effect = action - reference
        assert jnp.all(jnp.abs(effect[:, 1]) < 1e-3)

    def test_clamped_state_tracks_constant(self):
        vf = _dense_matrix_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]), "cint": jnp.zeros(2)},)
        time_grid = jnp.linspace(0.0, 5.0, 11)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=ConstantValueFn(jnp.asarray(3.0))),)
        )
        traj = simulate(vf, params, intervention, jnp.array([0.0, 0.0]), time_grid)
        assert jnp.allclose(traj[:, 0], 3.0, atol=1e-3)


# =============================================================================
# Estimand helpers
# =============================================================================


class TestSummarizeDraws:
    @pytest.mark.contract
    @pytest.mark.parametrize("draws", [jnp.array([]), jnp.ones((2, 2)), jnp.array([jnp.nan])])
    def test_rejects_unreportable_draws(self, draws):
        with pytest.raises(ValueError, match=r"nonempty vector|finite number"):
            summarize_draws(draws)

    @pytest.mark.inference(concern="simulation")
    def test_reports_mean_interval_and_prob_positive(self):
        draws = jnp.array([-1.0, 0.0, 2.0, 3.0])
        summary = summarize_draws(draws)
        assert summary.mean == 1.0
        assert summary.median == 1.0
        assert summary.prob_positive == 0.5
        assert summary.lower_95 == pytest.approx(-0.925)
        assert summary.upper_95 == pytest.approx(2.925)


@pytest.mark.contract
class TestLinearRamp:
    def test_holds_endpoints(self):
        ramp = LinearRampValueFn(
            t_start=jnp.asarray(1.0),
            t_end=jnp.asarray(3.0),
            value_start=jnp.asarray(10.0),
            value_end=jnp.asarray(0.0),
        )
        assert float(ramp(jnp.asarray(0.0))) == 10.0
        assert float(ramp(jnp.asarray(5.0))) == 0.0
        assert float(ramp(jnp.asarray(2.0))) == pytest.approx(5.0, abs=1e-6)


# =============================================================================
# =============================================================================


@pytest.mark.inference(concern="simulation")
class TestNumericalCorrectness:
    """Regression tests pinning the numerical Diffrax+Optimistix paths to the
    closed-form linear math they replace. Catches solver-tolerance or step-size
    regressions against a dense-matrix closed form."""

    def test_trajectory_matches_closed_form(self):
        """``simulate`` vs ``exp(A·t)·η₀ + A⁻¹·(exp(A·t) - I)·c``."""
        import jax.scipy.linalg as jla
        from jax import vmap

        A = jnp.array([[-1.0, 0.0], [0.5, -2.0]])
        c = jnp.array([1.0, -0.5])
        eta0 = jnp.array([0.5, 1.5])
        time_grid = jnp.linspace(0.0, 5.0, 21)
        identity = jnp.eye(A.shape[0])

        def analytic(t):
            expAt = jla.expm(A * t)
            return expAt @ eta0 + jla.solve(A, expAt - identity) @ c

        closed_form = vmap(analytic)(time_grid)
        numerical = simulate(
            _dense_matrix_vector_field(n_latent=A.shape[0]),
            ({"drift": A, "cint": c},),
            Intervention.none(),
            eta0,
            time_grid,
            config=SimulationConfig(rtol=1e-8, atol=1e-10),
        )

        # float32-native: trajectory integration vs. closed-form matrix-exp agree
        # to float32 precision, not float64 bit-equality.
        assert jnp.allclose(numerical, closed_form, atol=1e-4)

    def test_steady_state_matches_inverse_for_coupled_system(self):
        """``compute_steady_state`` vs ``-A⁻¹·c`` for a non-trivial 3-coupled system."""
        import jax.scipy.linalg as jla

        A = jnp.array(
            [
                [-1.5, 0.3, 0.1],
                [0.4, -2.0, 0.2],
                [0.1, 0.5, -1.8],
            ]
        )
        c = jnp.array([1.0, -0.5, 0.2])

        closed_form = -jla.solve(A, c)
        numerical = compute_steady_state(
            _dense_matrix_vector_field(n_latent=3),
            ({"drift": A, "cint": c},),
            Intervention.none(),
        )

        assert jnp.allclose(numerical, closed_form, atol=1e-6)

    def test_treatment_effect_matches_closed_form_do(self):
        """Dynamics vmap helper vs hand-computed ``do(η_j=v) - baseline``."""
        import jax.scipy.linalg as jla

        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        treat_idx, outcome_idx = 0, 1
        shift_size = 1.5

        baseline = -jla.solve(A, c)
        do_value = baseline[treat_idx] + shift_size
        A_mod = A.at[treat_idx, :].set(0.0).at[treat_idx, treat_idx].set(1.0)
        rhs = (-c).at[treat_idx].set(do_value)
        intervened = jla.solve(A_mod, rhs)
        expected = float(intervened[outcome_idx] - baseline[outcome_idx])

        vf = _dense_matrix_vector_field(n_latent=A.shape[0])
        numerical = float(
            vmap_steady_state_effect_dynamics(
                vf,
                [({"drift": A, "cint": c},)],
                treat_idx=treat_idx,
                outcome_idx=outcome_idx,
                mode="shift",
                amount=shift_size,
            )[0]
        )

        assert abs(numerical - expected) < 1e-5
