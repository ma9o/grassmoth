"""Diffrax SDE replay, noise moments, and deterministic-limit contracts."""

from __future__ import annotations

from types import SimpleNamespace

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import jit, vmap

from grassmoth.models.ssm.dynamics.intervention import Intervention
from grassmoth.models.ssm.dynamics.simulator import (
    BrownianTreeSpec,
    IndexedBrownianSpec,
    ProcessNoise,
    SimulationConfig,
    _IndexedBrownianPath,
    _PreparedSteps,
    simulate,
)
from grassmoth.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
from grassmoth.models.ssm.dynamics.vector_field import VectorFieldArgs
from grassmoth_test_support.dynamics import DenseLinear, potential_term, reference_vector_field


def _linear_setup():
    vf = reference_vector_field(n_latent=1, components=(DenseLinear(),))
    params = ({"drift": jnp.array([[-1.0]]), "cint": jnp.array([0.5])},)
    return vf, params, jnp.zeros(1), jnp.linspace(0.0, 2.0, 9)


class TestSimulateSDEMode:
    @pytest.mark.inference(concern="simulation")
    @pytest.mark.parametrize("indexed", [False, True], ids=["brownian-tree", "indexed"])
    def test_replay_and_moments_match_ornstein_uhlenbeck_solution(self, indexed):
        vf, params, y0, time_grid = _linear_setup()
        config = SimulationConfig(
            brownian=IndexedBrownianSpec(step_size=0.02)
            if indexed
            else BrownianTreeSpec(step_size=0.02)
        )
        keys = jr.split(jr.PRNGKey(23), 512)
        # Reuse one compiled batch for replay and analytic moment checks. The
        # repeated key is excluded from the independent sample used for moments.
        samples = jit(
            vmap(
                lambda key: simulate(
                    vf,
                    params,
                    Intervention.none(),
                    y0,
                    time_grid,
                    config=config,
                    noise=ProcessNoise(key=key, diffusion=jnp.linalg.cholesky(jnp.eye(1) * 0.2)),
                )
            )
        )(jnp.concatenate((keys, keys[:1])))
        assert samples.shape == (513, len(time_grid), 1)
        assert bool(jnp.isfinite(samples).all())
        assert jnp.array_equal(samples[0], samples[-1])
        assert not jnp.allclose(samples[0], samples[1], atol=1e-3)

        final_states = samples[:-1, -1, 0]
        expected_mean = 0.5 * (1.0 - jnp.exp(-2.0))
        expected_variance = 0.1 * (1.0 - jnp.exp(-4.0))

        assert jnp.mean(final_states) == pytest.approx(float(expected_mean), abs=0.05)
        assert jnp.var(final_states) == pytest.approx(float(expected_variance), abs=0.02)

    @pytest.mark.inference(concern="simulation")
    def test_zero_diffusion_matches_ode(self):
        vf, params, y0, time_grid = _linear_setup()
        det = simulate(vf, params, Intervention.none(), y0, time_grid)
        sde = simulate(
            vf,
            params,
            Intervention.none(),
            y0,
            time_grid,
            noise=ProcessNoise(
                key=jr.PRNGKey(42), diffusion=jnp.linalg.cholesky(jnp.eye(1) * 1e-10)
            ),
        )
        assert jnp.allclose(det, sde, atol=5e-3)


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize("indexed", [False, True], ids=["brownian-tree", "indexed"])
def test_irregular_prepared_times_are_actual_solver_endpoints(monkeypatch, indexed):
    grid = jnp.array([2.0, 2.00003, 2.019, 2.053, 2.1])
    vf, params, initial, _ = _linear_setup()
    native_solve = dfx.diffeqsolve
    recorded = []

    def solve(*args, **kwargs):
        if not indexed:
            brownian = args[0].terms[1].control
            # Diffrax stores tree tolerance normalized to its full time span.
            assert brownian.tol * (grid[-1] - grid[0]) < jnp.min(jnp.diff(grid))
        solution = native_solve(
            *args,
            **{
                **kwargs,
                "saveat": dfx.SaveAt(
                    subs={
                        "prepared": dfx.SubSaveAt(ts=grid),
                        "steps": dfx.SubSaveAt(t0=True, steps=True),
                    }
                ),
            },
        )
        recorded.append(solution)
        return SimpleNamespace(ys=solution.ys["prepared"])

    monkeypatch.setattr(dfx, "diffeqsolve", solve)
    result = simulate(
        vf,
        params,
        Intervention.none(),
        initial,
        grid,
        SimulationConfig(
            max_steps=3,  # Mandatory prepared intervals have their own allowance.
            brownian=IndexedBrownianSpec(step_size=0.02)
            if indexed
            else BrownianTreeSpec(step_size=0.02),
        ),
        noise=ProcessNoise(key=jr.key(71), diffusion=jnp.array([[0.3]])),
    )
    endpoints = np.asarray(recorded[0].ts["steps"])
    endpoints = endpoints[np.isfinite(endpoints)]
    assert np.isfinite(result).all()
    assert np.isin(np.asarray(grid), endpoints).all()
    assert len(endpoints) > len(grid)
    assert np.all(np.diff(endpoints) > 0)
    assert np.max(np.diff(endpoints)) <= 0.020001
    # SaveAt only evaluates at true solver endpoints, including short irregular gaps.
    actual_states = np.asarray(recorded[0].ys["steps"])[
        np.searchsorted(endpoints, np.asarray(grid))
    ]
    np.testing.assert_allclose(result, actual_states, atol=2e-7)


@pytest.mark.inference(concern="simulation")
def test_irregular_indexed_increments_are_independent_and_replay_exactly():
    grid = jnp.array([0.0, 0.04, 0.055, 0.19, 0.35])
    steps = _PreparedSteps(grid, jnp.asarray(0.1))
    # Includes clipped prepared intervals and additional resolution substeps.
    endpoints = jnp.array([0.0, 0.04, 0.055, 0.1225, 0.19, 0.27, 0.35])
    key = jr.key(390)
    path = _IndexedBrownianPath(steps, (8192,), key)
    increments = jit(vmap(path.evaluate))(endpoints[:-1], endpoints[1:])
    replay = _IndexedBrownianPath(steps, (8192,), jr.key(390))
    np.testing.assert_array_equal(
        increments, jit(vmap(replay.evaluate))(endpoints[:-1], endpoints[1:])
    )
    standardized = np.asarray(increments / jnp.sqrt(jnp.diff(endpoints))[:, None])
    np.testing.assert_allclose(standardized.mean(1), 0, atol=0.05)
    np.testing.assert_allclose(np.cov(standardized), np.eye(len(endpoints) - 1), atol=0.055)
    levy = path.evaluate(endpoints[2], endpoints[3], use_levy=True)
    np.testing.assert_array_equal(levy.W, increments[2])


@pytest.mark.inference(concern="simulation")
def test_fine_grid_nonlinear_heun_paths_resolve_em_innovations(record_property):
    field = compile_dynamics(
        DynamicsSpec(1, (potential_term(0, center=0, stiffness=0.6, quartic=0.05),))
    ).vector_field
    grid = jnp.linspace(0.0, 4.0, 513)
    step = 4.0 / 200  # The old heuristic skipped more than half the requested times.
    sigma = 0.35
    keys = jr.split(jr.key(721), 128)
    config = SimulationConfig(brownian=IndexedBrownianSpec(step_size=step))
    paths = jit(
        vmap(
            lambda key: simulate(
                field,
                ({},),
                Intervention.none(),
                jnp.zeros(1),
                grid,
                config,
                noise=ProcessNoise(key, jnp.array([[sigma]])),
            )
        )
    )(keys)
    args = VectorFieldArgs(params=({},), intervention=Intervention.none())
    drifts = jit(vmap(vmap(lambda t, x: field(t, x, args), in_axes=(0, 0)), in_axes=(None, 0)))(
        grid[:-1], paths[:, :-1]
    )
    dt = np.diff(np.asarray(grid))
    residuals = (np.diff(np.asarray(paths[..., 0]), axis=1) - dt * np.asarray(drifts[..., 0])) / (
        sigma * np.sqrt(dt)
    )
    increments = jit(
        vmap(
            lambda key: vmap(
                _IndexedBrownianPath(_PreparedSteps(grid, jnp.asarray(step)), (1,), key).evaluate
            )(grid[:-1], grid[1:])
        )
    )(keys)
    innovations = np.asarray(increments[..., 0]) / np.sqrt(dt)
    count = residuals.size
    mean, sd = float(residuals.mean()), float(residuals.std(ddof=1))
    lag1 = float(np.corrcoef(residuals[:, :-1].ravel(), residuals[:, 1:].ravel())[0, 1])
    difference_rms = float(np.sqrt(np.mean((residuals - innovations) ** 2)))
    for name, value in {
        "count": count,
        "em_residual_mean": mean,
        "em_residual_sd": sd,
        "em_residual_lag1": lag1,
        "heun_em_standardized_rms_difference": difference_rms,
    }.items():
        record_property(name, value)
    assert abs(mean) < 5 / np.sqrt(count)
    assert abs(sd - 1) < 5 / np.sqrt(2 * count)
    assert abs(lag1) < 5 / np.sqrt(count)
    assert 0.001 < difference_rms < 0.01  # Heun remains distinct from EM at this resolution.
