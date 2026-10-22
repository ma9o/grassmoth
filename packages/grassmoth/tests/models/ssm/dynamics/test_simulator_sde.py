"""Diffrax SDE replay, noise moments, and deterministic-limit contracts."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import jit, vmap

from grassmoth.models.ssm.dynamics.intervention import Intervention
from grassmoth.models.ssm.dynamics.simulator import (
    BrownianTreeSpec,
    IndexedBrownianSpec,
    ProcessNoise,
    SimulationConfig,
    simulate,
)
from tests.support.dynamics import DenseLinear, reference_vector_field


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
