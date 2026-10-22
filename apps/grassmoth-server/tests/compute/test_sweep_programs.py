"""Staged sweep programs bind each fit's numerical operands."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.models.ssm.inference.samplers import sweep_plan, sweeps
from grassmoth_server.compute.compilation_cache import prepare_program
from grassmoth_test_support.compute import COMPUTE_BUDGET

pytestmark = pytest.mark.inference(concern="sampling")


def test_runtime_inputs_rebind_nested_observations_priors_geometry_and_tuning():
    """A compiled sweep accepts another same-layout fit's numerical inputs."""

    def factory(values):
        observations, location, scale, geometry, tuning = values
        prior = dist.Normal(location, scale)

        @partial(jax.jit, inline=jax.Inline.JAX_EARLY)
        def transition(state):
            position = geometry @ state + tuning * observations
            return position, {"density": prior.log_prob(position)}

        def step(state, _posterior):
            return transition(state)

        return lambda _batch: step

    initial = jnp.ones((2, 1))
    first = (jnp.zeros((2, 1)), jnp.asarray(0.0), jnp.asarray(1.0), jnp.eye(2), jnp.asarray(0.1))
    second = (
        jnp.ones((2, 1)),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
        jnp.eye(2) * 2,
        jnp.asarray(0.2),
    )
    plans = tuple(
        sweeps.compile_sweeps(
            factory(values),
            initial,
            num_warmup=1,
            num_samples=1,
            max_batch_size=1,
            budget=COMPUTE_BUDGET,
            prepare_program=prepare_program,
        )
        for values in (first, second)
    )
    first_phase, second_phase = (plan.posterior for plan in plans)
    assert isinstance(first_phase, sweep_plan.CompiledSweepPhase)
    assert isinstance(second_phase, sweep_plan.CompiledSweepPhase)
    state, actual = first_phase.executable(
        second_phase.arguments, initial, jnp.asarray(1), jnp.asarray(True)
    )
    expected_state, expected = factory(second)(1)(initial, jnp.asarray(True))
    np.testing.assert_array_equal(state, expected_state)
    np.testing.assert_array_equal(actual["density"][0], expected["density"])
