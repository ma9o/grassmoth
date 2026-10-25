"""State construction binds parameters, trajectories and tuning in one shape scope."""

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from blackjax.adaptation.step_size import dual_averaging_adaptation

from grassmoth.models.ssm.inference.mcmc_state import trajectory_state
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import _initialize_chain_state
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.fixture
def target():
    return interval_target(
        np.arange(3.0), np.zeros((3, 1)), np.arange(3.0)[:, None], ("last",), state_dim=2
    )


@pytest.fixture
def state(target):
    position = target.initial_position
    step = jnp.array(0.01)
    return trajectory_state(
        position=position,
        parameter_gradient=jnp.zeros_like(position),
        latent_context=target.context(position, target.times),
        latent_trajectory=jnp.zeros((3, 2)),
        trajectory_log_prob=jnp.array(0.0),
        complete_log_posterior=jnp.array(0.0),
        latent_delta=jnp.full(3, 0.01),
        param_step_size=step,
        parameter_chol=jnp.eye(position.size),
        param_da=dual_averaging_adaptation(target=0.5)[0](step),
    )


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("parameter_gradient", (2,)),
        ("parameter_chol", (2, 2)),
        ("latent_delta", (4,)),
        ("latent_delta", (3, 3)),
        ("complete_log_posterior", (1,)),
    ],
)
def test_state_rejects_inconsistent_fields(state, field, shape):
    with pytest.raises(jaxtyping.TypeCheckError, match=field):
        trajectory_state(**(state._asdict() | {field: jnp.zeros(shape)}))


def test_state_constructor_preserves_pytree_batching_and_diagonal_path_scales(state):
    def construct(position):
        return trajectory_state(
            **(state._asdict() | {"position": position, "latent_delta": jnp.full((3, 2), 0.01)})
        )

    result = jax.jit(jax.vmap(construct))(jnp.ones((4, 1)))
    assert result.position.shape == (4, 1)
    assert result.parameter_gradient.shape == (4, 1)
    assert result.latent_trajectory.shape == (4, 3, 2)
    assert result.latent_delta.shape == (4, 3, 2)
    assert jax.tree.structure(result) == jax.tree.structure(state)


def test_initializer_rejects_broadcastable_wrong_parameter_gradient(target, state):
    with pytest.raises(jaxtyping.TypeCheckError, match="parameter_gradient"):
        _initialize_chain_state(
            state.position,
            observations=target.observations,
            times=target.times,
            target=target,
            parameter_gradient_fn=lambda _position: jnp.zeros(2),
            initial_latent_delta=state.latent_delta,
            param_step_size=0.01,
            parameter_chol=state.parameter_chol,
            param_min_scale=1e-6,
            param_max_scale=1e3,
            initial_latent_trajectory=state.latent_trajectory,
        )
