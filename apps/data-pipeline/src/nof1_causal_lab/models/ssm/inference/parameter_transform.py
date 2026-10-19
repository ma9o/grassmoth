"""Flatten NumPyro's parameter coordinates for the local particle sampler."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


@dataclass(frozen=True)
class ParameterTransform:
    """NumPyro's prior density and constrained replay in flat coordinates."""

    initial_position: jax.Array
    unravel: Callable[[jax.Array], dict[str, jax.Array]]
    constrain: Callable[[jax.Array], dict[str, jax.Array]]
    log_prior: Callable[[jax.Array], jax.Array]


def prepare_parameter_transform(
    prior_model: Callable[..., object],
    key: jax.Array,
    *,
    model_args: tuple[object, ...],
    initial_values: Mapping[str, jax.Array],
) -> ParameterTransform:
    """Keep conditional priors, factors, transforms, and Jacobians in NumPyro."""
    info, potential, postprocess, _ = initialize_model(
        # A singleton key batch stages NumPyro's initial density and gradient
        # validation together; its single-key path executes the first try eagerly.
        key[None],
        prior_model,
        model_args=model_args,
        init_strategy=init_to_value(values=initial_values),
    )
    position = jax.tree.map(lambda value: value[0], info.z)
    initial, unravel = ravel_pytree(position)
    return ParameterTransform(
        initial,
        unravel,
        lambda position: postprocess(unravel(position)),
        lambda position: -potential(unravel(position)),
    )
