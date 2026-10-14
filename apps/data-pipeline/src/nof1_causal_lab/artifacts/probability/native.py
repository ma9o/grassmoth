"""Authoring policy and native NumPyro prior composition.

Only the approved family names and their authoring aliases are application
policy. NumPyro supplies argument constraints, broadcasting, densities,
sampling, moments, and transformed-distribution Jacobians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms

from nof1_causal_lab.artifacts.probability.families import PriorDistributionFamily
from nof1_causal_lab.artifacts.probability.serialization import rebuild_distribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jax.typing import ArrayLike


_PRIOR_CONSTRUCTORS = {
    PriorDistributionFamily.NORMAL: dist.Normal,
    PriorDistributionFamily.HALF_NORMAL: dist.HalfNormal,
    PriorDistributionFamily.BETA: dist.Beta,
    PriorDistributionFamily.UNIFORM: dist.Uniform,
    PriorDistributionFamily.TRUNCATED_NORMAL: dist.TruncatedNormal,
    PriorDistributionFamily.GAMMA: dist.Gamma,
    PriorDistributionFamily.LOG_NORMAL: dist.LogNormal,
    PriorDistributionFamily.EXPONENTIAL: dist.Exponential,
}

_ARGUMENT_ALIASES = {
    "loc": "mu",
    "scale": "sigma",
    "low": "lower",
    "high": "upper",
    "concentration1": "alpha",
    "concentration0": "beta",
}


def prior_argument_constraints(
    family: PriorDistributionFamily,
) -> dict[str, constraints.Constraint]:
    """Native argument constraints for the approved authoring surface."""
    if family == PriorDistributionFamily.TRUNCATED_NORMAL:
        return {
            **dist.Normal.arg_constraints,
            "low": constraints.real,
            "high": constraints.real,
        }
    return dict(_PRIOR_CONSTRUCTORS[family].arg_constraints)


def authored_argument_names(family: PriorDistributionFamily) -> dict[str, str]:
    """Map NumPyro constructor names to the project's authoring vocabulary."""
    return {
        native: _ARGUMENT_ALIASES.get(native, native)
        for native in prior_argument_constraints(family)
    }


def distribution_from_params(
    family: PriorDistributionFamily,
    params: Mapping[str, ArrayLike | Sequence[float]],
) -> dist.Distribution:
    """Validate a complete authored parameter mapping and construct its law."""
    arguments = authored_argument_names(family)
    expected = set(arguments.values())
    if set(params) != expected:
        raise ValueError(
            f"{family.value} requires exactly {sorted(expected)}; got {sorted(params)}"
        )
    native = {
        key: jnp.asarray(params[authored], dtype=jnp.float32) for key, authored in arguments.items()
    }
    if family in {
        PriorDistributionFamily.UNIFORM,
        PriorDistributionFamily.TRUNCATED_NORMAL,
    } and np.any(np.asarray(native["low"]) >= np.asarray(native["high"])):
        raise ValueError(f"{family.value} requires lower < upper")
    return _PRIOR_CONSTRUCTORS[family](**native, validate_args=True)


def distribution_support_bounds(distribution: dist.Distribution) -> tuple[ArrayLike, ArrayLike]:
    """Resolve scalar interval constraints, including transformed supports."""
    support = distribution.support
    if support is constraints.real:
        return -np.inf, np.inf
    if isinstance(support, type(constraints.interval(0.0, 1.0))):
        return support.lower_bound, support.upper_bound
    if isinstance(support, type(constraints.greater_than(0.0))):
        return support.lower_bound, np.inf
    if isinstance(support, type(constraints.less_than(0.0))):
        return -np.inf, support.upper_bound
    raise ValueError(f"Unsupported scalar prior support {support!r}")


def persistence_to_decay(
    distribution: dist.Distribution, interval_days: ArrayLike
) -> dist.Distribution:
    """Push the authored persistence law through decay = -log(rho) / dt."""
    low, high = distribution_support_bounds(distribution)
    if np.any(np.asarray(low) < 0.0) or np.any(np.asarray(high) > 1.0):
        raise ValueError(
            "Persistence priors must have support within [0, 1]; author a Beta, "
            "Uniform, or TruncatedNormal with valid bounds."
        )
    interval = jnp.asarray(interval_days, dtype=jnp.float32)
    if np.any(np.asarray(interval) <= 0.0):
        raise ValueError("The prior reference interval must be positive")
    # Declare the forward map from positive decay to unit-interval persistence,
    # then invert it. Its inverse codomain remains positive under JAX tracing;
    # it does not depend on AffineTransform inferring the sign of a tracer.
    forward = transforms.ComposeTransform(
        [
            transforms.AffineTransform(0.0, -interval, domain=constraints.positive),
            transforms.ExpTransform(domain=constraints.interval(-jnp.inf, 0.0)),
        ]
    )
    return dist.TransformedDistribution(distribution, forward.inv)


def interval_effect_to_rate(
    distribution: dist.Distribution, interval_days: ArrayLike
) -> dist.Distribution:
    """Rescale the entire authored effect law, preserving its family and bounds."""
    interval = jnp.asarray(interval_days, dtype=jnp.float32)
    if np.any(np.asarray(interval) <= 0.0):
        raise ValueError("The prior reference interval must be positive")
    support = distribution.support
    if support is None:
        raise ValueError("A scalar prior must declare its support")
    return dist.TransformedDistribution(
        distribution, transforms.AffineTransform(0.0, 1.0 / interval, domain=support)
    )


def prior_reference_value(distribution: dist.Distribution) -> jax.Array:
    """A diagnostic anchor: the base-law mean passed through declared transforms.

    For nonlinear transforms this is explicitly not the transformed law's mean.
    It never defines a density, a sampler, or a reported posterior moment.
    """
    if isinstance(distribution, (dist.ExpandedDistribution, dist.MaskedDistribution)):
        return jnp.broadcast_to(
            prior_reference_value(distribution.base_dist), distribution.batch_shape
        )
    if isinstance(distribution, dist.MixtureGeneral):
        anchors = jnp.stack(
            [
                jnp.broadcast_to(prior_reference_value(component), distribution.batch_shape)
                for component in distribution.component_distributions
            ],
            axis=-1,
        )
        weights = jnp.asarray(distribution.mixing_distribution.probs)
        return (weights * jnp.where(weights > 0, anchors, 0.0)).sum(axis=-1)
    if type(distribution) is dist.TransformedDistribution:
        value = prior_reference_value(distribution.base_dist)
        for transform in distribution.transforms:
            value = jnp.asarray(transform(value))
        return value
    return jnp.asarray(distribution.mean)


def batch_prior_distributions(
    distributions: Sequence[dist.Distribution],
    shape: tuple[int, ...],
    *,
    support: constraints.Constraint,
) -> dist.Distribution:
    """Assemble independent coordinate laws with NumPyro/JAX composition.

    Matching native parameter trees batch directly. Different trees use a
    deterministic categorical selector per coordinate: a product of prescribed
    laws with no uncertain mixture membership.
    """
    count = int(np.prod(shape))
    if len(distributions) != count:
        raise ValueError(
            f"Prior shape {shape} requires {count} coordinate laws, got {len(distributions)}"
        )
    if not distributions or any(law.batch_shape or law.event_shape for law in distributions):
        raise ValueError("Coordinate priors must be nonempty scalar distributions")
    if not shape:
        return distributions[0]
    laws = [rebuild_distribution(law) for law in distributions]
    trees = [jax.tree.flatten(law) for law in laws]
    leaves, structure = trees[0]
    leaf_shapes = [jnp.shape(leaf) for leaf in leaves]
    if all(
        tree == structure and [jnp.shape(leaf) for leaf in parameters] == leaf_shapes
        for parameters, tree in trees[1:]
    ):
        batched = jax.tree.map(
            lambda *values: jnp.stack(values).reshape((*shape, *jnp.shape(values[0]))), *laws
        )
        return rebuild_distribution(batched)
    selector = dist.Categorical(
        logits=jnp.where(jnp.eye(count).reshape((*shape, count)), 0.0, -jnp.inf)
    )
    # Inactive coordinate laws must not evaluate values outside their domains:
    # e.g. a real coordinate may be negative while another is LogNormal.
    # Native masking supplies support-interior values before differentiation.
    components = [
        distribution.mask(jnp.arange(count).reshape(shape) == index)
        for index, distribution in enumerate(distributions)
    ]
    return dist.MixtureGeneral(selector, components, support=support)
