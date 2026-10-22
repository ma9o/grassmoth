"""Display curves for fitted input laws, evaluated by NumPyro on the quantity scale."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from grassmoth.artifacts.reports.posterior_diagnostics import DensityCurve

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import ParameterId
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


def prior_density(prior: dist.Distribution) -> DensityCurve:
    """Plot scalar continuous laws; joint, batched and point masses have no scalar PDF.

    A small deterministic native draw selects only the viewport. Every ordinate
    is the native log_prob, never a density estimate or an inference input.
    """
    if prior.batch_shape or prior.event_shape or prior.is_discrete or isinstance(prior, dist.Delta):
        return DensityCurve()
    return _density_curve(prior)


@lru_cache(maxsize=128)
def _density_curve(prior: dist.Distribution) -> DensityCurve:
    draws = np.asarray(prior.sample(jax.random.PRNGKey(0), (256,)))
    low, high = np.quantile(draws, [0.01, 0.99])
    x = jnp.linspace(low, high, 100)
    y = np.asarray(jnp.exp(prior.log_prob(x)))
    points = tuple(
        (float(value), float(density))
        for value, density in zip(np.asarray(x), y, strict=True)
        if np.isfinite(value) and np.isfinite(density)
    )
    return DensityCurve(
        x=tuple(point[0] for point in points), density=tuple(point[1] for point in points)
    )


def quantity_prior_densities(
    dynamical_model_spec: DynamicalModelSpec,
    parameters: frozenset[ParameterId],
) -> dict[ParameterId, DensityCurve]:
    """Resolve the fit's input quantity laws before retaining their density curves."""
    from grassmoth.artifacts.probability.serialization import distribution_shape
    from grassmoth.models.ssm.compile.prior_compilation import quantity_parameter_law

    return {
        parameter.id: prior_density(quantity_parameter_law(dynamical_model_spec, parameter)[0])
        for parameter in dynamical_model_spec.parameters
        if parameter.id in parameters
        if parameter.distribution is not None
        and not any(distribution_shape(dynamical_model_spec.distributions[parameter.distribution]))
    }
