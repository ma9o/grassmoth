"""Scientific default priors, represented directly by NumPyro distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

from nof1_causal_lab.artifacts.model.parameter import SupportClass
from nof1_causal_lab.artifacts.probability.families import PriorDistributionFamily
from nof1_causal_lab.artifacts.probability.native import (
    distribution_from_params,
    distribution_support_bounds,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

DEFAULT_PRIORS_BY_FIELD: dict[str, dist.Distribution] = {
    "dynamics_decay": dist.Gamma(concentration=2.0, rate=4.0),
    "dynamics_cint": dist.Normal(loc=0.0, scale=1.0),
    "dynamics_potential_center": dist.Normal(loc=0.0, scale=1.0),
    "dynamics_potential_quartic": dist.HalfNormal(scale=0.5),
    "linear_edge_weight": dist.Normal(loc=0.0, scale=0.5),
    "multiplicative_weight": dist.Normal(loc=0.0, scale=1.0),
    "hill_emax": dist.LogNormal(loc=0.0, scale=1.0),
    "hill_ec50": dist.LogNormal(loc=0.0, scale=1.0),
    "hill_n": dist.TruncatedNormal(loc=2.0, scale=0.5, low=1.0, high=4.0),
    "diffusion_diag": dist.HalfNormal(scale=1.0),
    "diffusion_offdiag": dist.Normal(loc=0.0, scale=0.5),
    "static_state_sd": dist.HalfNormal(scale=1.0),
    "lambda_free": dist.Normal(loc=0.5, scale=0.5),
    "manifest_means": dist.Normal(loc=0.0, scale=2.0),
    "manifest_var_diag": dist.HalfNormal(scale=1.0),
    "obs_df": dist.Gamma(concentration=5.0, rate=1.0),
    "obs_shape": dist.Gamma(concentration=2.0, rate=1.0),
    "obs_r": dist.Gamma(concentration=2.0, rate=0.5),
    "obs_concentration": dist.Gamma(concentration=5.0, rate=0.5),
    "obs_ordered_base": dist.Normal(loc=0.0, scale=1.0),
    "obs_ordered_gaps": dist.HalfNormal(scale=1.0),
    "obs_cat_intercepts": dist.Normal(loc=0.0, scale=1.0),
    "obs_cat_slopes": dist.Normal(loc=0.0, scale=1.0),
    "proc_df": dist.Gamma(concentration=5.0, rate=1.0),
    "t0_means": dist.Normal(loc=0.0, scale=2.0),
    "t0_var_diag": dist.HalfNormal(scale=2.0),
    "t0_var_offdiag": distribution_from_params(
        PriorDistributionFamily.TRUNCATED_NORMAL,
        {"mu": 0.0, "sigma": 0.5, "lower": -1.0, "upper": 1.0},
    ),
}


def default_prior_for_descriptor(site: SiteDescriptor) -> dist.Distribution:
    """Return the scientific default for an active site."""
    if site.prior_field is None:
        raise AggregatedCompileError([f"Site {site.name!r} has no prior field"])
    return DEFAULT_PRIORS_BY_FIELD[site.prior_field]


def site_constraint(site: SiteDescriptor) -> constraints.Constraint:
    """The scientific value domain of a scalar site coordinate."""
    return {
        SupportClass.REAL: constraints.real,
        SupportClass.POSITIVE: constraints.positive,
        SupportClass.CORRELATION: constraints.interval(-1.0, 1.0),
    }[site.support]


def validate_site_prior(site: SiteDescriptor, prior: dist.Distribution) -> None:
    """Check a native law against the site's scientific value domain."""
    if prior.event_shape:
        raise AggregatedCompileError([f"Site {site.name!r} requires scalar coordinate laws"])
    if isinstance(prior, (dist.ExpandedDistribution, dist.MaskedDistribution)):
        validate_site_prior(site, prior.base_dist)
        return
    if isinstance(prior, dist.MixtureGeneral):
        for component in prior.component_distributions:
            validate_site_prior(site, component)
        return
    lower, upper = distribution_support_bounds(prior)
    if site.support == SupportClass.POSITIVE and np.any(np.asarray(lower) < 0.0):
        raise AggregatedCompileError(
            [f"Prior for positive site {site.name!r} has non-positive support"]
        )
    if site.support == SupportClass.CORRELATION and (
        np.any(np.asarray(lower) < -1.0) or np.any(np.asarray(upper) > 1.0)
    ):
        raise AggregatedCompileError(
            [f"Correlation prior for {site.name!r} requires support within [-1, 1]"]
        )


def resolve_site_priors(
    sites: Sequence[SiteDescriptor],
    priors: Mapping[str, dist.Distribution] | None = None,
) -> dict[str, dist.Distribution]:
    """Apply explicit overrides to scientific defaults and broadcast to site shapes."""
    supplied = {} if priors is None else priors
    unknown = set(supplied) - {site.name for site in sites}
    if unknown:
        raise AggregatedCompileError([f"Priors refer to inactive sample sites: {sorted(unknown)}"])
    result = {}
    for site in sites:
        prior = supplied[site.name] if site.name in supplied else default_prior_for_descriptor(site)
        validate_site_prior(site, prior)
        result[site.name] = prior.expand(site.shape)
    return result
