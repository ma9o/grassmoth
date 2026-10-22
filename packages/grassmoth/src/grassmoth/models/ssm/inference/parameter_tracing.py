"""Shared utilities for inference backends.

Particle inference and LAPS initialization share NumPyro's prior replay and
transformations. Application metadata supports initialization proposals and
reporting; it never reconstructs the prior density or deterministic model sites.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.reparameterization import Strategy
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel

import functools

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers

from grassmoth.compute.programs import ProgramCompiler, compile_jit
from grassmoth.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES
from grassmoth.models.ssm.inference.parameter_transform import (
    ParameterTransform,
    prepare_parameter_transform,
)
from grassmoth.models.ssm.inference.prior_program import numpyro_model
from grassmoth.models.ssm.inference.sample_projection import (
    _filter_public_samples,
    _trace_public_sites,
)


@dataclass(frozen=True)
class SiteInfoEntry:
    """Typed trace metadata for one unconstrained NumPyro sample site."""

    shape: tuple[int, ...]
    distribution: dist.Distribution
    transform: dist.transforms.Transform
    value: jnp.ndarray


type SiteInfo = Mapping[str, SiteInfoEntry]


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    rng_key: jax.Array,
    reparam: Strategy | None = None,
    *,
    compiler: ProgramCompiler = compile_jit,
) -> SiteInfo:
    """Trace model once to discover sample sites (names, shapes, transforms).

    Site discovery is structural: it only needs the latent sample/deterministic
    sites emitted by ``numpyro_model``. Discovery replays the model with a zero
    likelihood; the particle target owns the exact joint likelihood used by
    LAPS and sampling.
    """
    model_fn = functools.partial(numpyro_model, panel, priors, likelihood_backend=None)
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)

    def draw_sites(key: jax.Array) -> dict[str, tuple[dist.Distribution, jax.Array]]:
        with handlers.seed(rng_seed=key):
            trace = handlers.trace(model_fn).get_trace()
        return {
            name: (site["fn"], site["value"])
            for name, site in trace.items()
            if site["type"] == "sample"
            and not site.get("is_observed", False)
            and name not in INTERNAL_DIAGNOSTIC_SITES
        }

    return MappingProxyType(
        {
            name: SiteInfoEntry(
                shape=value.shape,
                distribution=distribution,
                transform=dist.transforms.biject_to(distribution.support),
                value=value,
            )
            for name, (distribution, value) in compiler(draw_sites, rng_key).call(rng_key).items()
        }
    )


# ---------------------------------------------------------------------------
# Pure-JAX deterministic site assembly
# ---------------------------------------------------------------------------


def prepare_model_parameters(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    trace_key: jax.Array,
    reparam: Strategy | None,
    *,
    compiler: ProgramCompiler = compile_jit,
) -> tuple[ParameterTransform, SiteInfo, frozenset[str]]:
    """Use NumPyro's prior replay and transformations for every inference path."""
    prior_model = functools.partial(numpyro_model, panel, priors, likelihood_backend=None)
    site_info = _discover_sites(priors, panel, trace_key, reparam=reparam, compiler=compiler)
    public_sites = _trace_public_sites(prior_model) if reparam is not None else frozenset(site_info)
    if reparam is not None:
        prior_model = handlers.reparam(prior_model, config=reparam)
    parameters = prepare_parameter_transform(
        prior_model,
        trace_key,
        model_args=(),
        initial_values={name: info.value for name, info in site_info.items()},
    )
    return parameters, site_info, public_sites


def extract_constrained_samples(
    particles: jnp.ndarray,
    parameters: ParameterTransform,
    public_sites: Set[str],
) -> dict[str, jnp.ndarray]:
    """Replay the prior program and retain its authored parameter/deterministic sites."""
    constrained: dict[str, jnp.ndarray] = jax.jit(
        lambda positions: _filter_public_samples(
            jax.vmap(parameters.constrain)(positions), public_sites
        )
    )(particles)
    return constrained
