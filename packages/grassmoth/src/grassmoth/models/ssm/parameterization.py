"""Compiled scientific site metadata and prior-predictive parameter assembly.

The registry derives authored sample-site shapes and bindings from DynamicalModelSpec
without tracing. NumPyro distributions preserve native parameter pytrees and
stable per-site random streams. Particle inference and initialization use
NumPyro replay through Dynestyx for reparameterized sites, transformations, and
the prior density; this module does not define a second inference target.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.models.ssm.execution.parameters import assemble_model_matrices
from grassmoth.models.ssm.priors import resolve_site_priors

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpyro.distributions as dist

    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.structure.sites import SiteDescriptor


@dataclass(frozen=True)
class PriorRuntimeBundle:
    """Reusable runtime components derived from compiled prior semantics."""

    registry: tuple[SiteDescriptor, ...]
    priors: Mapping[str, dist.Distribution]

    def __post_init__(self) -> None:
        """Own immutable collections binding prior distributions to compiled execution sites."""
        freeze_fields(self)


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_site_registry(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[SiteDescriptor, ...]:
    """Return the site registry already owned by the compiled model."""
    return compiled_dynamical_model.site_registry


def likelihood_sites(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[SiteDescriptor, ...]:
    """Select sampling-site descriptors belonging to likelihood-specific auxiliary coefficients."""
    return tuple(
        site
        for site in compiled_dynamical_model.site_registry
        if site.assembly_group == "likelihood"
    )


def process_sites(compiled_dynamical_model: CompiledDynamicalModel) -> tuple[SiteDescriptor, ...]:
    """Select sampling-site descriptors belonging to process-innovation coefficients."""
    return tuple(
        site for site in compiled_dynamical_model.site_registry if site.assembly_group == "process"
    )


# ---------------------------------------------------------------------------
# Selection and assembly of authored parameter values
# ---------------------------------------------------------------------------


def _resolve_num_draws(
    samples: dict[str, jnp.ndarray],
    n_draws: int | None,
) -> int:
    if n_draws is not None:
        return n_draws
    if samples:
        return int(next(iter(samples.values())).shape[0])
    raise ValueError("n_draws is required when assembling deterministic values without samples")


def assemble_deterministics_from_registry(
    samples: dict[str, jnp.ndarray],
    compiled_dynamical_model: CompiledDynamicalModel,
    *,
    n_draws: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Batch the same scientific matrix assembly used by NumPyro inference."""
    n_draws = _resolve_num_draws(samples, n_draws)

    def assemble_draw(index: jax.Array) -> dict[str, jax.Array]:
        return assemble_model_matrices(
            compiled_dynamical_model, {name: value[index] for name, value in samples.items()}
        )[0]

    return jax.vmap(assemble_draw)(jnp.arange(n_draws))


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


def build_prior_runtime_bundle(
    compiled_dynamical_model: CompiledDynamicalModel,
    priors: Mapping[str, dist.Distribution] | None = None,
) -> PriorRuntimeBundle:
    """Resolve the scientific site declarations to native NumPyro laws."""
    registry = build_site_registry(compiled_dynamical_model)
    return PriorRuntimeBundle(
        registry=tuple(registry), priors=MappingProxyType(resolve_site_priors(registry, priors))
    )
