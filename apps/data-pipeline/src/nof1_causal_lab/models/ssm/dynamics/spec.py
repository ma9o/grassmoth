"""Bind scientific expressions, parameter sites, and node potentials for Dynestyx."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro

from nof1_causal_lab.models.ssm.compile.expression_programs import compile_shared_drift
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpyro.distributions as dist
    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    type PriorFn = Callable[[str], dist.Distribution]


@dataclass(frozen=True, eq=False)
class DynamicsSpec:
    """Declarative SSM dynamics: latent dimension + tuple of component specs."""

    n_latent: int
    components: tuple[ExpressionComponentSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, eq=False)
class CompiledDynamics:
    """A vector field with the parameter sampler and site registry compiled for it."""

    spec: DynamicsSpec
    vector_field: VectorField
    sample_params: Callable[[PriorFn], tuple[dict[str, Array], ...]]
    site_registry: tuple[SiteDescriptor, ...]


def compile_dynamics(spec: DynamicsSpec, *, prefix: str = "vf") -> CompiledDynamics:
    """Compile symbolic dynamics into a vector field and matching parameter sampler.

    Args:
        spec: State dimension and ordered drift or potential component definitions.
        prefix: Sampling-site prefix distinguishing this vector field from other
            models or nested components.

    Returns:
        The vector field, its ordered sampling-site registry, and a NumPyro
        callable producing the matching component parameter tuple.
    """
    components = tuple(component_spec.build() for component_spec in spec.components)
    vector_field = VectorField(
        n_latent=spec.n_latent,
        components=components,
        drift_program=compile_shared_drift(spec),
        potential_indices=tuple(
            i for i, item in enumerate(spec.components) if item.kind == "potential"
        ),
    )
    component_sites = tuple(
        tuple(
            zip(
                (identity for identity, _ in component.parameters),
                component.iter_sites(prefix=f"{prefix}_{index}", n_latent=spec.n_latent),
                strict=True,
            )
        )
        for index, component in enumerate(spec.components)
    )
    site_registry = tuple(site for sites in component_sites for _, site in sites)

    def _sample_all_params(prior_fn: PriorFn) -> tuple[dict[str, Array], ...]:
        return tuple(
            {
                identity: jnp.asarray(numpyro.sample(site.name, prior_fn(site.name)))
                for identity, site in sites
            }
            for sites in component_sites
        )

    return CompiledDynamics(
        spec=spec,
        vector_field=vector_field,
        sample_params=_sample_all_params,
        site_registry=site_registry,
    )


def pack_component_params_from_samples(
    spec: DynamicsSpec,
    samples: Mapping[str, Array],
    *,
    prefix: str = "vf",
) -> tuple[dict[str, Array], ...]:
    """Pack flat sample-site values into the vector-field param tuple.

    This mirrors ``CompiledDynamics.sample_params`` without entering a NumPyro
    sampling context, so post-fit likelihood evaluators can rebuild the same
    runtime dynamics object from constrained parameter draws.
    """
    return tuple(
        component.pack_params(f"{prefix}_{index}", samples)
        for index, component in enumerate(spec.components)
    )
