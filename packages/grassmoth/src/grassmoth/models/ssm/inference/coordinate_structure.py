"""Compile time-constant translation geometry from model and prior structure.

The structural chart recognizes a fixed nonzero loading and an intercept whose
NumPyro replay is affine in the actual sampler coordinates. It differentiates
that proved affine replay once, rather than assuming a physical parameter is an
unconstrained coordinate. Learned sparsity follows reading and drift ownership;
prior transforms propagate those dependencies into their sampling coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import transforms
from numpyro.infer.reparam import LocScaleReparam, ProjectedNormalReparam, TransformReparam

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.compute.programs import ProgramCompiler
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.inference.parameter_tracing import SiteInfo
    from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
    from grassmoth.models.ssm.inference.reparameterization import Strategy
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle


@dataclass(frozen=True)
class CompiledTranslationStructure:
    """Fixed structural slopes and permitted learned state/parameter pairs."""

    initial_slopes: jax.Array
    active_pairs: jax.Array


@dataclass(frozen=True)
class _SiteCoordinates:
    """Sampler dependencies of each scalar in one canonical scientific site."""

    columns: tuple[tuple[int, ...], ...]
    affine: bool


def _affine_transform(transform: transforms.Transform) -> bool:
    match transform:
        case transforms.AffineTransform() | transforms.IdentityTransform():
            return True
        case transforms.ComposeTransform(parts=parts):
            return all(_affine_transform(part) for part in parts)
        case transforms.IndependentTransform(base_transform=base):
            return _affine_transform(base)
        case _:
            return False


def _coupled_dimensions(transform: transforms.Transform) -> int:
    """Event axes joined by a transform; independent wrapping joins no axes."""
    match transform:
        case transforms.AffineTransform() | transforms.IdentityTransform():
            return 0
        case transforms.ComposeTransform(parts=parts):
            return max((_coupled_dimensions(part) for part in parts), default=0)
        case transforms.IndependentTransform(base_transform=base):
            return _coupled_dimensions(base)
        case _:
            return int(max(transform.domain.event_dim, transform.codomain.event_dim))


def _unwrapped(distribution: dist.Distribution) -> dist.Distribution:
    while isinstance(
        distribution, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
    ):
        distribution = distribution.base_dist
    return distribution


def _site_coordinates(
    name: str,
    distribution: dist.Distribution,
    source_indices: Mapping[str, jax.Array],
    site_info: SiteInfo,
    reparam: Strategy | None,
) -> _SiteCoordinates:
    """Follow the same native reparameterizers used by prior replay.

    Elementwise transforms keep coordinate ownership even when their derivative
    happens to vanish numerically. An event transform may couple its event, but
    never unrelated batch members or another independently sampled site.
    """
    output_shape = distribution.batch_shape + distribution.event_shape
    affine = True
    coupled = 0
    while reparam is not None and (choice := reparam.config[name]) is not None:
        distribution = _unwrapped(distribution)
        if isinstance(choice, LocScaleReparam):
            if choice.centered == 1.0:
                break
            name = f"{name}_decentered"
        elif isinstance(choice, TransformReparam):
            assert isinstance(distribution, dist.TransformedDistribution)
            affine = affine and all(_affine_transform(t) for t in distribution.transforms)
            coupled = max(
                coupled,
                max((_coupled_dimensions(t) for t in distribution.transforms), default=0),
            )
            distribution = distribution.base_dist
            name = f"{name}_base"
        elif isinstance(choice, ProjectedNormalReparam):
            affine = False
            coupled = max(coupled, 1)
            name = f"{name}_normal"
        else:
            raise TypeError(f"Unsupported prior reparameterizer: {type(choice).__name__}")
    transform = site_info[name].transform
    affine = affine and _affine_transform(transform)
    coupled = max(coupled, _coupled_dimensions(transform))
    indices = tuple(int(value) for value in np.asarray(source_indices[name]).reshape(-1))
    if coupled == 0:
        assert len(indices) == prod(output_shape)
        columns = tuple((index,) for index in indices)
    else:
        groups = prod(output_shape[:-coupled])
        output_width = prod(output_shape[-coupled:])
        source_width = len(indices) // groups
        columns = tuple(
            indices[group * source_width : (group + 1) * source_width]
            for group in range(groups)
            for _ in range(output_width)
        )
    return _SiteCoordinates(columns, affine)


def compile_translation_structure(
    model: CompiledDynamicalModel,
    priors: PriorRuntimeBundle,
    parameters: ParameterTransform,
    *,
    site_info: SiteInfo,
    reparam: Strategy | None,
    compiler: ProgramCompiler,
) -> CompiledTranslationStructure:
    """Compile structural shifts and sparse learned regressions in sampler coordinates.

    The first eligible reading in compiled observation order anchors a stochastic
    state. Fixed zero or sampled loadings, nonlinear intercept transforms and
    deterministic states do not define a structural shift. Their absence is a
    structural absence of this pattern, not a replacement approximate target.
    """
    size = int(parameters.initial_position.size)
    source_indices = parameters.unravel(jnp.arange(size))
    sites = {
        name: _site_coordinates(name, prior, source_indices, site_info, reparam)
        for name, prior in priors.priors.items()
    }
    dynamic = frozenset(model.state_space.transition) & frozenset(model.state_space.initial)
    loadings = np.asarray(model.loading_block.template)
    free_loadings = model.loading_block.free_support
    reading_support = (loadings != 0.0) | free_loadings
    reading_sites = (
        *(
            (
                int(state),
                model.observation_mean_block.free_site_name,
                index,
            )
            for index, channel in enumerate(model.observation_mean_block.free_positions)
            for state in np.flatnonzero(reading_support[channel])
        ),
        *(
            (state, model.loading_block.free_site_name, index)
            for index, (_, state) in enumerate(model.loading_block.free_positions)
        ),
    )
    drift_sites = tuple(
        (component.target, site.name, 0)
        for index, component in enumerate(model.dynamics.spec.components)
        for _, site in component.parameter_sites(f"vf_{index}")
    )
    dependencies = frozenset(
        (state, column)
        for state, name, index in (*reading_sites, *drift_sites)
        if state in dynamic
        for column in sites[name].columns[index]
    )
    active_pairs = jnp.asarray(
        tuple(
            tuple((state, column) in dependencies for column in range(size))
            for state in range(model.state_space.size)
        ),
        dtype=jnp.bool_,
    )
    intercepts = model.observation_mean_block
    eligible = tuple(
        (channel, model.observations[channel].state_index)
        for channel in intercepts.free_positions
        if sites[intercepts.free_site_name].affine
        and model.observations[channel].state_index in dynamic
        and not free_loadings[channel, model.observations[channel].state_index]
        and loadings[channel, model.observations[channel].state_index] != 0.0
    )
    anchors = tuple(
        next(((channel, state) for channel, owner in eligible if owner == state), None)
        for state in range(model.state_space.size)
    )
    if not eligible:
        return CompiledTranslationStructure(
            jnp.zeros((model.state_space.size, size), dtype=parameters.initial_position.dtype),
            active_pairs,
        )

    def intercept_derivatives(position: jax.Array) -> jax.Array:
        return jnp.asarray(
            jax.jacfwd(lambda eta: parameters.constrain(eta)["manifest_means"])(position)
        )

    jacobian = compiler(intercept_derivatives, parameters.initial_position).call(
        parameters.initial_position
    )
    slopes = jnp.stack(
        tuple(
            jnp.zeros((size,), dtype=parameters.initial_position.dtype)
            if anchor is None
            else -jacobian[anchor[0]] / loadings[anchor]
            for anchor in anchors
        )
    )
    return CompiledTranslationStructure(slopes, active_pairs)
