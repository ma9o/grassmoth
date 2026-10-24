"""Compile observation loadings, noise, anchors, and likelihood sites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.model.expressions import COEFFICIENT_MEANINGS
from grassmoth.artifacts.model.likelihood import OBSERVATION_FAMILY_SPECS, CategoricalLawSpec
from grassmoth.artifacts.model.parameter import SiteKind, SupportClass
from grassmoth.artifacts.model.semantics import indicator_requires_observation_intercept
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.parameters import execution_parameters
from grassmoth.models.ssm.compile.layout.axes import (
    diffusion_families,
    n_observations,
    n_states,
    observation_families,
    observation_level_counts,
    observed_indicators,
)
from grassmoth.models.ssm.compile.layout.coefficients import _quantity_values
from grassmoth.models.ssm.compile.layout.errors import NumericalSupportError
from grassmoth.models.ssm.structure.assembly import (
    dense_vector_positions,
    rect_matrix_positions,
)
from grassmoth.models.ssm.structure.blocks import ManifestCholBlockSpec, SparseBlockSpec
from grassmoth.models.ssm.structure.sites import SiteDescriptor
from grassmoth.models.structure import get_reference_indicator_lookup, selected_state_ids

if TYPE_CHECKING:
    from grassmoth.models.structure import StructuralSelection


def categorical_anchors(selection: StructuralSelection) -> tuple[bool, ...]:
    """Pin the reference contrast only for states with exclusively nominal emissions."""
    dynamical_model_spec = selection.dynamical_model_spec
    references = get_reference_indicator_lookup(selection)
    categorical_states = {
        identity
        for identity in selected_state_ids(selection)
        if all(
            indicator.likelihood is not None
            and indicator.likelihood.law.family == DistributionFamily.CATEGORICAL
            for indicator in dynamical_model_spec.get_construct(identity).indicators
        )
    }
    return tuple(
        dynamical_model_spec.indicator_owner(indicator.observation.id).id in categorical_states
        and indicator.observation.name
        == references[dynamical_model_spec.indicator_owner(indicator.observation.id).name]
        for indicator in observed_indicators(selection)
    )


def loading_block(selection: StructuralSelection) -> SparseBlockSpec[tuple[int, int]]:
    """Compile the observation-by-state loading template and its free coefficient positions."""
    shape = (n_observations(selection), n_states(selection))
    template, support = _quantity_values(
        selection, SiteKind.LOADING, np.zeros(shape), np.zeros(shape, dtype=bool)
    )
    return SparseBlockSpec[tuple[int, int]](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(rect_matrix_positions(support, *shape)),
        free_site_name="lambda_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.LOADING,
        assembly_group="lambda",
        prior_field="lambda_free",
    )


def observation_mean_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile active observation intercepts, rejecting free intercepts unused by their laws."""
    inactive = [
        indicator.observation.name
        for indicator in observed_indicators(selection)
        if indicator.likelihood is not None
        and isinstance(indicator.likelihood.parsed.intercept.value, str)
        and not indicator_requires_observation_intercept(
            indicator.likelihood.law.family,
            indicator.likelihood.parsed.link,
            standardized=indicator.likelihood.standardized,
        )
    ]
    if inactive:
        raise NumericalSupportError(
            [
                f"Observation intercept 'manifest_mean_{name}' is inactive for the locked likelihood "
                "semantics; fix the channel location through its compiler-owned anchor instead."
                for name in inactive
            ]
        )
    template, support = _quantity_values(
        selection,
        SiteKind.MANIFEST_MEANS,
        np.zeros(n_observations(selection)),
        np.zeros(n_observations(selection), dtype=bool),
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(dense_vector_positions(support, n_observations(selection))),
        free_site_name="manifest_means_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.MANIFEST_MEANS,
        assembly_group="manifest",
        prior_field="manifest_means",
    )


def observation_noise_block(selection: StructuralSelection) -> ManifestCholBlockSpec:
    """Compile the observation-noise template and free diagonal scale positions."""
    n = n_observations(selection)
    template, support = _quantity_values(
        selection,
        SiteKind.MANIFEST_VAR_DIAG,
        np.zeros((n, n)),
        np.zeros(n, dtype=bool),
        diagonal=True,
    )
    return ManifestCholBlockSpec(
        n_manifest=n_observations(selection), diag_support=support, template=jnp.asarray(template)
    )


def likelihood_sites(selection: StructuralSelection) -> tuple[SiteDescriptor, ...]:
    """Declare emission/process hyperparameters in their native sampling order."""
    sites = []
    families = set(observation_families(selection))
    for law in OBSERVATION_FAMILY_SPECS:
        if len(law.parameter_roles) != 1:
            continue
        meaning = COEFFICIENT_MEANINGS[law.parameter_roles[0]]
        family, kind = law.family, meaning.quantity
        name = kind.value
        if family in families:
            sites.append(
                SiteDescriptor(
                    name=name,
                    shape=(),
                    support=meaning.support,
                    assembly_group="likelihood",
                    site_kind=kind,
                    prior_field=name,
                )
            )
    n = n_observations(selection)
    cutpoints = max(max(observation_level_counts(selection), default=0) - 1, 0)
    if DistributionFamily.ORDERED_LOGISTIC in families and cutpoints:
        base = COEFFICIENT_MEANINGS["cutpoint_base"]
        sites.append(
            SiteDescriptor(
                name=base.quantity.value,
                shape=(n,),
                support=base.support,
                assembly_group="likelihood",
                site_kind=base.quantity,
                prior_field=base.quantity.value,
            )
        )
        if cutpoints > 1:
            gaps = COEFFICIENT_MEANINGS["cutpoint_gaps"]
            sites.append(
                SiteDescriptor(
                    name=gaps.quantity.value,
                    shape=(n, cutpoints - 1),
                    support=gaps.support,
                    assembly_group="likelihood",
                    site_kind=gaps.quantity,
                    prior_field=gaps.quantity.value,
                )
            )
    if DistributionFamily.CATEGORICAL in families and cutpoints:
        for role in CategoricalLawSpec.parameter_roles:
            meaning = COEFFICIENT_MEANINGS[role]
            kind = meaning.quantity
            name = kind.value
            sites.append(
                SiteDescriptor(
                    name=name,
                    shape=(n, cutpoints),
                    support=meaning.support,
                    assembly_group="likelihood",
                    site_kind=kind,
                    prior_field=name,
                )
            )
    if DistributionFamily.STUDENT_T in diffusion_families(selection):
        meaning = COEFFICIENT_MEANINGS["process_degrees_of_freedom"]
        sites.append(
            SiteDescriptor(
                name=meaning.quantity.value,
                shape=(),
                support=meaning.support,
                assembly_group="process",
                site_kind=meaning.quantity,
                prior_field=meaning.quantity.value,
            )
        )
    sampled_quantities = {
        selection.dynamical_model_spec.parameter_context(parameter.id).quantity
        for parameter in execution_parameters(selection)
    }
    return tuple(site for site in sites if site.site_kind in sampled_quantities)
