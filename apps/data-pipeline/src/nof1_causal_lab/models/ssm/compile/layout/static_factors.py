"""Compile marginalized baseline factor identities, loadings, and scales."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.model.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.model.parameter_context import coefficient_value
from nof1_causal_lab.models.ssm.compile.layout.axes import state_names
from nof1_causal_lab.models.ssm.compile.layout.coefficients import _quantity_values
from nof1_causal_lab.models.ssm.compile.layout.errors import NumericalSupportError
from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions
from nof1_causal_lab.models.ssm.structure.blocks import SparseBlockSpec
from nof1_causal_lab.models.structure import selected_state_ids

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.models.structure import StructuralSelection


def _build_static_factor_structure(
    selection: StructuralSelection, latent_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Derive baseline-factor incidence and fixed loadings from explicit common causes."""
    from nof1_causal_lab.artifacts.model.expressions import linear_coefficient
    from nof1_causal_lab.models.parameters import baseline_factor_groups

    dynamical_model_spec = selection.dynamical_model_spec
    groups = baseline_factor_groups(selection)
    factors = [group[0] for group in groups]
    state_index = {identity: index for index, identity in enumerate(selected_state_ids(selection))}
    loadings = np.zeros((len(latent_names), len(factors)))
    for index, group in enumerate(groups):
        sources = {construct.id for construct in group}
        for source in sources:
            construct = dynamical_model_spec.get_construct(source)
            initial_mean = construct.coefficient("initial_mean")
            assert initial_mean is not None
            if coefficient_value(initial_mean) != 0.0:
                raise NumericalSupportError(
                    ["Marginalized baseline factors require a fixed zero initial mean"]
                )
            if (
                construct.indicators
                or construct.temporal_status != "time_invariant"
                or any(edge.effect.id == source for edge in dynamical_model_spec.edges)
            ):
                raise NumericalSupportError(
                    ["A marginalized baseline factor must be an unmeasured time-invariant root"]
                )
            for edge in dynamical_model_spec.edges:
                if edge.cause.id != source or edge.effect.id not in state_index:
                    continue
                weight = 1.0
                if edge.mechanisms:
                    if len(edge.mechanisms) != 1:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    coefficient = coefficient_value(
                        linear_coefficient(edge.mechanisms[0].expression, source)
                    )
                    if coefficient is None:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    weight = coefficient
                loadings[state_index[edge.effect.id], index] = weight
        if not np.any(loadings[:, index]):
            raise NumericalSupportError(["A baseline factor must affect a retained state"])
    return (
        np.ones(len(factors), dtype=bool),
        np.zeros(len(factors)),
        np.asarray(loadings),
        tuple(factor.name for factor in factors),
    )


def static_factor_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """Return the representative construct identity for each compiled baseline-factor group."""
    from nof1_causal_lab.models.parameters import baseline_factor_groups

    return tuple(group[0].id for group in baseline_factor_groups(selection))


def static_factor_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Label baseline factors from their initial-scale parameters or owning constructs."""
    dynamical_model_spec = selection.dynamical_model_spec
    names = []
    for identity in static_factor_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        coefficient = construct.coefficient("initial_scale")
        assert coefficient is not None
        names.append(
            dynamical_model_spec.parameter(coefficient).name
            if isinstance(coefficient, str)
            else construct.name
        )
    return tuple(names)


def static_factor_loadings(selection: StructuralSelection) -> jnp.ndarray:
    """Build the state-by-factor loadings used to represent marginalized static effects."""
    return jnp.asarray(_build_static_factor_structure(selection, state_names(selection))[2])


def static_scale_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile positive baseline-factor scales and their free parameter positions."""
    n = len(static_factor_ids(selection))
    values, support = _quantity_values(
        selection, SiteKind.STATIC_STATE_SD, np.zeros(n), np.ones(n, dtype=bool)
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(values),
        free_positions=tuple(dense_vector_positions(support, len(values))),
        free_site_name="static_state_sd_free",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        prior_field="static_state_sd",
    )
