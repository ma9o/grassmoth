"""Compile state diffusion, initial conditions, and dynamics components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.model.parameter import SiteKind, SupportClass
from grassmoth.models.ssm.compile.layout.axes import (
    input_mask,
    n_states,
    state_ids,
    time_invariant_mask,
)
from grassmoth.models.ssm.compile.layout.coefficients import _quantity_values
from grassmoth.models.ssm.dynamics.spec import DynamicsSpec
from grassmoth.models.ssm.structure.assembly import dense_vector_positions
from grassmoth.models.ssm.structure.blocks import (
    DiffusionBlockSpec,
    SparseBlockSpec,
    T0CholBlockSpec,
)

if TYPE_CHECKING:
    from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
    from grassmoth.models.structure import StructuralSelection


def diffusion_block(selection: StructuralSelection) -> DiffusionBlockSpec:
    """Compile diffusion Cholesky support, including induced correlations and zero static rows."""
    count = n_states(selection)
    support = np.eye(count, dtype=bool)
    axis = {identity: index for index, identity in enumerate(state_ids(selection))}
    for first_id, second_id, kind in selection.induced_dependencies:
        if kind == "innovation_correlation":
            first, second = axis[first_id], axis[second_id]
            support[max(first, second), min(first, second)] = True
    static = time_invariant_mask(selection) | input_mask(selection)
    support[static, :] = False
    support[:, static] = False
    template, support = _quantity_values(
        selection, SiteKind.DIFFUSION_DIAG, np.eye(count), np.zeros_like(support), diagonal=True
    )
    template, support = _quantity_values(selection, SiteKind.DIFFUSION_LOWER, template, support)
    template[static, :] = 0.0
    template[:, static] = 0.0
    support[static, :] = False
    support[:, static] = False
    return DiffusionBlockSpec(
        n_latent=count,
        diffusion_chol_support=support,
        diffusion_chol_template=jnp.asarray(template),
    )


def initial_mean_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile fixed and free initial-state mean coordinates in state-axis order."""
    support = np.zeros(n_states(selection), dtype=bool)
    template, support = _quantity_values(
        selection, SiteKind.T0_MEANS, np.zeros(n_states(selection)), support
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(dense_vector_positions(support, n_states(selection))),
        free_site_name="t0_means_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        prior_field="t0_means",
    )


def initial_covariance_block(selection: StructuralSelection) -> T0CholBlockSpec:
    """Compile initial-state scales and correlations into a Cholesky template with fixed inputs."""
    count = n_states(selection)
    support = np.zeros(count, dtype=bool)
    std, support = _quantity_values(selection, SiteKind.T0_VAR_DIAG, np.ones(count), support)
    std[input_mask(selection)] = 0.0
    correlations, correlation_support = _quantity_values(
        selection, SiteKind.T0_VAR_LOWER, np.eye(count), np.zeros((count, count), dtype=bool)
    )
    lower = np.tril(np.asarray(correlations), -1)
    corr = np.eye(count) + lower + lower.T
    template = jnp.asarray(np.asarray(std)[:, None] * np.linalg.cholesky(corr))
    return T0CholBlockSpec(
        n_latent=count,
        diag_support=support,
        correlation_support=correlation_support,
        template=jnp.asarray(template),
    )


def dynamics_expressions(selection: StructuralSelection) -> tuple[ExpressionComponentSpec, ...]:
    """Lower authored dynamics mechanisms into executable symbolic component specifications."""
    from grassmoth.models.ssm.compile.mechanisms import lower_mechanisms

    return lower_mechanisms(selection)


def dynamics_components(selection: StructuralSelection) -> DynamicsSpec:
    """Pair lowered dynamics components with their shared compiled state dimension."""
    return DynamicsSpec(n_latent=n_states(selection), components=dynamics_expressions(selection))
