"""Shared parameter sampling and matrix assembly for the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro

from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.covariance import symmetrize

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpyro.distributions as dist

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


def sample_sites(
    sites: Iterable[SiteDescriptor], prior: Callable[[str], dist.Distribution]
) -> dict[str, jnp.ndarray]:
    """Interpret the declared site order through NumPyro's sampling handlers."""
    return {site.name: jnp.asarray(numpyro.sample(site.name, prior(site.name))) for site in sites}


def assemble_model_matrices(
    compiled_dynamical_model: CompiledDynamicalModel, samples: dict[str, jnp.ndarray]
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Assemble one parameter draw, including the initial covariance constraint.

    Every declared free site is required. Missing entries in ``values`` below
    therefore correspond only to fixed blocks, whose templates own their values.
    The returned minimum eigenvalue defines the NumPyro constraint factor.
    """
    values = {
        site.name: samples[site.name]
        for block in numeric.parameter_blocks(compiled_dynamical_model)
        for site in block.iter_sites()
    }
    manifest_chol = compiled_dynamical_model.observation_noise_block.assemble(
        values.get("manifest_var_diag_free")
    )
    static_sds = compiled_dynamical_model.static_scale_block.assemble(
        values.get("static_state_sd_free")
    )
    covariance = compiled_dynamical_model.initial_covariance_block.assemble_cov(
        values.get("t0_var_diag_free"), values.get("t0_var_lower_free")
    )
    if static_sds.size:
        loadings = jnp.asarray(compiled_dynamical_model.static_factor_loadings)
        covariance = covariance + loadings @ jnp.diag(static_sds**2) @ loadings.T
    endogenous = jnp.asarray(compiled_dynamical_model.state_space.initial, dtype=jnp.int32)
    endogenous_covariance = symmetrize(covariance[jnp.ix_(endogenous, endogenous)])
    min_eigenvalue = (
        jnp.min(jnp.linalg.eigvalsh(endogenous_covariance))
        if compiled_dynamical_model.state_space.initial
        else jnp.asarray(jnp.inf, dtype=covariance.dtype)
    )
    initial_covariance = (
        jnp.zeros_like(covariance).at[jnp.ix_(endogenous, endogenous)].set(endogenous_covariance)
    )
    return {
        "diffusion": compiled_dynamical_model.diffusion_block.assemble(
            values.get("diffusion_diag_free"), values.get("diffusion_lower_free")
        ),
        "lambda": compiled_dynamical_model.loading_block.assemble(values.get("lambda_free")),
        "manifest_means": compiled_dynamical_model.observation_mean_block.assemble(
            values.get("manifest_means_free")
        ),
        "manifest_cov": manifest_chol @ manifest_chol.T,
        "t0_means": compiled_dynamical_model.initial_mean_block.assemble(
            values.get("t0_means_free")
        ),
        "t0_cov": initial_covariance,
        "static_state_sds": static_sds,
    }, min_eigenvalue
