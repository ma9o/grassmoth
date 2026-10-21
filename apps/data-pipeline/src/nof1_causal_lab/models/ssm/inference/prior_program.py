"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Time-series trajectories
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.dynamics.intervention import (
    Intervention,
    PrecomputedValueFn,
    VariableOverride,
)
from nof1_causal_lab.models.ssm.execution.dynamical_model import (
    continuous_state_evolution,
    initial_state_distribution,
)
from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices, sample_sites

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.execution.contracts import InitializationLikelihoodBackend
    from nof1_causal_lab.models.ssm.preparation.binding import BoundPanel

from nof1_causal_lab.models.ssm.compile.observations import materialize_observation_laws
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance import INITIAL_STATE_COV_MIN_EIGENVALUE
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    likelihood_sites,
    process_sites,
)


@jax.custom_vjp
def _nan_safe_ll(ll):
    """Return ll if finite, else -1e30. Gradient is zeroed when ll is non-finite."""
    return jnp.where(jnp.isfinite(ll), ll, -1e30)


def _nan_safe_ll_fwd(ll):
    y = _nan_safe_ll(ll)
    return y, jnp.isfinite(ll)


def _nan_safe_ll_bwd(is_finite, g):
    return (jnp.where(is_finite, jnp.nan_to_num(g, nan=0.0), 0.0),)


_nan_safe_ll.defvjp(_nan_safe_ll_fwd, _nan_safe_ll_bwd)


def sample_parameters(
    compiled_dynamical_model: CompiledDynamicalModel, priors: PriorRuntimeBundle
) -> dict[str, jnp.ndarray]:
    """Sample declared sites and emit the canonical scientific matrices."""
    sites = chain.from_iterable(
        block.iter_sites() for block in numeric.parameter_blocks(compiled_dynamical_model)
    )
    matrices, min_eigenvalue = assemble_model_matrices(
        compiled_dynamical_model, sample_sites(sites, priors.priors.__getitem__)
    )
    for name, value in matrices.items():
        # Empty input/static-factor blocks have no public deterministic site.
        if name != "static_state_sds" or value.size:
            numpyro.deterministic(name, value)
    numpyro.factor(
        "t0_correlation_positive_definite",
        jnp.where(
            min_eigenvalue > INITIAL_STATE_COV_MIN_EIGENVALUE,
            0.0,
            -1e6 * (INITIAL_STATE_COV_MIN_EIGENVALUE - min_eigenvalue),
        ),
    )
    return matrices


def initialization_input_intervention(panel: BoundPanel, times: jnp.ndarray) -> Intervention:
    """Use the bound input path when tracing structural parameters."""
    return Intervention(
        tuple(
            VariableOverride(
                int(index),
                PrecomputedValueFn(times - times[0], panel.input_values[:, index]),
            )
            for index in np.flatnonzero(numeric.input_mask(panel.compiled_dynamical_model))
        )
    )


def numpyro_model(
    panel: BoundPanel,
    priors: PriorRuntimeBundle,
    likelihood_backend: InitializationLikelihoodBackend | None,
) -> None:
    """Replay declared priors, constructing dynamics only for an initialization likelihood."""
    compiled_dynamical_model = panel.compiled_dynamical_model
    observations, times = panel.observations, panel.times
    sampled = sample_parameters(compiled_dynamical_model, priors)

    diffusion_chol = sampled["diffusion"]
    lambda_mat = sampled["lambda"]
    manifest_means = sampled["manifest_means"]
    t0_means = sampled["t0_means"]

    manifest_cov = sampled["manifest_cov"]
    t0_cov = sampled["t0_cov"]
    sample_sites(process_sites(compiled_dynamical_model), priors.priors.__getitem__)
    observation_sites = sample_sites(
        likelihood_sites(compiled_dynamical_model), priors.priors.__getitem__
    )
    dynamics_params = compiled_dynamical_model.dynamics.sample_params(priors.priors.__getitem__)
    if likelihood_backend is None:
        numpyro.factor("log_likelihood", jnp.array(0.0))
        return
    observation_laws = materialize_observation_laws(compiled_dynamical_model, observation_sites)
    dynamics = continuous_state_evolution(
        vector_field=compiled_dynamical_model.dynamics.vector_field,
        vf_params=dynamics_params,
        diffusion=diffusion_chol,
        intervention=initialization_input_intervention(panel, times),
    )

    meas_params = MeasurementParams(
        lambda_mat=lambda_mat,
        manifest_means=manifest_means,
        manifest_cov=manifest_cov,
    )

    time_intervals = jnp.diff(times, prepend=times[0])
    time_intervals = time_intervals.at[0].set(MIN_DT)

    init = initial_state_distribution(
        compiled_dynamical_model, t0_means, t0_cov, input_values=panel.input_values
    )
    lnc = likelihood_backend.compute_log_likelihood(
        dynamics,
        meas_params,
        init,
        observations,
        time_intervals,
        observation_laws=observation_laws,
    )

    # lnc is (T,) cumulative log-normalizing constants from the filter.
    # lnc[-1] = total log p(y|θ).
    # diff(lnc) exposes per-timestep contributions to the initialization
    # objective. Reported LOO uses emission factors on joint particle draws.
    if lnc.ndim == 0:
        total_ll = _nan_safe_ll(lnc)
        numpyro.factor("log_likelihood", total_ll)
    else:
        total_ll = _nan_safe_ll(lnc[-1])
        numpyro.factor("log_likelihood", total_ll)
        ll_per_timestep = jnp.diff(lnc, prepend=0.0)
        numpyro.deterministic("ll_per_timestep", ll_per_timestep)
