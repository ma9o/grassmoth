"""Posterior sample extraction shared by exact joint MCMC methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from grassmoth.models.ssm.inference.parameter_tracing import extract_constrained_samples

if TYPE_CHECKING:
    from collections.abc import Mapping

    import jax.numpy as jnp

    from grassmoth.compute.programs import ProgramCompiler
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.inference.problem import ParticleProblem


def build_pmcmc_mcmc_result(
    *,
    chain_samples: dict[str, jnp.ndarray],
    chain_extra_fields: Mapping[str, jnp.ndarray],
    num_chains: int,
    num_samples: int,
    backend: str,
) -> TrajectoryMCMCResult:
    """Project retained particle samples into the native posterior representation."""
    return TrajectoryMCMCResult(
        chain_samples=chain_samples,
        chain_extra_fields=chain_extra_fields,
        num_chains=num_chains,
        num_samples=num_samples,
        backend=backend,
    )


def extract_grouped_public_samples(
    grouped_positions: jnp.ndarray,
    *,
    bundle: ParticleProblem,
    num_chains: int,
    num_samples: int,
    compiler: ProgramCompiler,
) -> dict[str, jnp.ndarray]:
    """Project retained particle samples into the native posterior representation."""
    flat_positions = grouped_positions.reshape(
        (num_chains * num_samples, bundle.runtime.initial_position.size)
    )
    public_samples = extract_constrained_samples(
        flat_positions, bundle.runtime.parameters, bundle.public_sites, compiler=compiler
    )
    return {
        name: values.reshape((num_chains, num_samples, *values.shape[1:]))
        for name, values in public_samples.items()
    }


def project_initialization(
    values: Mapping[str, jnp.ndarray],
    *,
    bundle: ParticleProblem,
    compiled_dynamical_model: CompiledDynamicalModel,
    compiler: ProgramCompiler,
) -> dict[str, jnp.ndarray]:
    """Retain delivered starts in the same scientific coordinates as posterior laws."""
    samples = extract_constrained_samples(
        values["positions"], bundle.runtime.parameters, bundle.public_sites, compiler=compiler
    )
    return {
        **values,
        "time_points": bundle.runtime.times,
        **{
            str(identity): samples[coordinate.site_name][(slice(None), *coordinate.indices)]
            for binding in compiled_dynamical_model.bindings
            for identity, coordinate in binding.coordinates.items()
        },
        **{
            str(state.id): values["paths"][:, :, index]
            for index, state in enumerate(compiled_dynamical_model.states)
        },
    }
