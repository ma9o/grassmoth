"""Owned starts and proposal geometry for exact particle samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.continuation import ParticleTuning, SamplerCheckpoint
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ContinuedInitializationDiagnostics,
    GaussianInitializationDiagnostics,
)
from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.programs import ProgramCompiler, compile_jit
from grassmoth.models.ssm.inference.joint_coordinates import (
    CentredParameterization,
    JointChart,
    build_joint_coordinates,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from grassmoth.artifacts.reports.posterior_diagnostics import InitializationDiagnostics
    from grassmoth.artifacts.sampling import GaussianInitializationSpec
    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True)
class ParticleInitialization:
    """Paired chain starts and frozen proposal moments in the production coordinates."""

    positions: jax.Array
    paths: jax.Array
    parameter_chol: jax.Array
    pilot_mean: jax.Array
    pilot_variance: jax.Array
    diagnostics: InitializationDiagnostics

    @property
    def pilot_proposal_variance(self) -> jax.Array:
        """Fixed path proposal spread across chain starts and observation times."""
        temporal_variance = jnp.var(self.pilot_mean, axis=0)
        variance_floor = 1e-6 * (1.0 + temporal_variance)
        return self.pilot_variance + temporal_variance[None, :] + variance_floor[None, :]


def prepare_particle_initialization(
    target: ParticleTarget,
    *,
    spec: GaussianInitializationSpec,
    key: jax.Array,
    num_chains: int,
    compute_budget: ComputeBudget,
    compiler: ProgramCompiler = compile_jit,
) -> ParticleInitialization | InitializationFailure:
    """Select explicit starting geometry without changing the retained target."""
    coordinates = build_joint_coordinates(target, "centred", compiler=compiler)
    assert isinstance(coordinates.parameterization, CentredParameterization)

    def draw(
        chart: JointChart, draw_key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        points = spec.scale * jax.random.normal(
            draw_key, (num_chains, max(1, coordinates.dimension))
        )
        positions, paths = jax.vmap(coordinates.bind(chart).decode)(points)
        return (
            positions,
            paths,
            spec.scale * jnp.eye(coordinates.parameter_dimension, dtype=positions.dtype),
            jnp.mean(paths, axis=0),
            jnp.var(paths, axis=0),
        )

    program = compiler(draw, coordinates.chart, key)
    if not compute_budget.capacity(compiled_memory(program.executable), limit=1):
        raise MemoryError("Gaussian initialization exceeds the numerical execution memory budget")
    positions, paths, parameter_chol, pilot_mean, pilot_variance = program.call(
        coordinates.chart, key
    )
    if not all(np.isfinite(np.asarray(value)).all() for value in (positions, paths)):
        return InitializationFailure.unavailable(
            "The Gaussian initializer produced non-finite starting coordinates"
        )
    return ParticleInitialization(
        positions=positions,
        paths=paths,
        parameter_chol=parameter_chol,
        pilot_mean=pilot_mean,
        pilot_variance=pilot_variance,
        diagnostics=GaussianInitializationDiagnostics(
            joint_dimension=coordinates.dimension, scale=spec.scale
        ),
    )


def continued_particle_initialization(checkpoint: SamplerCheckpoint) -> ParticleInitialization:
    """Start at retained values, using the particle kernel's saved proposal geometry."""
    tuning = checkpoint.tuning
    if not isinstance(tuning, ParticleTuning):
        positions = jnp.asarray(checkpoint.parameters.values)
        paths = jnp.asarray(checkpoint.paths.values)
        return ParticleInitialization(
            positions=positions,
            paths=paths,
            parameter_chol=jnp.eye(positions.shape[-1], dtype=positions.dtype),
            pilot_mean=jnp.mean(paths, axis=0),
            pilot_variance=jnp.var(paths, axis=0),
            diagnostics=ContinuedInitializationDiagnostics(reused_tuning=False),
        )
    return ParticleInitialization(
        positions=jnp.asarray(checkpoint.parameters.values),
        paths=jnp.asarray(checkpoint.paths.values),
        parameter_chol=jnp.asarray(tuning.parameter_chol.values),
        pilot_mean=jnp.asarray(tuning.pilot_mean.values),
        pilot_variance=jnp.asarray(tuning.pilot_variance.values),
        diagnostics=ContinuedInitializationDiagnostics(reused_tuning=True),
    )
