"""Owned starts and proposal geometry for exact particle samplers."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.continuation import ParticleTuning, SamplerCheckpoint
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ContinuedInitializationDiagnostics,
    GaussianInitializationDiagnostics,
    InitializationDiagnostics,
)
from grassmoth.artifacts.sampling import GaussianInitializationSpec
from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.context import ModelExecutionContext
from grassmoth.models.ssm.inference.joint_coordinates import (
    CentredParameterization,
    JointChart,
    build_joint_coordinates,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.warmup.ieks import ieks_path
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth.models.ssm.shapes import Array, Float, PRNGKeyArray


@dataclass(frozen=True)
class ParticleInitialization:
    """Paired chain starts and frozen proposal moments in the production coordinates."""

    positions: Float[Array, "B U"]
    paths: Float[Array, "B T D"]
    parameter_chol: Float[Array, "U U"]
    pilot_mean: Float[Array, "T D"]
    pilot_variance: Float[Array, "T D"]
    diagnostics: InitializationDiagnostics

    @property
    def pilot_proposal_variance(
        self,
    ) -> Float[Array, "{self.pilot_mean.shape[0]} {self.pilot_mean.shape[1]}"]:
        """Fixed path proposal spread across chain starts and observation times."""
        temporal_variance = jnp.var(self.pilot_mean, axis=0)
        variance_floor = 1e-6 * (1.0 + temporal_variance)
        return self.pilot_variance + temporal_variance[None, :] + variance_floor[None, :]


def prepare_particle_initialization(
    target: ParticleTarget,
    *,
    spec: GaussianInitializationSpec,
    key: PRNGKeyArray,
    num_chains: int,
    execution_context: ModelExecutionContext,
) -> ParticleInitialization | InitializationFailure:
    """Select explicit starting geometry without changing the retained target."""
    coordinates = build_joint_coordinates(target, "centred", compiler=execution_context.compiler)
    assert isinstance(coordinates.parameterization, CentredParameterization)

    def seeded_chart(chart: JointChart) -> JointChart:
        assert chart.affine is not None
        context = target.context(chart.parameter_center, target.times)
        center = ieks_path(target, context, chart.affine.path_center)
        return JointChart(
            chart.parameter_center, CentredParameterization(center, chart.affine.path_chol)
        )

    seed_program = execution_context.compiler(seeded_chart, coordinates.chart)
    if not execution_context.budget.capacity(compiled_memory(seed_program.executable), limit=1):
        raise MemoryError("IEKS initialization exceeds the numerical execution memory budget")
    chart = seed_program.call(coordinates.chart)
    assert chart.affine is not None
    if not np.isfinite(np.asarray(chart.affine.path_center)).all():
        return InitializationFailure.unavailable(
            "The IEKS initializer produced a non-finite path seed"
        )

    def gaussian_initialization(
        chart: JointChart, draw_key: PRNGKeyArray
    ) -> tuple[
        Float[Array, "B U"],
        Float[Array, "B T D"],
        Float[Array, "U U"],
        Float[Array, "T D"],
        Float[Array, "T D"],
    ]:
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

    program = execution_context.compiler(gaussian_initialization, chart, key)
    if not execution_context.budget.capacity(compiled_memory(program.executable), limit=1):
        raise MemoryError("Gaussian initialization exceeds the numerical execution memory budget")
    positions, paths, parameter_chol, pilot_mean, pilot_variance = program.call(chart, key)
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
        positions=jax.device_put(checkpoint.parameters.values),
        paths=jax.device_put(checkpoint.paths.values),
        parameter_chol=jax.device_put(tuning.parameter_chol.values),
        pilot_mean=jax.device_put(tuning.pilot_mean.values),
        pilot_variance=jax.device_put(tuning.pilot_variance.values),
        diagnostics=ContinuedInitializationDiagnostics(reused_tuning=True),
    )
