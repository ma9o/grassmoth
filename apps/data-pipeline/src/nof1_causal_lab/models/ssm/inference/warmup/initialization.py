"""Owned starts and proposal geometry for exact particle samplers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.continuation import ParticleTuning, SamplerCheckpoint
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import (
    ContinuedInitializationDiagnostics,
    GaussianInitializationDiagnostics,
)
from nof1_causal_lab.artifacts.sampling import GaussianInitializationSpec
from nof1_causal_lab.compute.batching import compiled_memory
from nof1_causal_lab.compute.programs import ProgramCompiler, compile_jit
from nof1_causal_lab.models.ssm.inference.joint_coordinates import (
    CentredParameterization,
    JointChart,
    JointCoordinates,
    build_joint_coordinates,
)
from nof1_causal_lab.models.ssm.inference.warmup.laps_ensemble import (
    LapsTraceRecord,
    NoLapsTrace,
    run_laps_ensemble,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import InitializationDiagnostics
    from nof1_causal_lab.artifacts.sampling import (
        LapsInitializationSpec,
        ParticleInitializationSpec,
    )
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True)
class LapsInitializationTrace:
    """Native ensemble history together with the exact parameterization that produced it."""

    coordinates: JointCoordinates
    ensemble: LapsTraceRecord


@dataclass(frozen=True)
class ParticleInitialization:
    """Paired chain starts and frozen proposal moments in the production coordinates."""

    positions: jax.Array
    paths: jax.Array
    parameter_chol: jax.Array
    pilot_mean: jax.Array
    pilot_variance: jax.Array
    diagnostics: InitializationDiagnostics
    trace: LapsInitializationTrace | NoLapsTrace = field(default_factory=NoLapsTrace)

    @property
    def pilot_proposal_variance(self) -> jax.Array:
        """Fixed path proposal spread across ensemble members and observation times."""
        temporal_variance = jnp.var(self.pilot_mean, axis=0)
        variance_floor = 1e-6 * (1.0 + temporal_variance)
        return self.pilot_variance + temporal_variance[None, :] + variance_floor[None, :]


def prepare_particle_initialization(
    target: ParticleTarget,
    *,
    spec: ParticleInitializationSpec,
    key: jax.Array,
    num_chains: int,
    compute_budget: ComputeBudget,
    compiler: ProgramCompiler = compile_jit,
) -> ParticleInitialization | InitializationFailure:
    """Select explicit starting geometry without changing the retained target."""
    if isinstance(spec, GaussianInitializationSpec):
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
            raise MemoryError(
                "Gaussian initialization exceeds the numerical execution memory budget"
            )
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
    return prepare_laps_initialization(
        target,
        spec=spec,
        key=key,
        num_chains=num_chains,
        compute_budget=compute_budget,
        compiler=compiler,
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


def prepare_laps_initialization(
    target: ParticleTarget,
    *,
    spec: LapsInitializationSpec,
    key: jax.Array,
    num_chains: int,
    compute_budget: ComputeBudget,
    record_trace: bool = False,
    compiler: ProgramCompiler = compile_jit,
) -> ParticleInitialization | InitializationFailure:
    """Produce one initialization package without executing a particle smoother."""
    if num_chains > spec.ensemble_size:
        return InitializationFailure.unavailable(
            "LAPS ensemble_size must cover all requested particle chains"
        )
    coordinates = build_joint_coordinates(target, "centred", compiler=compiler)
    assert isinstance(coordinates.parameterization, CentredParameterization)
    if not all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            coordinates.parameter_center,
            coordinates.parameterization.path_center,
            coordinates.parameterization.path_chol,
        )
    ):
        return InitializationFailure.unavailable(
            "The native starting point has no finite LAPS coordinate scale"
        )
    ensemble_key, selection_key = jax.random.split(key)
    ensemble = run_laps_ensemble(
        coordinates.log_density,
        dimension=max(2, coordinates.dimension),
        scientific_dimension=coordinates.dimension,
        key=ensemble_key,
        spec=spec,
        compute_budget=compute_budget,
        record_trace=record_trace,
    )
    if isinstance(ensemble, InitializationFailure):
        return ensemble
    positions, paths = compiler(jax.vmap(coordinates.decode), ensemble.positions).call(
        ensemble.positions
    )
    centered = positions - jnp.mean(positions, axis=0)
    covariance = centered.T @ centered / (spec.ensemble_size - 1)
    covariance = covariance + spec.covariance_regularization * jnp.eye(
        coordinates.parameter_dimension, dtype=positions.dtype
    )
    chol = jnp.linalg.cholesky(covariance)
    if not np.isfinite(np.asarray(chol)).all() or not np.isfinite(np.asarray(paths)).all():
        return InitializationFailure.unavailable(
            "The LAPS ensemble has non-finite paths or parameter covariance"
        )
    selected = jax.random.choice(
        selection_key, spec.ensemble_size, shape=(num_chains,), replace=False
    )
    return ParticleInitialization(
        positions=positions[selected],
        paths=paths[selected],
        parameter_chol=chol,
        pilot_mean=jnp.mean(paths, axis=0),
        pilot_variance=jnp.var(paths, axis=0),
        diagnostics=ensemble.diagnostics,
        trace=LapsInitializationTrace(coordinates, ensemble.trace)
        if isinstance(ensemble.trace, LapsTraceRecord)
        else ensemble.trace,
    )
