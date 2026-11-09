"""Analytical admission contracts, checked against owned buffer layouts."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from grassmoth.actions.fit.memory import _unconstrained_size, particle_mgrad_memory_shape
from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
from grassmoth.artifacts.sampling import (
    GuidedMethodSpec,
    InferenceHardware,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    PathUpdateSpec,
    SamplerSpec,
)
from grassmoth.compute.particle_memory import (
    ParticleMgradMemoryFailure,
    ParticleMgradMemoryReport,
    ParticleMgradMemoryShape,
    estimate_particle_mgrad_memory,
)
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    excluded_particle_diagnostics,
)
from grassmoth.models.ssm.inference.samplers.sweep_plan import SWEEP_BLOCK_SIZE
from grassmoth.models.structure import StructuralSelection
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.helpers import observation_dataset
from grassmoth_test_support.inference import bind_panel_fixture, compile_fit_fixture
from grassmoth_test_support.model import load_model_fixture
from grassmoth_test_support.values import revise_fixture

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = pytest.mark.contract


def shape():
    return ParticleMgradMemoryShape(
        time_points=9,
        state_dimensions=3,
        initial_dimensions=2,
        transition_dimensions=2,
        parameter_dimensions=4,
        observation_dimensions=2,
        free_path_coordinates=18,
        initial_covariance="diagonal",
        transition_covariance="diagonal",
        sweep_block_size=SWEEP_BLOCK_SIZE,
    )


def sampler(*, particles=64, labels=2, metrics=()):
    return SamplerSpec(
        num_chains=2,
        num_warmup=3,
        num_samples_per_chain=7,
        algorithm=MarginalParticleGibbsSpec(
            n_parameter_particles=labels,
            diagnostic_metrics=metrics,
            path_update=PathUpdateSpec(method=GuidedMethodSpec(num_particles=particles)),
        ),
    )


def report(settings):
    result = estimate_particle_mgrad_memory(shape(), settings)
    assert isinstance(result, ParticleMgradMemoryReport), result
    return result


@pytest.fixture(scope="module")
def native_target_storage():
    """Trace actual nonlinear projection and mixed-law noise factor owners."""
    model = compile_fit_fixture(
        load_model_fixture(
            "ecosystem_consumer/nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model.json"
        )
    )
    panel = bind_panel_fixture(
        model.compiled_dynamical_model,
        jnp.zeros((9, 2), dtype=jnp.float32),
        jnp.arange(9, dtype=jnp.float32),
    )
    target = build_particle_problem(
        model.prior_runtime_bundle,
        panel,
        trace_key=jax.random.key(7),
        reparam=None,
        compiler=compile_program,
    ).runtime
    context = target.context(target.initial_position, target.times)
    projection = jax.eval_shape(
        jax.vmap(lambda path: target.observation_log_probs(context, path, target.observations)),
        jax.ShapeDtypeStruct((2, 9, target.state_space.size), jnp.float32),
    )
    assert target.observations.shape[-1] == 2
    assert projection.shape == (2, 9)
    noise = jax.eval_shape(target.prepare_noise_factors, context, target.observations)
    return projection, noise.observations


def test_particle_growth_does_not_reintroduce_quadratic_pair_table():
    small, medium, large = (report(sampler(particles=n)) for n in (64, 128, 256))
    assert (
        medium.sampling.device_bytes - small.sampling.device_bytes
        == (large.sampling.device_bytes - medium.sampling.device_bytes) // 2
    )
    assert small.initialization == medium.initialization == large.initialization
    assert report(sampler(labels=4)).initialization == small.initialization
    assert not any("pair" in item.name for item in large.buffers)


@pytest.mark.parametrize(
    "metrics",
    [(), ("particle_weights", "particle_identity", "parameter_movement", "regional_movement")],
)
@pytest.mark.inference(concern="sampling")
def test_retention_matches_native_history_field_layout(metrics, native_target_storage):
    model, settings = shape(), sampler(metrics=metrics)
    c, t, d = settings.num_chains, model.time_points, model.state_dimensions
    fields: dict[str, NDArray[np.float32 | np.int32]] = {
        key: np.empty((c,), dtype=np.float32)
        for key in (
            "log_posterior",
            "parameter_accept_prob",
            "latent_accept_prob",
            "selected_parameter_label",
            "selected_particle",
            "latent_move_rms",
            "latent_move_max_abs",
            "latent_frozen_frac",
            "amala_grad_norm_mean",
            "amala_grad_norm_max",
            "reference_path_hit_rate",
            "selected_particle_unique_count",
            "parameter_jump_rms",
        )
    }
    fields.update(
        positions=np.empty((c, model.parameter_dimensions), dtype=np.float32),
        paths=np.empty((c, t, d), dtype=np.float32),
        latent_frozen_frac_by_d=np.empty((c, d), dtype=np.float32),
        final_label_log_probs=np.empty((c, 2), dtype=np.float32),
        latent_move_rms_per_t=np.empty((c, t), dtype=np.float32),
        selected_particle_per_t=np.empty((c, t), dtype=np.int32),
    )
    fields.update(
        {metric.history_key: np.empty((c, t), dtype=np.float32) for metric in ParticleWeightMetric}
    )
    excluded = excluded_particle_diagnostics(metrics)
    retained = {key: value for key, value in fields.items() if key not in excluded}
    warmup = {key: value for key, value in retained.items() if key not in {"positions", "paths"}}
    projection_shape, _ = native_target_storage
    projection = np.empty(projection_shape.shape, dtype=projection_shape.dtype)
    expected = settings.num_warmup * sum(value.nbytes for value in warmup.values())
    expected += settings.num_samples_per_chain * (
        sum(value.nbytes for value in retained.values()) + projection.nbytes
    )
    result = report(settings)
    history = next(item for item in result.buffers if item.name == "retained_histories")
    posterior = next(item for item in result.buffers if item.name == "posterior_device_history")
    assert history.demand.device_bytes + posterior.demand.device_bytes == expected
    assert history.demand.device_bytes == settings.num_warmup * sum(
        value.nbytes for value in warmup.values()
    )
    assert history.demand.host_bytes > expected


@pytest.mark.inference(concern="sampling")
def test_observation_factor_bound_covers_native_mixed_law_arrays(native_target_storage):
    _, factors = native_target_storage
    per_label = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(factors))
    settings = sampler()
    result = report(settings)
    storage = next(item for item in result.buffers if item.name == "observation_density_factors")
    assert isinstance(settings.algorithm, MarginalParticleGibbsSpec)
    assert storage.demand.device_bytes >= (
        settings.num_chains * settings.algorithm.n_parameter_particles * per_label
    )


def test_dense_and_retained_storage_are_distinct_from_particle_banks():
    base = report(sampler())
    dense = estimate_particle_mgrad_memory(
        revise_fixture(shape(), initial_covariance="dense", transition_covariance="dense"),
        sampler(),
    )
    assert isinstance(dense, ParticleMgradMemoryReport)
    assert dense.sampling.device_bytes > base.sampling.device_bytes
    longer = report(revise_fixture(sampler(), num_samples_per_chain=10000))
    assert longer.initialization == base.initialization
    assert next(x for x in longer.buffers if x.name == "particle_banks") == next(
        x for x in base.buffers if x.name == "particle_banks"
    )
    assert longer.sampling.device_bytes > base.sampling.device_bytes
    assert base.peak.device_bytes == max(
        base.initialization.device_bytes, base.sampling.device_bytes
    )


def test_stage_maximum_does_not_add_particle_workspace_to_final_upload():
    short = report(sampler())
    assert short.sampling.device_bytes < sum(
        item.demand.device_bytes for item in short.buffers if item.phase == "sampling"
    )
    # Once final device retention dominates, a larger particle bank changes the
    # earlier sweep peak but cannot enlarge the completed-history upload.
    long = report(revise_fixture(sampler(), num_samples_per_chain=10000))
    more_particles = report(revise_fixture(sampler(particles=128), num_samples_per_chain=10000))
    assert long.sampling.device_bytes == more_particles.sampling.device_bytes
    assert any(item.name == "concatenation_outputs" for item in short.buffers)


def test_inline_cuda_rng_is_only_used_for_proven_diagonal_remote_branch():
    local = revise_fixture(sampler(), hardware=InferenceHardware.LOCAL)
    remote = revise_fixture(local, hardware=InferenceHardware.L40S)
    cpu_report, cuda_report = report(local), report(remote)
    dense_report = estimate_particle_mgrad_memory(
        revise_fixture(shape(), transition_covariance="dense"), remote
    )
    assert isinstance(dense_report, ParticleMgradMemoryReport)

    def rng_bytes(result):
        return next(
            item.demand.device_bytes for item in result.buffers if item.name == "resampling_noise"
        )

    assert rng_bytes(cuda_report) < rng_bytes(cpu_report)
    assert rng_bytes(dense_report) == rng_bytes(cpu_report)


def test_other_sampler_has_typed_failure():
    result = estimate_particle_mgrad_memory(shape(), SamplerSpec(algorithm=NutsDASpec()))
    assert isinstance(result, ParticleMgradMemoryFailure)


def test_parameter_dimensions_follow_support_and_reparameterization_shapes():
    simplex = dist.Dirichlet(jnp.ones(3))
    projected = dist.ProjectedNormal(jnp.ones(3))
    transformed = dist.TransformedDistribution(
        dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1), dist.transforms.StickBreakingTransform()
    )
    assert _unconstrained_size(simplex, simplex.batch_shape + simplex.event_shape) == 2
    assert _unconstrained_size(projected, projected.batch_shape + projected.event_shape) == 3
    assert _unconstrained_size(transformed, transformed.batch_shape + transformed.event_shape) == 2


def test_metadata_uses_compiled_model_and_merged_observation_grid():
    model = load_model_fixture("fit_preflight/standardized_channel_model.json")
    compiled = compile_fit_fixture(model)
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "indicator_id": str(item.id),
            "value": 0.25,
            "anchor_time": origin + timedelta(days=day),
            "support_start": origin + timedelta(days=day),
            "support_end": origin + timedelta(days=day),
            "support_kind": item.observation.support_kind.value,
            "summary_operator": item.observation.summary_operator.value,
            "anchor_policy": item.observation.anchor_policy.value,
            "observation_window": str(item.observation.observation_window),
        }
        for item in compiled.compiled_dynamical_model.observations
        for day in (1, 3)
    ]
    observations = observation_dataset(
        pl.DataFrame(rows),
        tuple(item.observation for item in compiled.compiled_dynamical_model.observations),
        time_origin=origin,
    )
    result = particle_mgrad_memory_shape(StructuralSelection(model, None), observations)
    assert isinstance(result, ParticleMgradMemoryShape), result
    assert result.time_points == 3  # The grid owner inserts initial day zero.
    assert result.state_dimensions == compiled.compiled_dynamical_model.state_space.size
    assert result.observation_dimensions == len(compiled.compiled_dynamical_model.observations)
    assert result.sweep_block_size == SWEEP_BLOCK_SIZE
