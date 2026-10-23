"""Particle fits reuse JAX executables across nonstructural settings changes."""

import logging
from datetime import UTC, datetime
from time import monotonic

import jax
import numpy as np
import pytest
from jax._src import cache_key, compiler

from grassmoth.artifacts.sampling import (
    DsmcSpec,
    FixedLatentStepSpec,
    GaussianInitializationSpec,
)
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth_server.compute.compilation_cache import (
    compile_program,
    local_compilation_cache,
    prepare_program,
)
from grassmoth_test_support.compute import COMPUTE_BUDGET
from grassmoth_test_support.interval_particle import interval_target

pytestmark = [pytest.mark.inference(concern="sampling"), pytest.mark.inference(concern="warmup")]


def test_particle_fit_settings_reuse_jax_executable_and_preserve_transitions(tmp_path, monkeypatch):
    target = interval_target(
        np.array([0.0, 1.0]), np.array([[0.2], [0.3]]), np.array([[0.0], [1.0]]), ("last",)
    )
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(91),
        num_chains=2,
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_program,
    )
    assert isinstance(initialization, ParticleInitialization)
    active = False
    keys = []
    compilations = []
    get_key = cache_key.get
    backend_compile = compiler.backend_compile_and_load

    def observe_key(*args, **kwargs):
        key = get_key(*args, **kwargs)
        if active:
            keys.append(key)
        return key

    def observe_compile(*args, **kwargs):
        if active:
            compilations.append(True)
        return backend_compile(*args, **kwargs)

    def prepare(function, *args, **kwargs):
        compile_function, operands, shapes, memory = prepare_program(function, *args, **kwargs)
        if function.__name__ != "block":
            return compile_function, operands, shapes, memory

        def compile_sweep():
            nonlocal active
            active = True
            try:
                return compile_function()
            finally:
                active = False

        return compile_sweep, operands, shapes, memory

    monkeypatch.setattr(cache_key, "get", observe_key)
    monkeypatch.setattr(compiler, "backend_compile_and_load", observe_compile)
    results = []
    for diagnostic, fixed, warmup, samples, scale in (
        (False, False, 4, 2, 0.02),
        (True, True, 5, 4, 0.03),
        (True, False, 4, 2, 0.02),
    ):
        jax.clear_caches()
        kernel = build_marginal_particle_gibbs_kernel(
            target,
            num_particles=4,
            num_parameter_particles=2,
            param_step_size=scale,
            parameter_preconditioner_chol=initialization.parameter_chol,
            diagnostic_metrics_all=diagnostic,
            paid_mix=PaidMixture(
                initialization.pilot_mean,
                initialization.pilot_proposal_variance,
                scalar_operand(value=fixed),
            ),
            dsmc=DsmcSpec(latent_step=FixedLatentStepSpec()) if fixed else DsmcSpec(),
        )
        with local_compilation_cache(tmp_path, commit=lambda: None):
            results.append(
                run_marginal_particle_gibbs(
                    target,
                    compiler=compile_program,
                    prepare_program=prepare,
                    kernel=kernel,
                    num_warmup=warmup,
                    num_samples=samples,
                    num_chains=2,
                    key=jax.random.key(123),
                    adaptation_rate=0.05,
                    initialization=initialization,
                    compute_budget=COMPUTE_BUDGET,
                    clock=monotonic,
                    wall_clock=lambda: datetime.now(UTC),
                    logger=logging.getLogger(__name__),
                )
            )
    assert len(keys) == 3
    assert len(set(keys)) == 1
    assert len(compilations) == 1
    np.testing.assert_array_equal(results[0].grouped_positions, results[2].grouped_positions)
    np.testing.assert_array_equal(results[0].last_paths, results[2].last_paths)
    assert "parameter_jump_rms" not in results[0].chain_extra_fields
    assert "parameter_jump_rms" in results[2].chain_extra_fields
