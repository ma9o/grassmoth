import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.reports.posterior_diagnostics import NutsDADiagnostics
from grassmoth.artifacts.sampling import (
    FixedTrajectorySpec,
    NutsDASpec,
    NutsTrajectorySpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.samplers.nuts import NutsDARun, run_nuts_da
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_test_support.compute import EXECUTION_CONTEXT, compile_program
from grassmoth_test_support.inference import (
    nonlinear_fit_fixture,
)
from grassmoth_test_support.interval_particle import gaussian_interval_case

"""Joint NUTS recovery and production posterior contracts."""


pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize(
    ("parameterization", "trajectory"),
    [
        ("centred", NutsTrajectorySpec()),
        ("non_centred", FixedTrajectorySpec()),
        ("mixed", NutsTrajectorySpec()),
    ],
)
def test_parallel_joint_nuts_recovers_analytic_parameter_state_covariance(
    trajectory, parameterization
):
    target, mean, covariance = gaussian_interval_case()
    # Fixed-length HMC has more autocorrelated squared coordinates; retain enough
    # effective draws to check covariance with the same tolerance as NUTS.
    num_samples = 3000 if isinstance(trajectory, FixedTrajectorySpec) else 750
    run = run_nuts_da(
        target,
        sampler=SamplerSpec(
            algorithm=NutsDASpec(trajectory=trajectory, parameterization=parameterization),
            num_warmup=250,
            num_samples_per_chain=num_samples,
            num_chains=4,
        ),
        start=jax.random.key(904),
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(run, NutsDARun)
    assert run.paths is not None
    assert run.positions.shape[:2] == run.paths.shape[:2] == (4, num_samples)
    joint = np.concatenate((run.positions, run.paths[:, :, :, 0]), axis=-1).reshape(-1, mean.size)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), (joint - mean).T).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=0.12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(mean.size), atol=0.18)
    assert not np.array_equal(run.positions[0], run.positions[1])
    extra = run.phase_extra_fields["post_warmup"]
    assert extra["diverging"].shape == (4, num_samples)
    assert float(jnp.mean(extra["diverging"])) < 0.02
    assert run.phase_extra_fields["warmup"]["energy"].shape == (4, 250)
    # Adaptation has ended before any retained draw.
    np.testing.assert_allclose(
        extra["step_size"], np.broadcast_to(run.final_step_size[:, None], (4, num_samples))
    )
    if isinstance(trajectory, NutsTrajectorySpec):
        np.testing.assert_array_equal(extra["tree_depth"], np.ceil(np.log2(extra["num_steps"] + 1)))
    else:
        np.testing.assert_array_equal(extra["num_steps"], 8)
    for chain in range(4):
        context = target.context(run.positions[chain, 0], target.times)
        expected = target.observation_log_probs(context, run.paths[chain, 0], target.observations)
        np.testing.assert_allclose(run.observation_log_probs[chain, 0], expected, rtol=2e-6)


def test_nuts_fit_preserves_nonlinear_model_and_shared_posterior():
    prepared, panel = nonlinear_fit_fixture()
    result = fit(
        prepared.prior_runtime_bundle,
        panel,
        sampler=SamplerSpec(
            algorithm=NutsDASpec(trajectory=NutsTrajectorySpec(max_tree_depth=3)),
            num_warmup=0,
            num_samples_per_chain=4,
            num_chains=2,
        ),
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(result, MCMCPosterior)
    assert result.evidence.method == "nuts_da"
    assert result.draws.describe().n_draws == 8
    assert all(values.shape[0] == 8 for values in result.get_samples().values())
    assert result.draws.latent_paths is not None
    assert result.draws.latent_paths.shape == (8, len(panel.times), 1)
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
    assert isinstance(result.diagnostics.sampler_diagnostics, NutsDADiagnostics)
    assert result.diagnostics.phase_extra_fields is not None
    assert result.diagnostics.phase_extra_fields["warmup"]["energy"].shape == (2, 0)


@pytest.mark.parametrize(
    ("parameterization", "trajectory"),
    [
        ("centred", NutsTrajectorySpec(max_tree_depth=3)),
        ("non_centred", FixedTrajectorySpec(steps=4)),
        ("mixed", NutsTrajectorySpec(max_tree_depth=3)),
    ],
)
def test_continuation_matches_uninterrupted_random_stream_and_tuning(
    trajectory, parameterization, monkeypatch
):
    target, _, _ = gaussian_interval_case()
    settings = SamplerSpec(
        algorithm=NutsDASpec(trajectory=trajectory, parameterization=parameterization),
        num_warmup=12,
        num_samples_per_chain=8,
        num_chains=2,
    )

    def run(options, continuation=None):
        return run_nuts_da(
            target,
            sampler=options,
            start=jax.random.key(17) if continuation is None else continuation,
            execution_context=EXECUTION_CONTEXT,
        )

    full = run(settings)
    first = run(settings.revised(num_samples_per_chain=4))
    assert isinstance(full, NutsDARun)
    assert isinstance(first, NutsDARun)
    second = run(settings.revised(num_warmup=0, num_samples_per_chain=4), first.checkpoint)
    assert isinstance(second, NutsDARun)
    np.testing.assert_allclose(
        full.positions,
        np.concatenate((first.positions, second.positions), axis=1),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(second.final_step_size, first.final_step_size)
    np.testing.assert_array_equal(
        second.checkpoint.random_key.values, full.checkpoint.random_key.values
    )

    from grassmoth.models.ssm.inference.samplers import nuts

    restore = nuts.restore_state

    def stale_scores(template, leaves):
        restored = restore(template, leaves)
        return restored._replace(
            potential_energy=jnp.full_like(restored.potential_energy, jnp.nan),
            z_grad=jnp.full_like(restored.z_grad, jnp.inf),
        )

    monkeypatch.setattr(nuts, "restore_state", stale_scores)
    refreshed = run(settings.revised(num_warmup=0, num_samples_per_chain=4), first.checkpoint)
    assert isinstance(refreshed, NutsDARun)
    np.testing.assert_array_equal(refreshed.positions, second.positions)
    for actual, expected in zip(
        refreshed.checkpoint.native_state, second.checkpoint.native_state, strict=True
    ):
        np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.parametrize("parameterization", ["centred", "non_centred", "mixed"])
def test_compile_only_executes_neither_sweep(monkeypatch, parameterization):
    from grassmoth.models.ssm.inference.samplers import sweeps

    target, _, _ = gaussian_interval_case()
    observed = []
    original = sweeps.run_sweeps

    def record(plan, initial, **kwargs):
        observed.append(plan)
        result = original(plan, initial, **kwargs)
        for before, after in zip(
            jax.tree.leaves(initial), jax.tree.leaves(result.state), strict=True
        ):
            np.testing.assert_array_equal(
                jax.random.key_data(before)
                if jax.dtypes.issubdtype(before.dtype, jax.dtypes.prng_key)
                else before,
                jax.random.key_data(after)
                if jax.dtypes.issubdtype(after.dtype, jax.dtypes.prng_key)
                else after,
            )
        return result

    from grassmoth.models.ssm.inference.samplers import nuts

    monkeypatch.setattr(nuts, "run_sweeps", record)
    run = run_nuts_da(
        target,
        sampler=SamplerSpec(
            algorithm=NutsDASpec(
                trajectory=FixedTrajectorySpec(steps=2), parameterization=parameterization
            ),
            num_warmup=0,
            num_samples_per_chain=0,
            num_chains=2,
        ),
        start=jax.random.key(18),
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(run, NutsDARun)
    assert run.positions.shape[:2] == (2, 0)
    from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates

    centred = build_joint_coordinates(target, compiler=compile_program)
    parameters, paths = centred.decode(jnp.zeros(centred.dimension))
    np.testing.assert_allclose(
        run.checkpoint.parameters.values,
        np.broadcast_to(parameters, (2, *parameters.shape)),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        run.checkpoint.paths.values, np.broadcast_to(paths, (2, *paths.shape)), atol=2e-6
    )
    from grassmoth.artifacts.continuation import JointTuning

    assert isinstance(run.checkpoint.tuning, JointTuning)
    assert run.checkpoint.tuning.parameterization.kind == parameterization

    from grassmoth.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase

    assert isinstance(observed[0].warmup, CompiledSweepPhase)
    assert isinstance(observed[0].posterior, CompiledSweepPhase)
    assert observed[0].projection is not None
