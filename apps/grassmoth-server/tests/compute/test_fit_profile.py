"""A real profile uses native compiled parts without advancing the sampled chain."""

import json
from functools import partial

import numpy as np
import pytest

from grassmoth.artifacts.sampling import (
    DsmcSpec,
    GaussianInitializationSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    ParticleAGradSpec,
    SamplerSpec,
)
from grassmoth_server.actions.fit.action import run_fit
from grassmoth_server_test_support.fit_cache import ORIGIN, fixture

pytestmark = [pytest.mark.inference(concern="sampling"), pytest.mark.inference(concern="warmup")]


def test_profile_times_native_parts_and_preserves_the_chain(tmp_path):
    question, selection, data = fixture()
    sampler = SamplerSpec(
        hardware="local",
        num_chains=2,
        num_warmup=2,
        num_samples_per_chain=2,
        algorithm=MarginalParticleGibbsSpec(
            initialization=GaussianInitializationSpec(),
            path_update=ParticleAGradSpec(num_particles=4),
            companion="centred_nuts",
        ),
    )
    execute = partial(
        run_fit,
        question=question,
        selection=selection,
        data_for_model=data,
        time_origin=ORIGIN,
        sampler=sampler,
        compute_loo_diagnostics=False,
    )
    plain = execute()
    directory = tmp_path / "profile"
    profiled = execute(profile=True, profile_dir=directory)
    report = profiled.profile
    assert plain.profile is None
    assert report is not None
    assert report.phase == "after_warmup"
    assert report.repetitions == 10
    assert report.compile_seconds > 0
    assert report.device
    assert "cpu" in report.device[0]
    assert {row.component for row in report.components} == {
        "full_sweep",
        "parameter_move",
        "path_smoother",
        "nuts_companion",
        "conditional_gradient",
    }
    for row in report.components:
        assert 0 < row.timing.min_ms <= row.timing.median_ms <= row.timing.max_ms
    assert report.split_total is not None
    assert report.nuts is not None
    assert report.nuts.gradient_count_representative
    assert len(report.nuts.leapfrog_steps) == 2
    assert report.nuts.leapfrog_steps[0] > 0
    assert report.nuts.step_size[0] > 0
    assert report.nuts.tree_depth[0] > 0
    assert report.particle_dimensions is not None
    assert report.particle_dimensions.particles == 4
    assert report.particle_dimensions.labels == 2
    assert report.chains == report.particle_dimensions.replicas == 2
    assert report.time_steps == 3
    assert json.loads((directory / "summary.json").read_text()) == report.model_dump(mode="json")
    assert not list(directory.rglob("*.trace.json.gz"))
    entries = (*report.compilation, *report.profiling_compilation)
    names = tuple(entry.name for entry in entries)
    assert len(names) == len(set(names))
    assert {
        "production_sweep",
        "observation_projection",
        "particle_chain_initializer",
        "companion_initializer",
    } <= {entry.name for entry in report.compilation}
    assert {
        "profile_parameter_move",
        "profile_smoother_setup",
        "profile_path_smoother",
        "profile_companion",
        "profile_conditional_gradient",
    } == {entry.name for entry in report.profiling_compilation}
    assert all(not entry.name.startswith("profile_") for entry in report.compilation)
    assert all(entry.seconds > 0 and entry.persistent_cache != "unobserved" for entry in entries)
    assert report.compile_seconds <= sum(entry.seconds for entry in report.compilation)
    assert {path.name.removesuffix(".hlo.txt") for path in directory.glob("*.hlo.txt")} == set(
        names
    )
    for name, value in plain.inference.evidence.chain_extra_fields.items():
        np.testing.assert_array_equal(
            value.values, profiled.inference.evidence.chain_extra_fields[name].values
        )
    before = plain.inference.evidence.checkpoint
    after = profiled.inference.evidence.checkpoint
    assert before is not None
    assert after is not None
    np.testing.assert_array_equal(before.random_key.values, after.random_key.values)
    np.testing.assert_array_equal(before.parameters.values, after.parameters.values)
    np.testing.assert_array_equal(before.paths.values, after.paths.values)

    continued = execute(
        sampler=sampler.revised(num_warmup=0, num_samples_per_chain=1),
        profile=True,
        previous=profiled,
        continuation=after,
    )
    assert continued.profile is not None
    assert continued.profile.phase == "after_first_sweep"
    assert continued.profile.nuts is not None
    assert not continued.profile.nuts.gradient_count_representative
    assert continued.profile.repetitions == 10


@pytest.mark.parametrize("draws", [0, 1])
@pytest.mark.parametrize(
    "algorithm",
    [
        MarginalParticleGibbsSpec(
            initialization=GaussianInitializationSpec(), path_update=DsmcSpec(num_particles=4)
        ),
        NutsDASpec(trajectory=NutsTrajectorySpec(max_tree_depth=2)),
    ],
)
def test_profile_compile_only_and_without_companion(algorithm, draws):
    question, selection, data = fixture()
    result = run_fit(
        question=question,
        selection=selection,
        data_for_model=data,
        time_origin=ORIGIN,
        sampler=SamplerSpec(
            hardware="local",
            num_chains=1,
            num_warmup=0,
            num_samples_per_chain=draws,
            algorithm=algorithm,
        ),
        compute_loo_diagnostics=False,
        profile=True,
    )
    report = result.profile
    assert report is not None
    assert report.nuts is None
    assert report.phase == ("after_first_sweep" if draws else "compile_only")
    assert report.repetitions == (10 if draws else 0)
    expected = {"full_sweep"}
    if isinstance(algorithm, MarginalParticleGibbsSpec):
        expected |= {"parameter_move", "path_smoother"}
    assert {row.component for row in report.components} == (expected if draws else set())
