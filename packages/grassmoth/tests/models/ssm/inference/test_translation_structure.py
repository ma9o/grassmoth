"""Structural chart recognition respects real prior reparameterization and graph ownership."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.sampling import MarginalParticleGibbsSpec, SamplerSpec
from grassmoth.models.ssm.inference.coordinate_structure import compile_translation_structure
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.reparameterization import AutoReparam
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.inference import bind_panel_fixture, compile_fit_fixture
from grassmoth_test_support.model import one_state_gaussian_model


@pytest.mark.contract
def test_chart_settings_are_opt_in_and_part_of_continuation_identity():
    ordinary = SamplerSpec(algorithm=MarginalParticleGibbsSpec())
    assert isinstance(ordinary.algorithm, MarginalParticleGibbsSpec)
    assert not ordinary.algorithm.compiler_reparameterization
    assert not ordinary.algorithm.learned_translation
    identities = {ordinary.kernel_identity}
    for structural, learned in ((True, False), (False, True), (True, True)):
        sampler = SamplerSpec(
            algorithm=MarginalParticleGibbsSpec(
                compiler_reparameterization=structural, learned_translation=learned
            )
        )
        assert SamplerSpec.model_validate_json(sampler.model_dump_json()) == sampler
        identities.add(sampler.kernel_identity)
    assert len(identities) == 4


@pytest.mark.inference(concern="warmup")
@pytest.mark.parametrize(
    ("prior", "loading", "expected_slope"),
    [
        ("Normal", 1.0, -2.0),
        ("Normal", 2.0, -1.0),
        ("LogNormal", 1.0, 0.0),
        ("Normal", "sampled", 0.0),
    ],
)
def test_structural_chart_compiles_native_scale_and_sparse_graph(prior, loading, expected_slope):
    level_id, loading_id = "parameter:" + "a" * 64, "parameter:" + "b" * 64
    document = one_state_gaussian_model().model_dump(mode="json")
    construct = next(iter(document["constructs"].values()))
    if loading == "sampled":
        reference = deepcopy(next(iter(construct["indicators"].values())))
        reference["observation"]["name"] = "chart_reference"
        reference["likelihood"]["law"]["scale"]["value"] = 1.0
        construct["indicators"]["indicator:chart-reference"] = reference
    law = next(iter(construct["indicators"].values()))["likelihood"]["law"]
    law["loc"]["left"]["value"] = level_id
    law["loc"]["right"]["left"]["value"] = loading_id if loading == "sampled" else loading
    document["parameters"][level_id] = {
        "name": "level",
        "description": "Observation location used to test actual prior coordinates",
        "distribution": "distribution:chart-level",
    }
    document["distributions"]["distribution:chart-level"] = {
        "distribution": prior,
        "params": {"loc": 0.0, "scale": 2.0},
    }
    if loading == "sampled":
        document["parameters"][loading_id] = {
            "name": "loading",
            "description": "A sampled loading cannot define a fixed structural slope",
            "distribution": "distribution:chart-loading",
        }
        document["distributions"]["distribution:chart-loading"] = {
            "distribution": "Normal",
            "params": {"loc": 1.0, "scale": 0.1},
        }
    model = compile_fit_fixture(DynamicalModelSpec.model_validate(document).materialized())
    reparam = AutoReparam(centered=0.0)
    problem = build_particle_problem(
        model.prior_runtime_bundle,
        bind_panel_fixture(
            model.compiled_dynamical_model,
            jnp.zeros((3, len(model.compiled_dynamical_model.observations))),
            jnp.arange(3.0),
        ),
        trace_key=jax.random.key(0),
        reparam=reparam,
        compiler=compile_program,
    )
    structure = compile_translation_structure(
        model.compiled_dynamical_model,
        model.prior_runtime_bundle,
        problem.runtime.parameters,
        site_info=problem.site_info,
        reparam=reparam,
        compiler=compile_program,
    )
    indices = problem.runtime.parameters.unravel(jnp.arange(problem.runtime.initial_position.size))
    intercept_name = model.compiled_dynamical_model.observation_mean_block.free_site_name
    intercept_columns = next(
        value for name, value in indices.items() if name.startswith(intercept_name)
    )
    expected = np.zeros_like(structure.initial_slopes)
    expected[0, np.asarray(intercept_columns).reshape(-1)] = expected_slope
    np.testing.assert_allclose(structure.initial_slopes, expected, atol=1e-6)
    expected_pairs = np.zeros_like(structure.active_pairs)
    loading_name = model.compiled_dynamical_model.loading_block.free_site_name
    for name, columns in indices.items():
        if name.startswith((intercept_name, loading_name, "vf_")):
            expected_pairs[0, np.asarray(columns).reshape(-1)] = True
    np.testing.assert_array_equal(structure.active_pairs, expected_pairs)
