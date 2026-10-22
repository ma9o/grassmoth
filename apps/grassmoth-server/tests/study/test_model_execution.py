"""DynamicalModelSpec identity, parameter draws, and native model execution."""

from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter

from grassmoth.actions.fit.conditioning import condition_model
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.likelihood import ObservationLawSpec
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)
from grassmoth.models.structure import StructuralSelection
from grassmoth_server_test_support.fixtures import study_fixture_json
from grassmoth_server_test_support.model import load_study_model_fixture
from grassmoth_test_support.helpers import make_model
from grassmoth_test_support.inference import compile_model_fixture, model_draws, particle_posterior


def _conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model() -> (
    DynamicalModelSpec
):
    return load_study_model_fixture(
        "model_spec_execution/conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model.json"
    )


def _model_model() -> DynamicalModelSpec:
    return load_study_model_fixture("model_spec_execution/model_model.json")


@pytest.fixture(scope="module")
def dynamical_model_spec():
    return _model_model()


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize("categorical", [False, True])
def test_conditioning_revises_the_same_type_and_retains_joint_uncertainty(
    dynamical_model_spec, tmp_path, monkeypatch, categorical
):

    from grassmoth.artifacts.model.likelihood import LikelihoodSpec
    from grassmoth.artifacts.probability.serialization import (
        empirical_atoms,
        empirical_distribution,
    )
    from grassmoth.models.inputs import input_fingerprints
    from grassmoth.models.ssm.compile.bindings import parameter_bindings
    from grassmoth.models.ssm.predictive.parameters import sample_model_laws
    from grassmoth_server.storage import paths as data_module
    from grassmoth_server.study.persistence.store import ArtifactStore, read_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    if categorical:
        dynamical_model_spec = make_model(["A", "B"], [("A", "B")])
        construct = dynamical_model_spec.constructs[1]
        indicator = construct.indicators[0].revised(
            observation=construct.indicators[0].observation.revised(
                measurement_dtype="categorical",
                categorical_levels=("low", "medium", "high"),
                aggregation="last",
            ),
            likelihood=LikelihoodSpec(
                law=TypeAdapter(ObservationLawSpec).validate_json(
                    study_fixture_json(
                        "model_spec_execution/conditioning_revises_the_same_type_and_retains_joint_uncertainty_observation_law.json"
                    )
                ),
                reasoning="Joint law with category-specific parameter elements",
            ),
        )
        dynamical_model_spec = (
            _conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model()
        )
    count = 3
    samples = {
        site.name: 100 * (index + 1)
        + jnp.arange(count * np.prod(site.shape), dtype=float).reshape(count, *site.shape)
        for index, site in enumerate(
            build_site_registry(compile_model_fixture(dynamical_model_spec))
        )
    }
    bindings, auxiliary = parameter_bindings(compile_model_fixture(dynamical_model_spec))
    if categorical:
        assert any(len(binding.coordinates) > 1 for binding in bindings)
    for coordinate in auxiliary:
        samples[coordinate.site_name] = (
            samples[coordinate.site_name].at[(slice(None), *coordinate.indices)].set(0)
        )
    samples.update(
        assemble_deterministics_from_registry(samples, compile_model_fixture(dynamical_model_spec))
    )
    paths = jnp.arange(count * 4 * 2, dtype=float).reshape(count, 4, 2)
    result = particle_posterior(JointPosteriorDraws(samples, paths))
    conditioned, _ = condition_model(
        dynamical_model_spec,
        compile_model_fixture(dynamical_model_spec),
        result,
        times=jnp.arange(4),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert type(conditioned) is DynamicalModelSpec
    assert dynamical_model_spec.time_points == ()
    assert len(dynamical_model_spec.distributions) == len(dynamical_model_spec.parameters)
    assert len(conditioned.distributions) == 1
    assert all(
        p.distribution == next(iter(conditioned.distributions)) for p in conditioned.parameters
    )
    payload = conditioned.model_dump(mode="json")
    assert not {"posterior", "prior", "provenance", "diagnostics", "result"} & payload.keys()
    assert "array_ref" in conditioned.model_dump_json()
    from grassmoth_server.study.persistence.writes import write_model_revision

    info = write_model_revision(store, conditioned, derived_from={}, produced_by="fit")
    loaded = read_model(store, info.revision)
    assert loaded == conditioned
    assert (
        input_fingerprints(dynamical_model_spec)["compilation"]
        == input_fingerprints(loaded)["compilation"]
    )
    restored = model_draws(compile_model_fixture(loaded))
    for name, values in samples.items():
        np.testing.assert_array_equal(restored.parameters[name], values)
    np.testing.assert_array_equal(restored.latent_paths, paths)
    sampled = sample_model_laws(compile_model_fixture(loaded), draws=12, key=jax.random.PRNGKey(4))
    assert sampled.latent_paths is not None
    for draw in range(12):
        atom = int(sampled.latent_paths[draw, 0, 0] // 8)
        np.testing.assert_array_equal(sampled.latent_paths[draw], paths[atom])
        for name, values in samples.items():
            np.testing.assert_array_equal(sampled.parameters[name][draw], values[atom])
    # Entity/parameter list order carries no joint distribution coordinates.
    reordered = loaded.with_entities(parameters=tuple(reversed(loaded.parameters)))
    for name, values in restored.parameters.items():
        np.testing.assert_array_equal(
            model_draws(compile_model_fixture(reordered)).parameters[name], values
        )
    assert (
        loaded.with_entities(
            edges=replace_constructs(loaded.edges, tuple(reversed(loaded.constructs)))
        )
        == loaded
    )
    np.testing.assert_array_equal(
        empirical_atoms(next(iter(loaded.distributions.values()))),
        empirical_atoms(next(iter(conditioned.distributions.values()))),
    )
    identity, law = next(iter(loaded.distributions.items()))
    # A joint law's coordinates follow the outcome's scope, which checks them.
    with pytest.raises(ValueError, match="event width must match"):
        StructuralSelection(
            loaded.with_entities(
                distributions={identity: empirical_distribution(empirical_atoms(law)[:, :-1])}
            ),
            None,
        )
    with pytest.raises(ValueError, match="identity and event width must match"):
        StructuralSelection(
            loaded.with_entities(
                law_layouts={
                    identity: loaded.law_layouts[identity].revised(time_points=(0.0, 1.0, 2.0, 5.0))
                }
            ),
            None,
        )
    if categorical:
        construct = loaded.constructs[1]
        indicator = construct.indicators[0]
        renamed = loaded.with_entities(
            edges=replace_constructs(
                loaded.edges,
                (
                    construct.revised(
                        indicators=(
                            indicator.revised(
                                observation=indicator.observation.revised(
                                    categorical_levels=("medium", "low", "high")
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        assert renamed.law_layouts == loaded.law_layouts
