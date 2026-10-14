"""Small analytic checks for native prior laws and their JSON boundary."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints, transforms
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.expressions import restoring_force
from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model.parameter import SiteKind
from nof1_causal_lab.artifacts.probability.families import PriorDistributionFamily
from nof1_causal_lab.artifacts.probability.native import (
    batch_prior_distributions,
    distribution_from_params,
    interval_effect_to_rate,
    persistence_to_decay,
    prior_reference_value,
)
from nof1_causal_lab.artifacts.probability.serialization import NumPyroDistribution
from nof1_causal_lab.models.structure import StructuralSelection
from tests.support.inference import compile_fit_fixture, compile_model_fixture
from tests.support.model import (
    construct_named,
    load_model_fixture,
    parameter_for,
    parameter_laws,
    parameter_named,
    replace_parameters,
    without_parameters,
)


def _scientific_roundtrip_preserves_distinct_native_coordinate_laws_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = load_model_fixture(
        "dynamics_config/scientific_model_roundtrip_preserves_derived_dynamics_model_fixture.json"
    )
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    (latent_0_potential,) = latent_0.dynamics
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_latent_1_hill_emax = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EMAX, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_n = parameter_for(
        dynamical_model_spec, SiteKind.HILL_N, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_ec50 = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EC50, "latent_0", "latent_1"
    )
    latent_0_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_0")
    latent_1_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_1")
    latent_0_revised = latent_0.revised(
        dynamics=(
            DriftMechanismSpec(
                id=latent_0_potential.id,
                expression=restoring_force(
                    latent_0.id, center=0.0, stiffness=latent_0_dynamics_decay.id, quartic=0.0
                ),
            ),
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec,
        latent_0_latent_1_hill_emax,
        latent_0_latent_1_hill_n,
        latent_0_latent_1_hill_ec50,
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            tuple(
                edge
                for edge in dynamical_model_spec.edges
                if (edge.cause.name, edge.effect.name) not in (("latent_0", "latent_1"),)
            ),
            (latent_0_revised,),
        ),
        parameters=parameters,
        distributions={
            key: law
            for key, law in parameter_laws(
                dynamical_model_spec,
                {
                    latent_0_t0_means.id: dist.Normal(loc=-1.0, scale=0.5, validate_args=False),
                    latent_1_t0_means.id: dist.StudentT(
                        df=4.0, loc=0.3, scale=0.7, validate_args=False
                    ),
                },
            ).items()
            if key in distributions
        },
    )


def _compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_complete_test_model() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "prior_distributions/compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_complete_test_model.json"
    )


def _compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = (
        _compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_complete_test_model()
    )
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            rho_mood.revised(transform=rho_mood.transform.revised(interval_days=7.0)),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {rho_mood.id: dist.Beta(concentration1=2.0, concentration0=3.0, validate_args=False)},
        ),
    )


_ADAPTER = TypeAdapter(NumPyroDistribution)


@pytest.mark.inference(concern="sampling")
def test_beta_persistence_has_exact_density_and_jacobian():
    prior = persistence_to_decay(dist.Beta(2.0, 3.0), 7.0)
    decay = 0.2
    rho = math.exp(-7.0 * decay)
    expected = math.log(12.0) + math.log(rho) + 2.0 * math.log1p(-rho) + math.log(7.0 * rho)
    assert float(prior.log_prob(decay)) == pytest.approx(expected, abs=2e-6)
    gradient = jax.grad(prior.log_prob)(decay)
    assert float(gradient) == pytest.approx(-14.0 + 14.0 * rho / (1.0 - rho), abs=2e-5)
    assert not bool(prior.support(-0.1))
    assert bool(prior.support(decay))


@pytest.mark.inference(concern="predictive")
def test_persistence_draw_is_the_exact_pushforward():
    base = dist.Beta(2.0, 3.0)
    prior = persistence_to_decay(base, 3.0)
    key = jax.random.PRNGKey(7)
    expected = -jnp.log(base.sample(key, (3,))) / 3.0
    np.testing.assert_array_equal(prior.sample(key, (3,)), expected)


@pytest.mark.contract
@pytest.mark.parametrize(
    "base", [dist.Normal(0.5, 0.1), dist.Uniform(-0.1, 1.0), dist.Uniform(0.0, 1.1)]
)
def test_persistence_requires_an_authored_law_on_the_unit_interval(base):
    with pytest.raises(ValueError, match="support within"):
        persistence_to_decay(base, 1.0)


@pytest.mark.inference(concern="sampling")
def test_interval_effect_preserves_uniform_family_and_bounds():
    prior = interval_effect_to_rate(dist.Uniform(-2.0, 4.0), 2.0)
    assert float(prior.support.lower_bound) == -1.0
    assert float(prior.support.upper_bound) == 2.0
    assert float(prior.log_prob(0.5)) == pytest.approx(-math.log(3.0))


@pytest.mark.inference(concern="sampling")
def test_transformed_vector_prior_roundtrip_preserves_density_and_positive_support():
    coordinates = [persistence_to_decay(dist.Beta(2.0, 3.0), interval) for interval in [1.0, 7.0]]
    prior = batch_prior_distributions(coordinates, (2,), support=constraints.positive)
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(prior))
    values = jnp.array([0.4, 0.1])
    np.testing.assert_allclose(prior.log_prob(values), restored.log_prob(values))
    assert np.all(np.asarray(restored.support(values)))
    assert not np.any(np.asarray(restored.support(-values)))
    abstract = jax.eval_shape(lambda law, value: law.log_prob(value), restored, values)
    assert abstract.shape == (2,)


@pytest.mark.inference(concern="sampling")
def test_different_coordinate_families_have_no_mixture_uncertainty():
    coordinates = [dist.Normal(0.0, 1.0), dist.Uniform(-2.0, 2.0)]
    prior = batch_prior_distributions(coordinates, (2,), support=constraints.real)
    values = jnp.array([0.2, 0.3])
    expected = jnp.array([coordinates[0].log_prob(values[0]), coordinates[1].log_prob(values[1])])
    np.testing.assert_allclose(prior.log_prob(values), expected)
    assert jnp.isneginf(prior.log_prob(jnp.array([0.2, 3.0]))[1])
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(prior))
    np.testing.assert_allclose(restored.log_prob(values), expected)


@pytest.mark.contract
def test_distribution_arguments_are_complete_and_native_validated():
    with pytest.raises(ValueError, match="requires exactly"):
        distribution_from_params(PriorDistributionFamily.GAMMA, {"concentration": 2.0})
    with pytest.raises(ValueError, match="invalid rate"):
        distribution_from_params(
            PriorDistributionFamily.GAMMA, {"concentration": 2.0, "rate": -1.0}
        )
    with pytest.raises(ValueError, match="lower < upper"):
        distribution_from_params(PriorDistributionFamily.UNIFORM, {"lower": 2.0, "upper": 1.0})


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("coordinates", "values", "support"),
    [
        ([dist.Normal(0.0, 1.0), dist.LogNormal(0.0, 0.5)], [-2.0, 0.3], constraints.real),
        (
            [dist.Gamma(2.0, 3.0), persistence_to_decay(dist.Beta(2.0, 3.0), 7.0)],
            [0.4, 0.1],
            constraints.positive,
        ),
    ],
)
def test_mixed_coordinate_gradients_and_roundtrip_ignore_inactive_laws(
    coordinates, values, support
):
    prior = batch_prior_distributions(coordinates, (2,), support=support)
    values = jnp.array(values)
    expected = jnp.array(
        [jax.grad(law.log_prob)(values[index]) for index, law in enumerate(coordinates)]
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: prior.log_prob(x).sum())(values), expected, atol=2e-6
    )
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(prior))
    np.testing.assert_allclose(restored.log_prob(values), prior.log_prob(values), atol=2e-6)
    np.testing.assert_allclose(
        jax.grad(lambda x: restored.log_prob(x).sum())(values), expected, atol=2e-6
    )
    key = jax.random.PRNGKey(4)
    np.testing.assert_array_equal(restored.sample(key, (3,)), prior.sample(key, (3,)))
    np.testing.assert_allclose(
        prior_reference_value(prior), [prior_reference_value(law) for law in coordinates]
    )


@pytest.mark.contract
def test_expanded_transformed_reference_is_an_anchor_not_a_claimed_mean():
    law = persistence_to_decay(dist.Beta(2.0, 2.0), 1.0).expand((3,))
    np.testing.assert_allclose(prior_reference_value(law), jnp.full(3, math.log(2.0)))


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    "law_for",
    [
        lambda index: dist.StudentT(4.0 + index, index / 4, 0.7),
        lambda index: dist.TruncatedNormal(index / 4, 0.7, low=-2.0, high=2.0),
        lambda index: dist.TransformedDistribution(
            dist.Normal(index / 4, 0.7), transforms.SigmoidTransform()
        ),
        lambda index: dist.MixtureGeneral(
            dist.Categorical(probs=jnp.array([0.3, 0.7])),
            [dist.Normal(index / 4, 0.7), dist.StudentT(4.0, 0.5, 0.8)],
        ),
    ],
    ids=["student-t", "truncated", "native-transform", "native-mixture"],
)
def test_native_parameter_trees_batch_without_a_family_or_transform_registry(law_for):
    coordinates = [law_for(index) for index in range(4)]
    prior = batch_prior_distributions(coordinates, (2, 2), support=coordinates[0].support)
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(prior))
    assert type(restored) is type(coordinates[0])
    assert restored.batch_shape == (2, 2)
    values = jnp.array([[0.2, 0.3], [0.4, 0.5]])
    expected = jnp.array(
        [law.log_prob(value) for law, value in zip(coordinates, values.reshape(-1), strict=True)]
    ).reshape((2, 2))
    np.testing.assert_allclose(restored.log_prob(values), expected, atol=2e-6)
    key = jax.random.PRNGKey(7)
    assert restored.sample(key, (3,)).shape == (3, 2, 2)
    np.testing.assert_array_equal(restored.sample(key, (3,)), prior.sample(key, (3,)))


@pytest.mark.inference(concern="sampling")
def test_native_batching_preserves_gradients_with_respect_to_constructor_parameters():
    def log_prob(loc):
        law = batch_prior_distributions(
            [dist.Normal(loc, 1.0), dist.Normal(-loc, 2.0)], (2,), support=constraints.real
        )
        return law.log_prob(jnp.array([0.2, 0.3])).sum()

    assert float(jax.grad(log_prob)(0.1)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.contract
def test_mixture_reference_uses_its_weights_instead_of_treating_components_as_coordinates():
    law = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.array([0.25, 0.75])),
        [dist.Normal(-1.0, 0.5), dist.StudentT(4.0, 3.0, 0.8)],
    )
    assert float(prior_reference_value(law)) == pytest.approx(2.0)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_scientific_roundtrip_preserves_distinct_native_coordinate_laws():
    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

    dynamical_model_spec = _scientific_roundtrip_preserves_distinct_native_coordinate_laws_with_parameter_distributions()
    restored = DynamicalModelSpec.model_validate_json(
        dynamical_model_spec.model_dump_json()
    ).materialized()
    assert restored == dynamical_model_spec
    before = compile_priors(
        compile_model_fixture(dynamical_model_spec), StructuralSelection(dynamical_model_spec, None)
    )[0]["t0_means_free"]
    after = compile_priors(compile_model_fixture(restored), StructuralSelection(restored, None))[0][
        "t0_means_free"
    ]
    value = jnp.array([-1.2, 0.5])
    np.testing.assert_allclose(after.log_prob(value), before.log_prob(value), atol=2e-6)
    key = jax.random.PRNGKey(7)
    np.testing.assert_array_equal(after.sample(key, (3,)), before.sample(key, (3,)))


@pytest.mark.inference(concern="sampling")
def test_compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law():
    from numpyro import handlers

    from nof1_causal_lab.artifacts.model.parameter import SiteKind
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    definition = (
        _compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_complete_test_model()
    )
    decay = next(
        p
        for p in definition.parameters
        if definition.parameter_context(p.id).quantity == SiteKind.DYNAMICS_DECAY
    )
    definition = _compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law_with_parameter_distributions()
    restored = DynamicalModelSpec.model_validate_json(definition.model_dump_json()).materialized()
    model = compile_fit_fixture(restored)
    binding = next(
        b
        for b in parameter_bindings(compile_model_fixture(restored))[0]
        if b.parameter_id == decay.id
    )
    value = jnp.array(0.2)
    with handlers.substitute(data={binding.site.name: value}):
        trace = handlers.trace(model.compiled_dynamical_model.dynamics.sample_params).get_trace(
            model.prior_runtime_bundle.priors.__getitem__
        )
    law = trace[binding.site.name]["fn"]
    expected = dist.Beta(2.0, 3.0).log_prob(jnp.exp(-7.0 * value)) + jnp.log(7.0) - 7.0 * value
    np.testing.assert_allclose(law.log_prob(value), expected, atol=2e-6)
    assert np.isfinite(jax.grad(law.log_prob)(value))


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("family", "params", "value"),
    [
        (PriorDistributionFamily.NORMAL, {"mu": 0.2, "sigma": 0.4}, 0.3),
        (PriorDistributionFamily.HALF_NORMAL, {"sigma": 0.4}, 0.3),
        (PriorDistributionFamily.GAMMA, {"concentration": 2.0, "rate": 4.0}, 0.3),
        (PriorDistributionFamily.LOG_NORMAL, {"mu": 0.2, "sigma": 0.4}, 0.3),
        (PriorDistributionFamily.EXPONENTIAL, {"rate": 4.0}, 0.3),
        (PriorDistributionFamily.BETA, {"alpha": 2.0, "beta": 4.0}, 0.3),
        (PriorDistributionFamily.UNIFORM, {"lower": -1.0, "upper": 2.0}, 0.3),
        (
            PriorDistributionFamily.TRUNCATED_NORMAL,
            {"mu": 0.2, "sigma": 0.4, "lower": -1.0, "upper": 2.0},
            0.3,
        ),
    ],
)
def test_approved_family_json_roundtrip_keeps_its_native_density(
    family: PriorDistributionFamily, params, value
):
    law = distribution_from_params(family, params)
    restored = _ADAPTER.validate_json(_ADAPTER.dump_json(law))
    np.testing.assert_allclose(law.log_prob(value), restored.log_prob(value), atol=1e-6)
