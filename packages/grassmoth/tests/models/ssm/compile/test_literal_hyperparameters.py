"""Fixed law operands survive compilation, exact scoring, and forward sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import Expression, coefficient
from grassmoth.artifacts.model.likelihood import NegativeBinomial2LawSpec, StudentTLawSpec, function
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.compile.inputs import compile_priors
from grassmoth.models.ssm.execution.dynamical_model import (
    HeterogeneousObservation,
    build_dynamical_model,
)
from grassmoth.models.ssm.execution.parameters import assemble_model_matrices
from grassmoth.models.ssm.inference.samplers.particle.guided import _guided_information
from grassmoth.models.ssm.predictive.registry_runtime import simulate_predictive_draws
from grassmoth.models.structure import StructuralSelection
from grassmoth_test_support.compute import COMPUTE_BUDGET
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import construct_named, load_model_fixture


def _fixed_model():
    return load_model_fixture(
        "runtime_ssm/testssmmodeldynamicsdispatch_test_nonlinear_dynamics_uses_vector_field_backend_method_model_fixture.json"
    )


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize("family", ["StudentT", "NegativeBinomial2"])
def test_literal_observation_operand_has_no_prior_and_preserves_native_law(family):
    model = _fixed_model()
    constructs = []
    for name, fixed in (("latent_0", 5.0), ("latent_1", 9.0)):
        construct = construct_named(model, name)
        (indicator,) = construct.indicators
        likelihood = indicator.likelihood
        assert likelihood is not None
        predictor = likelihood.parsed.predictor
        law = (
            StudentTLawSpec[Expression](
                df=coefficient(fixed, "degrees_of_freedom"),
                loc=predictor,
                scale=coefficient(0.2, "observation_scale"),
            )
            if family == "StudentT"
            else NegativeBinomial2LawSpec[Expression](
                mean=function("exp", predictor), concentration=coefficient(fixed, "dispersion")
            )
        )
        observation = indicator.observation.revised(
            measurement_dtype="continuous" if family == "StudentT" else "count"
        )
        constructs.append(
            construct.revised(
                indicators=(
                    indicator.revised(
                        observation=observation, likelihood=likelihood.revised(law=law)
                    ),
                )
            )
        )
    model = model.with_entities(edges=replace_constructs(model.edges, tuple(constructs)))
    compiled = compile_model_fixture(model)
    priors, bindings, _ = compile_priors(compiled, StructuralSelection(model, None))
    assert compiled.site_registry == ()
    assert priors == {}
    assert bindings == ()
    values, _ = assemble_model_matrices(compiled, {})
    dynamical = build_dynamical_model(compiled, values, t0=jnp.asarray(0.0))
    observation_model = dynamical.observation_model
    assert isinstance(observation_model, HeterogeneousObservation)
    state = jnp.array([0.35, -0.2])
    reading = jnp.array([0.6, -0.4]) if family == "StudentT" else jnp.array([3.0, 0.0])
    hyperparameters = jnp.array([5.0, 9.0])

    def native(at):
        predictor = observation_model.linear_predictor(at)
        return (
            dist.StudentT(hyperparameters, predictor, 0.2)
            if family == "StudentT"
            else dist.NegativeBinomial2(jnp.exp(predictor), hyperparameters)
        )

    def compiled_score(at):
        return dynamical.observation_model(at, None, 0.0).log_prob(reading)

    def oracle_score(at):
        return native(at).log_prob(reading).sum()

    np.testing.assert_allclose(compiled_score(state), oracle_score(state), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        jax.grad(compiled_score)(state), jax.grad(oracle_score)(state), rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        jax.hessian(compiled_score)(state), jax.hessian(oracle_score)(state), rtol=2e-5, atol=2e-5
    )

    # The guide consumes the same bound score; compare to the independent NumPyro score.
    def information(score):
        return _guided_information(
            lambda x, _t: (jnp.zeros(1), jax.grad(score)(x)),
            state[None],
            jnp.full((1, 2), 0.02),
            jnp.ones((1, 2), bool),
        )

    actual, expected = information(compiled_score), information(oracle_score)
    np.testing.assert_allclose(actual.precision, expected.precision, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.shift, expected.shift, rtol=2e-5, atol=2e-5)
    key = jax.random.PRNGKey(19)
    actual_draw = dynamical.observation_model(state, None, 0.0).sample(key)
    keys = jax.random.split(key, 2)
    predictor = observation_model.linear_predictor(state)
    expected_draw = jax.vmap(
        lambda k, p, h: (
            dist.StudentT(h, p, 0.2)
            if family == "StudentT"
            else dist.NegativeBinomial2(jnp.exp(p), h)
        ).sample(k)
    )(keys, predictor, hyperparameters)
    np.testing.assert_array_equal(actual_draw, expected_draw)
    simulated = simulate_predictive_draws(
        compiled,
        {name: value[None] for name, value in values.items()},
        jnp.array([0.0, 0.1, 0.2]),
        compute_budget=COMPUTE_BUDGET,
        seed=29,
    )
    assert simulated.trajectory.observations.shape == (1, 3, 2)
    assert np.isfinite(simulated.trajectory.observations).all()


@pytest.mark.contract
def test_literal_process_df_compiles_without_a_phantom_site_but_remains_unsupported():
    model = _fixed_model()
    construct = construct_named(model, "latent_0")
    construct = construct.revised(
        innovation_family=DistributionFamily.STUDENT_T,
        coefficients=(*construct.coefficients, coefficient(5.0, "process_degrees_of_freedom")),
    )
    model = model.with_entities(edges=replace_constructs(model.edges, (construct,)))
    compiled = compile_model_fixture(model)
    priors, _, _ = compile_priors(compiled, StructuralSelection(model, None))
    assert compiled.site_registry == ()
    assert priors == {}
    with pytest.raises(ValueError, match="requires Gaussian process diffusion"):
        simulate_predictive_draws(
            compiled, {}, jnp.array([0.0, 0.1]), compute_budget=COMPUTE_BUDGET
        )
