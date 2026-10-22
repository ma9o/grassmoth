"""Conditional laws retain native density semantics, ownership, and explicit completion."""

import operator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.identity import (
    ConstructId,
    DistributionId,
    ParameterId,
    scientific_id,
)
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import (
    BinaryExpression,
    CallExpression,
    CoefficientExpression,
    Expression,
    LiteralExpression,
    StateExpression,
    coefficient,
    state,
)
from grassmoth.artifacts.model.likelihood import (
    LikelihoodSpec,
    NormalLawSpec,
    ObservationLawSpec,
    function,
)
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.model.parameter_context import iter_coefficient_uses
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from grassmoth.compilation_errors import IncompleteModelError
from tests.support.helpers import make_model
from tests.support.inference import compile_model_fixture
from tests.support.model import (
    construct_named,
    indicator_named,
    likelihood_named,
    parameter_named,
    x_y_model,
)
from tests.support.observation import observation_kernel


def _partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_complete_test_model() -> (
    DynamicalModelSpec
):
    _MANIFEST_MEAN_X_OBS_PARAMETER_ID = ParameterId(
        "parameter:45b9d3457845e74a7a9b59f18ad0a57bb8091bb187650d4c597c2c34aefdce05"
    )
    _MANIFEST_MEAN_X_OBS_DISTRIBUTION_ID = DistributionId(
        "distribution:691046b2f9e22a1f8ac7a79808b014fc604bbf2dae8819b0e0594fd70444c3f3"
    )
    dynamical_model_spec = x_y_model()
    x = construct_named(dynamical_model_spec, "X")
    x_obs = indicator_named(dynamical_model_spec, "X_obs")
    x_obs_likelihood = likelihood_named(dynamical_model_spec, "X_obs")
    rho_x = parameter_named(dynamical_model_spec, "rho_X")
    rho_y = parameter_named(dynamical_model_spec, "rho_Y")
    beta_x_y = parameter_named(dynamical_model_spec, "beta_X_Y")
    sigma_x = parameter_named(dynamical_model_spec, "sigma_X")
    sigma_y = parameter_named(dynamical_model_spec, "sigma_Y")
    x_obs_revised = x_obs.revised(
        likelihood=x_obs_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(_MANIFEST_MEAN_X_OBS_PARAMETER_ID, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(x.id))
                ),
                scale=coefficient(0.0, "observation_scale"),
            ),
            standardized=False,
            reasoning="Partial",
        )
    )
    x_revised = x.revised(indicators=(x_obs_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (x_revised,)),
        parameters=(
            rho_x,
            rho_y,
            beta_x_y,
            sigma_x,
            ParameterSpec(
                id=_MANIFEST_MEAN_X_OBS_PARAMETER_ID,
                name="manifest_mean_X_obs",
                description="likelihood.intercept for manifest_mean_X_obs",
                distribution=_MANIFEST_MEAN_X_OBS_DISTRIBUTION_ID,
            ),
            sigma_y,
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _MANIFEST_MEAN_X_OBS_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=0.5, validate_args=False
            ),
        },
    )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("family", "link", "observed", "observation_law_payload", "revise_law_payload"),
    [
        pytest.param(
            "gaussian",
            "identity",
            0.4,
            "likelihood_expressions/gaussian_identity_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gaussian-identity-0_4.json",
            id="gaussian-identity-0.4",
        ),
        pytest.param(
            "student_t",
            "identity",
            0.4,
            "likelihood_expressions/student_t_identity_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_student_t-identity-0_4.json",
            id="student_t-identity-0.4",
        ),
        pytest.param(
            "poisson",
            "log",
            2,
            "likelihood_expressions/poisson_log_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_poisson-log-2.json",
            id="poisson-log-2",
        ),
        pytest.param(
            "gamma",
            "log",
            1.2,
            "likelihood_expressions/gamma_log_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gamma-log-1_2.json",
            id="gamma-log-1.2",
        ),
        pytest.param(
            "gamma",
            "inverse",
            1.2,
            "likelihood_expressions/gamma_inverse_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gamma-inverse-1_2.json",
            id="gamma-inverse-1.2",
        ),
        pytest.param(
            "bernoulli",
            "logit",
            1,
            "likelihood_expressions/bernoulli_logit_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_bernoulli-logit-1.json",
            id="bernoulli-logit-1",
        ),
        pytest.param(
            "bernoulli",
            "probit",
            1,
            "likelihood_expressions/bernoulli_probit_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_bernoulli-probit-1.json",
            id="bernoulli-probit-1",
        ),
        pytest.param(
            "negative_binomial",
            "log",
            3,
            "likelihood_expressions/negative_binomial_log_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_negative_binomial-log-3.json",
            id="negative_binomial-log-3",
        ),
        pytest.param(
            "beta",
            "logit",
            0.4,
            "likelihood_expressions/beta_logit_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_beta-logit-0_4.json",
            id="beta-logit-0.4",
        ),
        pytest.param(
            "beta",
            "probit",
            0.4,
            "likelihood_expressions/beta_probit_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_beta-probit-0_4.json",
            id="beta-probit-0.4",
        ),
        pytest.param(
            "ordered_logistic",
            "cumulative_logit",
            1,
            "likelihood_expressions/ordered_logistic_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_ordered_logistic-cumulative_logit-1.json",
            id="ordered_logistic-cumulative_logit-1",
        ),
        pytest.param(
            "categorical",
            "softmax",
            2,
            "likelihood_expressions/categorical_softmax_law.json",
            "likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_categorical-softmax-2.json",
            id="categorical-softmax-2",
        ),
        pytest.param(
            "delta",
            "identity",
            0.35,
            "likelihood_expressions/exact_state_law.json",
            "likelihood_expressions/exact_state_likelihood.json",
            id="delta-identity-0.35",
        ),
        pytest.param(
            "delta",
            "identity",
            0.4,
            "likelihood_expressions/exact_state_law.json",
            "likelihood_expressions/exact_state_likelihood.json",
            id="delta-identity-0.4",
        ),
    ],
)
def test_native_conditional_law_matches_exact_emission_lowering(
    family, link, observed, observation_law_payload, revise_law_payload
):
    likelihood = LikelihoodSpec.model_validate_json(
        (Path(__file__).resolve().parents[3] / "fixtures/models" / revise_law_payload).read_text()
    )
    values = {
        "loading": 0.8,
        "observation_intercept": 0.3,
        "observation_scale": 0.7,
        "degrees_of_freedom": 7.0,
        "shape": 2.0,
        "dispersion": 4.0,
        "concentration": 5.0,
        "cutpoint_base": -0.6,
        "cutpoint_gaps": jnp.array([1.1]),
        "category_intercepts": jnp.array([-0.5, 0.6]),
        "category_slopes": jnp.array([0.8, -0.3]),
    }
    functions = {
        "exp": jnp.exp,
        "sigmoid": jax.nn.sigmoid,
        "normal_cdf": jax.scipy.special.ndtr,
        "ordered_cutpoints": lambda base, gaps: jnp.concatenate(
            (jnp.array([base]), base + jnp.cumsum(gaps))
        ),
        "category_logits": lambda predictor, intercepts, slopes: jnp.concatenate(
            (jnp.zeros(1), intercepts + slopes * predictor)
        ),
    }
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "power": operator.pow,
        "maximum": jnp.maximum,
    }

    def evaluate(expression):
        # Independent authored-tree reference; it does not call the SymPy compiler.
        match expression:
            case LiteralExpression():
                return jnp.asarray(expression.value)
            case StateExpression():
                return jnp.asarray(0.35)
            case CoefficientExpression():
                return values[expression.role]
            case BinaryExpression():
                return operations[expression.operator](
                    evaluate(expression.left), evaluate(expression.right)
                )
            case CallExpression():
                return functions[expression.function](
                    *(evaluate(argument) for argument in expression.arguments)
                )

    arguments = {name: evaluate(value) for name, value in likelihood.law.operands()}
    expected = getattr(dist, likelihood.law.distribution)(**arguments).log_prob(observed)
    terms = likelihood.parsed
    extra = {
        operand.meaning.quantity.value: values[operand.role]
        for operand in terms.auxiliary
        if operand.role != "observation_scale"
    }
    if family == "ordered_logistic":
        extra["obs_ordered_cutpoints"] = arguments["cutpoints"][None, :]
        extra["obs_level_counts"] = jnp.array([3])
    if family == "categorical":
        extra["obs_cat_intercepts"] = values["category_intercepts"][None, :]
        extra["obs_cat_slopes"] = values["category_slopes"][None, :]
        extra["obs_level_counts"] = jnp.array([3])
    density = observation_kernel([terms.family], [terms.link], extra).log_prob_fn
    actual = density(
        jnp.array([observed], dtype=float),
        jnp.atleast_1d(evaluate(terms.predictor)),
        jnp.array([[0.49]]),
        jnp.ones(1),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert (
        float(
            density(
                jnp.array([observed], dtype=float),
                jnp.atleast_1d(evaluate(terms.predictor)),
                jnp.array([[0.49]]),
                jnp.zeros(1),
            )
        )
        == 0
    )
    assert (terms.family, terms.link) == (family, link)


@pytest.mark.contract
@pytest.mark.parametrize(
    (
        "family",
        "link",
        "_observed",
        "observation_law_payload",
        "with_likelihood_coefficients_payload",
    ),
    [
        pytest.param(
            "gaussian",
            "identity",
            0.4,
            "likelihood_expressions/gaussian_identity_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gaussian-identity-0_4.json",
            id="gaussian-identity-0.4",
        ),
        pytest.param(
            "student_t",
            "identity",
            0.4,
            "likelihood_expressions/student_t_identity_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_student_t-identity-0_4.json",
            id="student_t-identity-0.4",
        ),
        pytest.param(
            "poisson",
            "log",
            2,
            "likelihood_expressions/poisson_log_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_poisson-log-2.json",
            id="poisson-log-2",
        ),
        pytest.param(
            "gamma",
            "log",
            1.2,
            "likelihood_expressions/gamma_log_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gamma-log-1_2.json",
            id="gamma-log-1.2",
        ),
        pytest.param(
            "gamma",
            "inverse",
            1.2,
            "likelihood_expressions/gamma_inverse_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gamma-inverse-1_2.json",
            id="gamma-inverse-1.2",
        ),
        pytest.param(
            "bernoulli",
            "logit",
            1,
            "likelihood_expressions/bernoulli_logit_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_bernoulli-logit-1.json",
            id="bernoulli-logit-1",
        ),
        pytest.param(
            "bernoulli",
            "probit",
            1,
            "likelihood_expressions/bernoulli_probit_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_bernoulli-probit-1.json",
            id="bernoulli-probit-1",
        ),
        pytest.param(
            "negative_binomial",
            "log",
            3,
            "likelihood_expressions/negative_binomial_log_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_negative_binomial-log-3.json",
            id="negative_binomial-log-3",
        ),
        pytest.param(
            "beta",
            "logit",
            0.4,
            "likelihood_expressions/beta_logit_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_beta-logit-0_4.json",
            id="beta-logit-0.4",
        ),
        pytest.param(
            "beta",
            "probit",
            0.4,
            "likelihood_expressions/beta_probit_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_beta-probit-0_4.json",
            id="beta-probit-0.4",
        ),
        pytest.param(
            "ordered_logistic",
            "cumulative_logit",
            1,
            "likelihood_expressions/ordered_logistic_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_ordered_logistic-cumulative_logit-1.json",
            id="ordered_logistic-cumulative_logit-1",
        ),
        pytest.param(
            "categorical",
            "softmax",
            2,
            "likelihood_expressions/categorical_softmax_law.json",
            "likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_categorical-softmax-2.json",
            id="categorical-softmax-2",
        ),
    ],
)
def test_authored_laws_preserve_coefficient_identities(
    family, link, _observed, observation_law_payload, with_likelihood_coefficients_payload
):
    authored = LikelihoodSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[3]
            / "fixtures/models"
            / with_likelihood_coefficients_payload
        ).read_text()
    )
    assert authored.parsed.loadings[ConstructId("construct:x")].value == -1
    assert authored.parsed.intercept.value == scientific_id("parameter", "baseline")
    assert (authored.parsed.family, authored.parsed.link) == (family, link)
    assert set(authored.model_dump()) == {"law", "standardized", "reasoning", "sources"}
    assert LikelihoodSpec.model_validate_json(authored.model_dump_json()) == authored


@pytest.mark.contract
def test_completion_binding_and_serialization_follow_the_same_cross_loading():
    dynamical_model_spec = x_y_model()
    owner, other = dynamical_model_spec.constructs
    indicator = owner.indicators[0]
    revised = LikelihoodSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[3]
            / "fixtures/models"
            / "likelihood_expressions/completion_binding_equations_and_serialization_follow_the_same_cross_loading_revise_law.json"
        ).read_text()
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (owner.revised(indicators=(indicator.revised(likelihood=revised),)),),
        )
    )
    compile_model_fixture(dynamical_model_spec)
    np.testing.assert_allclose(
        compile_model_fixture(dynamical_model_spec).loading_block.template, [[1, 0.25], [0, 1]]
    )
    uses = [
        use
        for use in iter_coefficient_uses(dynamical_model_spec)
        if use.quantity == SiteKind.LOADING
    ]
    cross = next(use for use in uses if use.value == 0.25)
    assert {ref.id for ref in cross.owners} == {indicator.observation.id, other.id}
    assert (
        DynamicalModelSpec.model_validate_json(
            dynamical_model_spec.model_dump_json()
        ).materialized()
        == dynamical_model_spec
    )
    renamed = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (other.revised(name="Renamed"),))
    )
    assert {p.id for p in renamed.parameters} == {p.id for p in dynamical_model_spec.parameters}


@pytest.mark.contract
def test_partial_law_is_explicit_and_unsupported_formulas_fail_before_execution():
    likelihood = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[3]
                / "fixtures/models"
                / "likelihood_expressions/gaussian_identity_law.json"
            ).read_text()
        ),
        reasoning="Partial",
    )
    assert all(operand.value is None for operand in likelihood.parsed.operands)
    with pytest.raises(ValidationError, match="affine"):
        LikelihoodSpec(
            law=NormalLawSpec(
                loc=function("exp", likelihood.parsed.predictor),
                scale=coefficient(None, "observation_scale"),
            ),
            reasoning="Unsupported nonlinear observation predictor",
        )
    with pytest.raises(ValidationError, match="Field required"):
        NormalLawSpec.model_validate({"rate": likelihood.parsed.predictor})
    with pytest.raises(ValidationError, match="non-negative"):
        coefficient(-1, "observation_scale")

    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    owner = dynamical_model_spec.constructs[0]
    indicator = owner.indicators[0]
    partial = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[3]
                / "fixtures/models"
                / "likelihood_expressions/partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_observation_law_2.json"
            ).read_text()
        ),
        reasoning="Partial",
    )

    def with_law(law):
        return dynamical_model_spec.with_entities(
            edges=replace_constructs(
                dynamical_model_spec.edges,
                (owner.revised(indicators=(indicator.revised(likelihood=law),)),),
            )
        )

    unfinished = with_law(partial)
    with pytest.raises(IncompleteModelError):
        compile_model_fixture(unfinished)
    compile_model_fixture(
        _partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_complete_test_model()
    )
    with pytest.raises(ValidationError, match="unknown constructs"):
        with_law(likelihood)
    wrong_owner = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[3]
                / "fixtures/models"
                / "common/observed_gaussian_observation_law.json"
            ).read_text()
        ),
        reasoning="Wrong owner",
    )
    with pytest.raises(ValidationError, match="must include its measured construct"):
        with_law(wrong_owner)


@pytest.mark.contract
def test_law_map_preserves_every_constructor_operand_and_composes():
    """Mapping operands neither changes a law nor drops a native argument."""
    from pydantic import TypeAdapter

    from grassmoth.artifacts.model.expressions import LiteralExpression
    from grassmoth.artifacts.model.likelihood import (
        OBSERVATION_LAW_TYPES,
        ObservationLawSpec,
        map_law,
    )

    def literal(value: Expression) -> float:
        assert isinstance(value, LiteralExpression)
        return value.value

    def identity(value: Expression) -> Expression:
        return value

    def increment(value: Expression) -> float:
        return literal(value) + 1

    def double(value: float) -> float:
        return value * 2

    def increment_then_double(value: Expression) -> float:
        return (literal(value) + 1) * 2

    for constructor in OBSERVATION_LAW_TYPES:
        law = TypeAdapter(ObservationLawSpec).validate_python(
            {
                "distribution": constructor.model_fields["distribution"].default,
                **{
                    name: LiteralExpression(value=index)
                    for index, name in enumerate(constructor.model_fields)
                    if name != "distribution"
                },
            }
        )
        assert map_law(law, identity).model_dump() == law.model_dump()
        first = map_law(law, increment)
        composed = map_law(first, double)
        direct = map_law(law, increment_then_double)
        assert composed.distribution == law.distribution
        assert composed.model_dump() == direct.model_dump()
        assert set(composed.model_dump()) == set(law.model_dump())
