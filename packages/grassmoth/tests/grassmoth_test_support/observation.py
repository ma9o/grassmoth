"""Synthetic typed operands for parameterized exact-law and predictive tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.identity import ConstructId
from grassmoth.artifacts.model.expressions import LiteralExpression, coefficient, state
from grassmoth.artifacts.model.likelihood import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    BernoulliLogitsLawSpec,
    BernoulliProbsLawSpec,
    BetaLawSpec,
    CategoricalLawSpec,
    DeltaLawSpec,
    GammaLawSpec,
    Law,
    LinkFunction,
    NegativeBinomial2LawSpec,
    NormalLawSpec,
    OrderedLogisticLawSpec,
    PoissonLawSpec,
    StudentTLawSpec,
    function,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.dynamics.expression import BoundExpression
from grassmoth.models.ssm.execution.emissions import gaussian_block_log_prob
from grassmoth.models.ssm.execution.observation_distributions import (
    evaluate_law,
    gaussian_distribution,
    masked_law_log_prob,
    safe_native,
    with_response,
)
from grassmoth.models.ssm.execution.observation_model import compile_observation_model

type ObservationParameters = Mapping[str, jax.Array | int | float]


def observation_laws(
    families: Sequence[DistributionFamily],
    links: Sequence[LinkFunction] | None = None,
    parameters: ObservationParameters | None = None,
    *,
    level_counts: Sequence[int] | None = None,
) -> tuple[Law[BoundExpression], ...]:
    """Build native fields, with dynamic leaves, from a test's family coverage table."""
    parameters = {} if parameters is None else parameters
    links = (
        [LinkFunction(OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[family][0]) for family in families]
        if links is None
        else links
    )
    category_present = any(
        family in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        for family in families
    )
    if category_present and level_counts is None:
        level_counts = tuple(int(count) for count in np.asarray(parameters["obs_level_counts"]))
    counts = tuple(level_counts) if level_counts is not None else (0,) * len(families)
    maximum = max(counts, default=0)
    predictor = state(ConstructId("construct:test-predictor"))
    laws = []
    for channel, (family, link) in enumerate(zip(families, links, strict=True)):
        if link.value not in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[family]:
            raise ValueError(f"{link.value!r} invalid for observation family {family.value!r}")

        def inverse(eta):
            valid = jnp.isfinite(eta) & (eta > 0)
            return jnp.where(valid, 1.0 / jnp.where(valid, eta, 1.0), jnp.nan)

        response = {
            LinkFunction.IDENTITY: lambda eta: eta,
            LinkFunction.LOG: jnp.exp,
            LinkFunction.INVERSE: inverse,
            LinkFunction.LOGIT: jax.nn.sigmoid,
            LinkFunction.PROBIT: jax.scipy.special.ndtr,
            LinkFunction.CUMULATIVE_LOGIT: lambda eta: eta,
            LinkFunction.SOFTMAX: lambda eta: eta,
        }[link]
        response_expression = {
            LinkFunction.LOG: function("exp", predictor),
            LinkFunction.LOGIT: function("sigmoid", predictor),
            LinkFunction.PROBIT: function("normal_cdf", predictor),
            LinkFunction.INVERSE: LiteralExpression(value=1) / predictor,
        }.get(link, predictor)

        def operand(
            evaluate,
            *values,
            expression=predictor,
            event_size=0,
            bound_response=response,
            bound_link=link,
        ):
            return BoundExpression(
                expression,
                evaluate,
                (),
                tuple(jnp.asarray(value) for value in values),
                event_size,
                maximum,
                bound_response,
                bound_link,
            )

        eta = operand(lambda eta, _scale, _values: eta)
        scale = operand(
            lambda _eta, scale, _values: scale, expression=coefficient(1, "observation_scale")
        )
        mu = operand(
            lambda eta, _scale, _values, response=response: response(eta),
            expression=response_expression,
        )
        if family == DistributionFamily.DELTA:
            law = DeltaLawSpec(v=eta)
        elif family == DistributionFamily.GAUSSIAN:
            law = NormalLawSpec(loc=eta, scale=scale)
        elif family == DistributionFamily.STUDENT_T:
            law = StudentTLawSpec(
                df=operand(lambda _eta, _scale, values: values[0], parameters["obs_df"]),
                loc=eta,
                scale=scale,
            )
        elif family == DistributionFamily.POISSON:
            law = PoissonLawSpec(rate=mu)
        elif family == DistributionFamily.GAMMA:
            shape = parameters["obs_shape"]
            rate_expression = coefficient(None, "shape") / response_expression
            rate = operand(
                lambda eta, _scale, values, response=response: values[0] / response(eta),
                shape,
                expression=rate_expression,
            )
            law = GammaLawSpec(
                concentration=operand(lambda _eta, _scale, values: values[0], shape), rate=rate
            )
        elif family == DistributionFamily.BERNOULLI:
            law = (
                BernoulliLogitsLawSpec(logits=eta)
                if link == LinkFunction.LOGIT
                else BernoulliProbsLawSpec(probs=mu)
            )
        elif family == DistributionFamily.NEGATIVE_BINOMIAL:
            law = NegativeBinomial2LawSpec(
                mean=mu,
                concentration=operand(lambda _eta, _scale, values: values[0], parameters["obs_r"]),
            )
        elif family == DistributionFamily.BETA:
            concentration = parameters["obs_concentration"]
            alpha = operand(
                lambda eta, _scale, values, response=response: response(eta) * values[0],
                concentration,
                expression=response_expression * coefficient(None, "concentration"),
            )
            beta = operand(
                lambda eta, _scale, values, response=response: (1.0 - response(eta)) * values[0],
                concentration,
            )
            law = BetaLawSpec(concentration1=alpha, concentration0=beta)
        elif family == DistributionFamily.ORDERED_LOGISTIC:
            cutpoints = jnp.asarray(parameters["obs_ordered_cutpoints"])[
                channel, : counts[channel] - 1
            ]
            law = OrderedLogisticLawSpec(
                predictor=eta,
                cutpoints=operand(
                    lambda _eta, _scale, values: values[0],
                    cutpoints,
                    event_size=counts[channel] - 1,
                ),
            )
        elif family == DistributionFamily.CATEGORICAL:
            width = counts[channel] - 1
            intercepts = jnp.asarray(parameters["obs_cat_intercepts"])[channel, :width]
            slopes = jnp.asarray(parameters["obs_cat_slopes"])[channel, :width]
            logits = operand(
                lambda eta, _scale, values: jnp.concatenate(
                    (jnp.zeros((1,), dtype=eta.dtype), values[0] + values[1] * eta)
                ),
                intercepts,
                slopes,
                event_size=counts[channel],
            )
            law = CategoricalLawSpec(logits=logits)
        else:
            raise ValueError(f"No observation fixture for {family}")
        laws.append(law)
    return tuple(laws)


def observation_kernel(families, links=None, parameters=None):
    laws = observation_laws(families, links, parameters)
    return compile_observation_model(laws, manifest_cov=jnp.eye(len(laws))).kernel


def mean_density(bound: Law[BoundExpression]):
    def density(y, mean, R, mask):
        baseline = jnp.ones_like(mean) if isinstance(bound, GammaLawSpec) else jnp.zeros_like(mean)
        law = with_response(evaluate_law(bound, baseline, jnp.sqrt(jnp.diag(R))), mean)
        if isinstance(law, NormalLawSpec):
            return gaussian_block_log_prob(y, mean, R, mask)
        return masked_law_log_prob(law, y, mask)

    return density


def mean_sampler(bound: Law[BoundExpression]):
    def sample(key, mean, R):
        baseline = jnp.ones_like(mean) if isinstance(bound, GammaLawSpec) else jnp.zeros_like(mean)
        law = with_response(evaluate_law(bound, baseline, jnp.sqrt(jnp.diag(R))), mean)
        native, valid = safe_native(law)
        if isinstance(law, NormalLawSpec):
            native = gaussian_distribution(mean, R)
        return jnp.where(valid, native.sample(key), jnp.nan)

    return sample
