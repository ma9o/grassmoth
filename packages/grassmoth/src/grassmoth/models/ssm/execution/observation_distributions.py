"""One native-law interpreter and exact observation boundary adapters."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, assert_never, override

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from pydantic import BaseModel

from grassmoth.artifacts.model.likelihood import (
    OBSERVATION_LAW_TYPES,
    BernoulliLogitsLawSpec,
    BernoulliProbsLawSpec,
    BetaLawSpec,
    CategoricalLawSpec,
    DeltaLawSpec,
    GammaLawSpec,
    Law,
    NegativeBinomial2LawSpec,
    NormalLawSpec,
    OrderedLogisticLawSpec,
    PoissonLawSpec,
    StudentTLawSpec,
    map_law,
)
from grassmoth.models.ssm.covariance import symmetrize_with_jitter
from grassmoth.models.ssm.execution.contracts import NUMERICAL_EPSILON

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jax.typing import ArrayLike

    from grassmoth.models.ssm.dynamics.expression import BoundExpression


def _log_mass(value: jax.Array) -> jax.Array:
    """Exact zero mass without differentiating log(0) on an unselected branch."""
    positive = value > 0.0
    return jnp.where(positive, jnp.log(jnp.where(positive, value, 1.0)), -jnp.inf)


@jax.custom_jvp
def _positive_log(value: jax.Array) -> jax.Array:
    """Log a positive operand without squaring its reciprocal in higher derivatives."""
    return jnp.log(value)


@_positive_log.defjvp
def _positive_log_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (value,), (tangent,) = primals, tangents
    logarithm = _positive_log(value)
    # Reuse the log rule under nested differentiation. Unlike d(1 / value),
    # this keeps derivatives of log(exp(eta)) finite in float32's tails.
    return logarithm, tangent * jnp.exp(-logarithm)


def _negative_binomial_normalizer(concentration: jax.Array, value: jax.Array) -> jax.Array:
    return (
        jax.scipy.special.gammaln(1 + value)
        + jax.scipy.special.gammaln(concentration)
        - jax.scipy.special.gammaln(value + concentration)
    )


def _negative_binomial_log_prob(
    mean: jax.Array, concentration: jax.Array, value: jax.Array, normalizer: jax.Array
) -> jax.Array:
    zero_mean = mean == 0.0
    logits = _positive_log(jnp.where(zero_mean, 1.0, mean)) - _positive_log(concentration)
    positive_count = -(
        concentration * jax.nn.softplus(logits) + value * jax.nn.softplus(-logits) + normalizer
    )
    positive_count = jnp.where(zero_mean, -jnp.inf, positive_count)
    # Preserve the exact right derivative at mean zero without reciprocal rates.
    boundary = -concentration * jnp.log1p(jnp.where(zero_mean, mean, 0.0) / concentration)
    zero_count = jnp.where(zero_mean, boundary, -concentration * jax.nn.softplus(logits))
    return jnp.where(value == 0, zero_count, positive_count)


class BinaryProbabilityDistribution(dist.CategoricalLogits):
    """Exact Bernoulli endpoints, with the original categorical draw stream."""

    probability: jax.Array
    arg_constraints: ClassVar[dict[str, constraints.Constraint]] = {  # pyright: ignore[reportIncompatibleVariableOverride] - numpyro reads this class metadata but annotates it as an instance variable
        "probs": constraints.unit_interval
    }

    def __init__(self, probs: jax.Array) -> None:
        """Construct binary log masses directly from the probability, preserving exact zero and one."""
        self.probability = probs
        super().__init__(logits=jnp.stack((_log_mass(1.0 - probs), _log_mass(probs)), axis=-1))

    @override
    def log_prob(self, value: ArrayLike) -> jax.Array:
        return jnp.where(
            jnp.asarray(value) == 1, _log_mass(self.probability), _log_mass(1.0 - self.probability)
        )

    @property
    @override
    def mean(self) -> jax.Array:
        return self.probability

    @property
    @override
    def variance(self) -> jax.Array:
        return self.probability * (1.0 - self.probability)


class ZeroMeanNegativeBinomial(dist.MixtureGeneral):
    """NB2 including its exact zero-mean limit and right derivative."""

    response_mean: jax.Array
    concentration: jax.Array
    arg_constraints: ClassVar[dict[str, constraints.Constraint]] = {  # pyright: ignore[reportIncompatibleVariableOverride] - numpyro reads this class metadata but annotates it as an instance variable
        "mean": constraints.nonnegative,
        "concentration": constraints.positive,
    }

    def __init__(self, mean: jax.Array, concentration: jax.Array) -> None:
        """Represent zero response means as point masses and positive means as negative binomials."""
        self.response_mean = mean
        self.concentration = concentration
        zero = mean == 0.0
        selector = dist.CategoricalLogits(
            logits=jnp.stack(
                (jnp.where(zero, 0.0, -jnp.inf), jnp.where(zero, -jnp.inf, 0.0)), axis=-1
            )
        )
        super().__init__(
            selector,
            [
                dist.DiscreteUniform(jnp.zeros_like(mean, dtype=jnp.int32), 0),
                dist.NegativeBinomial2(jnp.where(zero, 1.0, mean), concentration),
            ],
            support=constraints.nonnegative_integer,
        )

    @override
    def log_prob(self, value: ArrayLike, intermediates: list[jax.Array] | None = None) -> jax.Array:
        del intermediates
        values = jnp.asarray(value)
        return _negative_binomial_log_prob(
            self.response_mean,
            self.concentration,
            values,
            _negative_binomial_normalizer(self.concentration, values),
        )

    @property
    @override
    def mean(self) -> jax.Array:
        return self.response_mean

    @property
    @override
    def variance(self) -> jax.Array:
        return self.response_mean + self.response_mean**2 / self.concentration


class OrderedCategoryDistribution(dist.OrderedLogistic):
    """True-width ordered density with exact zero masses and finite selected gradients."""

    @override
    def log_prob(self, value: ArrayLike) -> jax.Array:
        selected = jnp.take_along_axis(self.probs, jnp.asarray(value)[..., None], axis=-1)[..., 0]
        return _log_mass(selected) - jnp.log(jnp.sum(self.probs, axis=-1))


def to_native(law: Law[jax.Array]) -> dist.Distribution:
    """Interpret the closed authored algebra without family/link keys or argument bags."""
    match law:
        case DeltaLawSpec():
            return dist.Delta(law.v)
        case NormalLawSpec():
            return dist.Normal(law.loc, law.scale)
        case StudentTLawSpec():
            return dist.StudentT(law.df, law.loc, law.scale)
        case PoissonLawSpec():
            return dist.Poisson(law.rate)
        case GammaLawSpec():
            return dist.Gamma(law.concentration, law.rate)
        case BernoulliLogitsLawSpec():
            return dist.BernoulliLogits(logits=law.logits)
        case BernoulliProbsLawSpec():
            return BinaryProbabilityDistribution(law.probs)
        case NegativeBinomial2LawSpec():
            return ZeroMeanNegativeBinomial(law.mean, law.concentration)
        case BetaLawSpec():
            return dist.Beta(law.concentration1, law.concentration0)
        case OrderedLogisticLawSpec():
            return OrderedCategoryDistribution(law.predictor, law.cutpoints)
        case CategoricalLawSpec():
            return dist.CategoricalLogits(logits=law.logits)
    assert_never(law)


def category_probabilities(
    law: OrderedLogisticLawSpec[jax.Array] | CategoricalLawSpec[jax.Array],
) -> jax.Array:
    """Masses over the declared categories."""
    if isinstance(law, CategoricalLawSpec):
        return jnp.asarray(dist.CategoricalLogits(logits=law.logits).probs)
    return jnp.asarray(OrderedCategoryDistribution(law.predictor, law.cutpoints).probs)


def feasible_law(law: Law[jax.Array]) -> tuple[Law[jax.Array], jax.Array]:
    """Reject invalid parameters, substituting feasible operands before density evaluation."""
    native = to_native(law)
    valid = jnp.ones(native.batch_shape, dtype=bool)
    for name, value in law.operands():
        constraint = (
            constraints.real if isinstance(law, DeltaLawSpec) else native.arg_constraints[name]
        )
        finite = jnp.isfinite(value)
        if constraint.event_dim:
            finite = jnp.all(finite, axis=tuple(range(-constraint.event_dim, 0)))
        valid = valid & finite & jnp.asarray(constraint(value))
    operands = iter(law.operands())

    def feasible(value: jax.Array) -> jax.Array:
        name, _ = next(operands)
        constraint = (
            constraints.real if isinstance(law, DeltaLawSpec) else native.arg_constraints[name]
        )
        expanded = valid.reshape((*valid.shape, *((1,) * constraint.event_dim)))
        return jnp.where(expanded, value, constraint.feasible_like(value))

    return map_law(law, feasible), valid


def safe_native(law: Law[jax.Array]) -> tuple[dist.Distribution, jax.Array]:
    """Build a numerically feasible native law and separately return the original validity mask."""
    feasible, valid = feasible_law(law)
    return to_native(feasible), valid


def _masked_law_inputs(
    law: Law[jax.Array], values: jax.Array, mask: jax.Array
) -> tuple[Law[jax.Array], jax.Array, jax.Array, jax.Array]:
    feasible, valid = feasible_law(law)
    native = to_native(feasible)
    support = native.support
    assert support is not None
    valid = valid & jnp.isfinite(values) & jnp.asarray(support(values))
    if isinstance(law, GammaLawSpec):
        valid = valid & (values > 0.0)
    elif isinstance(law, BetaLawSpec):
        valid = valid & (values > 0.0) & (values < 1.0)
    observed = mask > 0.5
    safe_values = jnp.where(valid & observed, values, support.feasible_like(values))
    if support.is_discrete:
        safe_values = safe_values.astype(jnp.int32)
    return feasible, valid, observed, safe_values


def masked_law_log_prob(law: Law[jax.Array], values: jax.Array, mask: jax.Array) -> jax.Array:
    """Missing factors/derivatives are zero; invalid observed events have zero mass."""
    feasible, valid, observed, safe_values = _masked_law_inputs(law, values, mask)
    log_probs = to_native(feasible).log_prob(safe_values)
    total = jnp.sum(jnp.where(valid & observed, log_probs, 0.0))
    return jnp.where(jnp.any(observed & ~valid), -jnp.inf, total)


def prepare_response_log_prob(
    law: Law[jax.Array], values: jax.Array, mask: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Bind observed values and response-independent operands before particle mapping."""
    if not isinstance(law, (GammaLawSpec, NegativeBinomial2LawSpec)):
        return lambda response: masked_law_log_prob(with_response(law, response), values, mask)
    feasible, valid, observed, safe_values = _masked_law_inputs(law, values, mask)
    assert isinstance(feasible, (GammaLawSpec, NegativeBinomial2LawSpec))
    concentration = jnp.asarray(feasible.concentration)
    if isinstance(feasible, GammaLawSpec):
        normalizer = jax.scipy.special.gammaln(concentration)
        event_term = jax.scipy.special.xlogy(concentration - 1, safe_values)

        def factors(response: jax.Array) -> tuple[jax.Array, jax.Array]:
            response_valid = jnp.isfinite(response) & (response > 0)
            rate = concentration / jnp.where(response_valid, response, 1.0)
            rate_valid = jnp.isfinite(rate) & (rate > 0)
            rate = jnp.where(rate_valid, rate, 1.0)
            density = event_term - rate * safe_values - (normalizer - concentration * jnp.log(rate))
            return density, response_valid & rate_valid
    else:
        normalizer = _negative_binomial_normalizer(concentration, safe_values)

        def factors(response: jax.Array) -> tuple[jax.Array, jax.Array]:
            response_valid = jnp.isfinite(response) & (response >= 0)
            density = _negative_binomial_log_prob(
                jnp.where(response_valid, response, 1.0),
                concentration,
                safe_values,
                normalizer,
            )
            return density, response_valid

    def density(response: jax.Array) -> jax.Array:
        log_probs, response_valid = factors(jnp.where(observed, response, 1.0))
        feasible = valid & response_valid
        total = jnp.sum(jnp.where(feasible & observed, log_probs, 0.0))
        return jnp.where(jnp.any(observed & ~feasible), -jnp.inf, total)

    return density


def evaluate_law(
    law: Law[BoundExpression],
    predictor: jax.Array,
    scale: jax.Array,
    observed: jax.Array | None = None,
) -> Law[jax.Array]:
    """Evaluate each symbolic law operand at the supplied predictor, scale, and observation mask."""

    def evaluate(operand: BoundExpression) -> jax.Array:
        return operand.evaluate(predictor, scale, observed)

    return map_law(law, evaluate)


def law_moments(law: Law[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Actual mean and variance: undefined moments are NaN, divergent ones infinite."""
    native = to_native(law)
    mean = jnp.asarray(native.mean)
    if isinstance(law, StudentTLawSpec):
        # NumPyro reports +inf where the mean is undefined (df <= 1).
        mean = jnp.where(jnp.asarray(law.df) > 1.0, mean, jnp.nan)
    return mean, jnp.asarray(native.variance)


def law_response(law: Law[jax.Array]) -> jax.Array:
    """Declared response/location, kept separate from statistical moments."""
    if isinstance(law, StudentTLawSpec):
        return jnp.asarray(law.loc)
    if isinstance(law, (CategoricalLawSpec, OrderedLogisticLawSpec)):
        probabilities = category_probabilities(law)
        return jnp.sum(probabilities * jnp.arange(probabilities.shape[-1]), axis=-1)
    return jnp.asarray(to_native(law).mean)


def with_response(law: Law[jax.Array], response: jax.Array) -> Law[jax.Array]:
    """Reconstruct native operands after projecting a declared response trajectory."""
    match law:
        case DeltaLawSpec():
            return DeltaLawSpec(v=response)
        case NormalLawSpec():
            return NormalLawSpec(loc=response, scale=law.scale)
        case StudentTLawSpec():
            return StudentTLawSpec(df=law.df, loc=response, scale=law.scale)
        case PoissonLawSpec():
            return PoissonLawSpec(rate=response)
        case GammaLawSpec():
            valid = jnp.isfinite(response) & (response > 0.0)
            rate = law.concentration / jnp.where(valid, response, 1.0)
            return GammaLawSpec(
                concentration=law.concentration, rate=jnp.where(valid, rate, jnp.nan)
            )
        case BernoulliLogitsLawSpec() | BernoulliProbsLawSpec():
            return BernoulliProbsLawSpec(probs=response)
        case NegativeBinomial2LawSpec():
            return NegativeBinomial2LawSpec(mean=response, concentration=law.concentration)
        case BetaLawSpec():
            concentration = law.concentration1 + law.concentration0
            return BetaLawSpec(
                concentration1=response * concentration,
                concentration0=(1.0 - response) * concentration,
            )
        case OrderedLogisticLawSpec() | CategoricalLawSpec():
            raise ValueError("Category laws have no scalar interval-summary response")
    assert_never(law)


def gaussian_distribution(mean: jax.Array, covariance: jax.Array) -> dist.Distribution:
    """Construct a multivariate Gaussian after symmetrizing and jittering its covariance."""
    return dist.MultivariateNormal(mean, covariance_matrix=symmetrize_with_jitter(covariance))


def point_observation_scales(covariance: jax.Array) -> jax.Array:
    """Retain the mixed point sampler's marginal scale floor."""
    return jnp.sqrt(jnp.maximum(jnp.diag(covariance), NUMERICAL_EPSILON))


def _flatten_law(law: BaseModel) -> tuple[tuple[object, ...], tuple[str, ...]]:
    fields = tuple((name, value) for name, value in law if name != "distribution")
    return tuple(value for _, value in fields), tuple(name for name, _ in fields)


def _unflatten_law[L: BaseModel](
    law_type: type[L], names: tuple[str, ...], values: Iterable[object]
) -> L:
    return law_type(**dict(zip(names, values, strict=True)))


# Register only the existing constructors; operands are the pytree children.
for _law_type in OBSERVATION_LAW_TYPES:
    jax.tree_util.register_pytree_node(_law_type, _flatten_law, partial(_unflatten_law, _law_type))
