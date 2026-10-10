"""One native-law interpreter and exact observation boundary adapters."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, assert_never, override

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from pydantic import BaseModel

from nof1_causal_lab.artifacts.likelihood import (
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
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.execution.contracts import NUMERICAL_EPSILON

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jax.typing import ArrayLike

    from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression


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
        zero_mean = self.response_mean == 0.0
        logits = _positive_log(jnp.where(zero_mean, 1.0, self.response_mean)) - _positive_log(
            self.concentration
        )
        positive_count = dist.NegativeBinomialLogits(self.concentration, logits).log_prob(value)
        positive_count = jnp.where(zero_mean, -jnp.inf, positive_count)
        # The boundary expression preserves the exact right derivative -1 at
        # mean zero; positive means use log space without reciprocal-rate terms.
        boundary = -self.concentration * jnp.log1p(
            jnp.where(zero_mean, self.response_mean, 0.0) / self.concentration
        )
        zero_count = jnp.where(zero_mean, boundary, -self.concentration * jax.nn.softplus(logits))
        return jnp.where(jnp.asarray(value) == 0, zero_count, positive_count)

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


def masked_law_log_prob(law: Law[jax.Array], values: jax.Array, mask: jax.Array) -> jax.Array:
    """Missing factors/derivatives are zero; invalid observed events have zero mass."""
    native, valid = safe_native(law)
    support = native.support
    assert support is not None
    valid = valid & jnp.isfinite(values) & support(values)
    if isinstance(law, GammaLawSpec):
        valid = valid & (values > 0.0)
    elif isinstance(law, BetaLawSpec):
        valid = valid & (values > 0.0) & (values < 1.0)
    observed = mask > 0.5
    safe_values = jnp.where(valid & observed, values, support.feasible_like(values))
    if support.is_discrete:
        safe_values = safe_values.astype(jnp.int32)
    log_probs = native.log_prob(safe_values)
    total = jnp.sum(jnp.where(valid & observed, log_probs, 0.0))
    return jnp.where(jnp.any(observed & ~valid), -jnp.inf, total)


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
