"""Compile constructor/layout groups for exact observation operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from grassmoth.artifacts.model.expressions import expression_states
from grassmoth.artifacts.model.likelihood import (
    CategoricalLawSpec,
    GammaLawSpec,
    NegativeBinomial2LawSpec,
    NormalLawSpec,
    OrderedLogisticLawSpec,
    PoissonLawSpec,
    StudentTLawSpec,
)
from grassmoth.models.ssm.execution import emissions
from grassmoth.models.ssm.execution.observation_distributions import (
    FixedLawTerms,
    fixed_law_log_prob,
    law_response,
    masked_law_log_prob,
    prepare_fixed_law_terms,
    prepare_response_log_prob,
)
from grassmoth.models.ssm.execution.observation_groups import LawGroup, compile_law_groups
from grassmoth.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from grassmoth.models.ssm.dynamics.expression import BoundExpression
    from grassmoth.models.ssm.execution.contracts import ObservationLaws
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.ssm.execution.gaussian_noise import GaussianNoiseFactors
    from grassmoth.models.ssm.execution.observation_dispatch import PointObservationSampler
    from grassmoth.models.ssm.execution.observation_operator import ObservationOperator
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime

type EmissionLogProbFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
type ResponseFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class ObservationKernel:
    """Compiled exact observation density and response transformation."""

    log_prob_fn: EmissionLogProbFn
    response_fn: ResponseFn


@dataclass(frozen=True)
class CompiledObservationModel:
    """Observation execution functions with point sampling and optional interval-support handling."""

    kernel: ObservationKernel
    point_sampler: PointObservationSampler
    mean_log_prob_fn: EmissionLogProbFn | None
    observation_operator: ObservationOperator | None

    @property
    def requires_interval_summary_handling(self) -> bool:
        """Whether any compiled observation requires accumulation across an interval."""
        return (
            self.observation_operator is not None
            and self.observation_operator.requires_interval_summary_handling
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ObservationDensityFactors:
    """Gaussian noise and fixed scalar event terms on the same observation grid."""

    gaussian: tuple[GaussianNoiseFactors, ...]
    scalar: tuple[FixedLawTerms | None, ...]

    def at(self, index: jax.Array | int) -> ObservationDensityFactors:
        """Select one time while preserving the static law grouping."""
        return cast("ObservationDensityFactors", jax.tree.map(lambda value: value[index], self))


def _group_density(
    group: LawGroup,
    y: jax.Array,
    predictors: jax.Array,
    R: jax.Array,
    mask: jax.Array,
    responses: jax.Array | None = None,
    log_precision: jax.Array | None = None,
) -> jax.Array:
    indices = jnp.asarray(group.indices)
    law = group.evaluate(predictors, jnp.sqrt(jnp.diag(R)), responses, mask)
    if isinstance(law, StudentTLawSpec) and log_precision is not None:
        return precision_normal_log_prob(law, y[indices], mask[indices], log_precision[indices])
    if isinstance(law, NormalLawSpec):
        return emissions.gaussian_block_log_prob(
            y[indices], law.loc, R[jnp.ix_(indices, indices)], mask[indices]
        )
    return masked_law_log_prob(law, y[indices], mask[indices])


def precision_normal_log_prob(
    law: StudentTLawSpec[jax.Array],
    y: jax.Array,
    mask: jax.Array,
    log_precision: jax.Array,
) -> jax.Array:
    """Conditional scalar Gaussian factors, without materializing tiny precisions.

    These independent factors never join the native correlated Normal blocks.
    The StudentT scale is a scale parameter, including when its variance is undefined.
    """
    log_inverse_scale = 0.5 * log_precision - jnp.log(law.scale)
    residual = jnp.where(mask, y - law.loc, 0.0) * jnp.exp(log_inverse_scale)
    density = log_inverse_scale - 0.5 * (jnp.log(2 * jnp.pi) + residual**2)
    return jnp.sum(jnp.where(mask, density, 0.0))


def grouped_density(
    groups: tuple[LawGroup, ...],
    y: jax.Array,
    eta: jax.Array,
    R: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Evaluate an already-owned observation grouping with explicit numerical inputs."""
    return sum(
        (_group_density(group, y, eta, R, mask) for group in groups),
        jnp.zeros((), dtype=y.dtype),
    )


def prepare_grouped_noise(
    groups: tuple[LawGroup, ...], covariance: jax.Array, observations: jax.Array
) -> tuple[GaussianNoiseFactors, ...]:
    """Prepare each Gaussian group's noise for every recorded missingness pattern."""

    def prepare(group: LawGroup) -> GaussianNoiseFactors:
        indices = jnp.asarray(group.indices, dtype=jnp.int32)
        return jax.vmap(
            lambda mask: emissions.prepare_gaussian_block_noise(
                covariance[jnp.ix_(indices, indices)], mask[indices]
            )
        )(~jnp.isnan(observations))

    return tuple(prepare(group) for group in groups if isinstance(group.laws[0], NormalLawSpec))


def prepare_grouped_density(
    groups: tuple[LawGroup, ...], covariance: jax.Array, observations: jax.Array
) -> ObservationDensityFactors:
    """Hoist only event terms whose authored concentration has no state operands."""
    scales = jnp.sqrt(jnp.diag(covariance))

    def prepare(group: LawGroup) -> FixedLawTerms | None:
        first = group.laws[0]
        if isinstance(first, (GammaLawSpec, NegativeBinomial2LawSpec)):
            if any(
                expression_states(
                    cast(
                        "GammaLawSpec[BoundExpression] | NegativeBinomial2LawSpec[BoundExpression]",
                        law,
                    ).concentration.expression
                )
                for law in group.laws
            ):
                return None
        elif not isinstance(first, PoissonLawSpec):
            return None
        indices = jnp.asarray(group.indices, dtype=jnp.int32)

        def terms(values: jax.Array) -> FixedLawTerms:
            observed = ~jnp.isnan(values)
            if isinstance(first, PoissonLawSpec):
                law = PoissonLawSpec(rate=jnp.ones(len(group.indices), dtype=values.dtype))
            else:
                concentration = jnp.stack(
                    tuple(
                        cast(
                            "GammaLawSpec[BoundExpression] | NegativeBinomial2LawSpec[BoundExpression]",
                            bound,
                        ).concentration.evaluate(
                            jnp.ones((), dtype=values.dtype), scales[index], observed[index]
                        )
                        for index, bound in zip(group.indices, group.laws, strict=True)
                    )
                )
                law = (
                    GammaLawSpec(concentration=concentration, rate=jnp.ones_like(concentration))
                    if isinstance(first, GammaLawSpec)
                    else NegativeBinomial2LawSpec(
                        concentration=concentration, mean=jnp.ones_like(concentration)
                    )
                )
            return prepare_fixed_law_terms(
                law, jnp.nan_to_num(values, nan=0.0)[indices], observed[indices]
            )

        return jax.vmap(terms)(observations)

    return ObservationDensityFactors(
        prepare_grouped_noise(groups, covariance, observations),
        tuple(prepare(group) for group in groups),
    )


def grouped_density_with_noise(
    model: HeterogeneousObservation,
    y: jax.Array,
    eta: jax.Array,
    noise: ObservationDensityFactors,
    log_precision: jax.Array | None = None,
) -> jax.Array:
    """Use prepared factors without changing any predictor-dependent law operands."""
    groups, covariance = model.density_groups(), model.measurement.manifest_cov
    observed = ~jnp.isnan(y)
    filled = jnp.nan_to_num(y, nan=0.0)
    factors = iter(noise.gaussian)
    total = jnp.zeros((), dtype=eta.dtype)
    for group, terms in zip(groups, noise.scalar, strict=True):
        if isinstance(group.laws[0], NormalLawSpec):
            indices = jnp.asarray(group.indices, dtype=jnp.int32)
            law = cast(
                "NormalLawSpec[jax.Array]",
                group.evaluate(eta, jnp.sqrt(jnp.diag(covariance)), mask=observed),
            )
            residual = jnp.where(observed[indices], filled[indices] - law.loc, 0.0)
            total = total + next(factors).log_prob(residual)
        elif terms is not None:
            indices = jnp.asarray(group.indices, dtype=jnp.int32)
            law = group.evaluate(eta, jnp.sqrt(jnp.diag(covariance)), mask=observed)
            assert isinstance(law, (GammaLawSpec, NegativeBinomial2LawSpec, PoissonLawSpec))
            total = total + fixed_law_log_prob(law, filled[indices], observed[indices], terms)
        else:
            total = total + _group_density(
                group, filled, eta, covariance, observed, log_precision=log_precision
            )
    return total


def grouped_response(
    groups: tuple[LawGroup, ...], eta: jax.Array, manifest_cov: jax.Array
) -> jax.Array:
    """Project declared responses using the same grouping as the observation density."""
    values = jnp.zeros_like(eta)
    for group in groups:
        native = group.evaluate(eta, jnp.sqrt(jnp.diag(manifest_cov)))
        if isinstance(native, (CategoricalLawSpec, OrderedLogisticLawSpec)):
            declared = law_response(native)
        else:
            declared = jnp.stack(
                tuple(
                    next(iter(law.operands()))[1].response_fn(eta[index])
                    for index, law in zip(group.indices, group.laws, strict=True)
                )
            )
        values = values.at[jnp.asarray(group.indices)].set(declared)
    return values


def prepare_mean_density(
    groups: tuple[LawGroup, ...],
    y: jax.Array,
    R: jax.Array,
    mask: jax.Array,
    log_precision: jax.Array | None = None,
) -> ResponseFn:
    """Bind one row's exact noise and scalar laws before mapping candidate means."""
    noise = prepare_grouped_noise(groups, R, jnp.where(mask > 0.5, y, jnp.nan)[None])
    return prepare_mean_density_with_noise(
        groups, y, R, mask, tuple(factor.at(0) for factor in noise), log_precision
    )


def prepare_mean_density_with_noise(
    groups: tuple[LawGroup, ...],
    y: jax.Array,
    R: jax.Array,
    mask: jax.Array,
    noise: tuple[GaussianNoiseFactors, ...],
    log_precision: jax.Array | None = None,
) -> ResponseFn:
    """Bind each interval law before mapping across candidate response means."""
    factors = iter(noise)

    def bind(group: LawGroup) -> ResponseFn:
        if isinstance(group.laws[0], StudentTLawSpec) and log_precision is not None:
            return lambda means: _group_density(
                group, y, jnp.zeros_like(means), R, mask, means, log_precision
            )
        if isinstance(group.laws[0], NormalLawSpec):
            factor = next(factors)
            indices = jnp.asarray(group.indices, dtype=jnp.int32)
            return lambda means: factor.log_prob(
                jnp.where(mask[indices] > 0.5, y[indices] - means[indices], 0.0)
            )
        indices = jnp.asarray(group.indices)
        law = group.evaluate(jnp.zeros_like(y), jnp.sqrt(jnp.diag(R)), jnp.ones_like(y), mask)
        score = prepare_response_log_prob(law, y[indices], mask[indices])
        return lambda means: score(means[indices])

    scores = tuple(bind(group) for group in groups)
    return lambda means: sum((score(means) for score in scores), jnp.zeros((), dtype=y.dtype))


def compile_observation_model(
    laws: ObservationLaws,
    *,
    manifest_cov: jax.Array,
    observation_support: ObservationSupportRuntime | None = None,
) -> CompiledObservationModel:
    """Compile exact laws, shared response projection, and observation sampling."""
    from grassmoth.models.ssm.execution.observation_dispatch import (
        build_point_observation_sampler,
    )

    if manifest_cov.shape != (len(laws), len(laws)):
        raise ValueError("manifest_cov must match the compiled observation law axes")
    if observation_support is not None and len(observation_support.support_kinds) != len(laws):
        raise ValueError("observation_support must match the compiled observation law axes")
    roles = (
        None
        if observation_support is None
        else tuple(str(kind) for kind in observation_support.support_kinds)
    )
    groups = compile_law_groups(laws, roles=roles)

    def density(y: jax.Array, eta: jax.Array, R: jax.Array, mask: jax.Array) -> jax.Array:
        return grouped_density(groups, y, eta, R, mask)

    def response(eta: jax.Array) -> jax.Array:
        return grouped_response(groups, eta, manifest_cov)

    operator = compile_observation_operator(observation_support)
    kernel = ObservationKernel(density, response)
    point_sampler = build_point_observation_sampler(laws, manifest_cov, groups=groups)
    if operator is None or not operator.requires_interval_summary_handling:
        return CompiledObservationModel(kernel, point_sampler, None, operator)
    interval_groups = compile_law_groups(laws, operator.interval_summary_indices, interval=True)
    if any(
        isinstance(group.laws[0], (CategoricalLawSpec, OrderedLogisticLawSpec))
        for group in interval_groups
    ):
        raise ValueError("Category laws have no scalar interval-summary response")

    def mean_density(y: jax.Array, means: jax.Array, R: jax.Array, mask: jax.Array) -> jax.Array:
        return prepare_mean_density(interval_groups, y, R, mask)(means)

    return CompiledObservationModel(
        kernel,
        point_sampler,
        mean_density,
        operator,
    )
