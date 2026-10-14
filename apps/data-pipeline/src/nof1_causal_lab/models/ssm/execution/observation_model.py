"""Compile constructor/layout groups for exact observation operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.artifacts.likelihood import (
    BernoulliLogitsLawSpec,
    BernoulliProbsLawSpec,
    CategoricalLawSpec,
    GammaLawSpec,
    Law,
    LinkFunction,
    NormalLawSpec,
    OrderedLogisticLawSpec,
    map_law,
)
from nof1_causal_lab.models.ssm.execution import emissions
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    evaluate_law,
    law_response,
    masked_law_log_prob,
    with_response,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.execution.observation_dispatch import (
        PointObservationSampler,
    )
    from nof1_causal_lab.models.ssm.execution.observation_operator import ObservationOperator
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

type EmissionLogProbFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
type ResponseFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class ObservationKernel:
    """Compiled exact observation density and response transformation."""

    log_prob_fn: EmissionLogProbFn
    response_fn: ResponseFn


@dataclass(frozen=True)
class LawGroup:
    """Compatible native operands with their original channel positions."""

    indices: tuple[int, ...]
    laws: ObservationLaws
    interval: bool = False

    def evaluate(
        self,
        predictors: jax.Array,
        scales: jax.Array,
        responses: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> Law[jax.Array]:
        """Evaluate compatible channel laws together, optionally substituting support-aggregated responses."""
        arrays: list[Law[jax.Array]] = []
        for index, law in zip(self.indices, self.laws, strict=True):
            eta = predictors[index]
            if responses is not None:
                eta = jnp.ones_like(eta) if isinstance(law, GammaLawSpec) else jnp.zeros_like(eta)
            observed = None if mask is None else mask[index] > 0.5
            native = evaluate_law(law, eta, scales[index], observed)
            if responses is not None:
                response = responses[index]
                if observed is not None:
                    response = jnp.where(observed, response, 1.0)
                native = with_response(native, response)
            arrays.append(native)
        operands = tuple(iter(value for _, value in law.operands()) for law in arrays)
        return map_law(
            arrays[0], lambda _value: jnp.stack(tuple(next(values) for values in operands))
        )


def compile_law_groups(
    laws: ObservationLaws,
    indices: Sequence[int] | None = None,
    *,
    interval: bool = False,
    roles: Sequence[str] | None = None,
) -> tuple[LawGroup, ...]:
    """Group observation laws by compatible family, event shape, support role, and response link."""
    selected = range(len(laws)) if indices is None else indices
    groups: dict[tuple[type, int, str, LinkFunction], list[int]] = {}
    for index in selected:
        law = laws[index]
        constructor = type(law)
        event_size = 0
        if isinstance(law, OrderedLogisticLawSpec):
            event_size = law.cutpoints.event_size
        elif isinstance(law, CategoricalLawSpec):
            event_size = law.logits.event_size
        if interval and isinstance(law, (BernoulliLogitsLawSpec, BernoulliProbsLawSpec)):
            constructor = BernoulliProbsLawSpec
        role = "point" if roles is None else roles[index]
        if isinstance(law, NormalLawSpec):
            role = "gaussian_block"
        groups.setdefault((constructor, event_size, role, _law_link(law)), []).append(index)
    return tuple(
        LawGroup(tuple(positions), tuple(laws[index] for index in positions), interval)
        for positions in groups.values()
    )


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


def _law_link(law: Law[BoundExpression]) -> LinkFunction:
    """Every operand of a bound law carries the law's parsed link."""
    return law.operands()[0][1].link


def _group_density(
    group: LawGroup,
    y: jax.Array,
    predictors: jax.Array,
    R: jax.Array,
    mask: jax.Array,
    responses: jax.Array | None = None,
) -> jax.Array:
    indices = jnp.asarray(group.indices)
    law = group.evaluate(predictors, jnp.sqrt(jnp.diag(R)), responses, mask)
    if isinstance(law, NormalLawSpec):
        return emissions.gaussian_block_log_prob(
            y[indices], law.loc, R[jnp.ix_(indices, indices)], mask[indices]
        )
    return masked_law_log_prob(law, y[indices], mask[indices])


def compile_observation_model(
    laws: ObservationLaws,
    *,
    manifest_cov: jax.Array,
    observation_support: ObservationSupportRuntime | None = None,
) -> CompiledObservationModel:
    """Compile exact laws, shared response projection, and observation sampling."""
    from .observation_dispatch import build_point_observation_sampler

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
        return sum(
            (_group_density(group, y, eta, R, mask) for group in groups),
            jnp.zeros((), dtype=y.dtype),
        )

    def response(eta: jax.Array) -> jax.Array:
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
        return sum(
            (
                _group_density(group, y, jnp.zeros_like(means), R, mask, means)
                for group in interval_groups
            ),
            jnp.zeros((), dtype=y.dtype),
        )

    return CompiledObservationModel(
        kernel,
        point_sampler,
        mean_density,
        operator,
    )
