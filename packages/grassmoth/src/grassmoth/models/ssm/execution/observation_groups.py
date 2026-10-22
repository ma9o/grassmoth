"""Native law groups shared by exact observation density and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.artifacts.model.likelihood import (
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
from grassmoth.models.ssm.execution.observation_distributions import (
    evaluate_law,
    with_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grassmoth.models.ssm.dynamics.expression import BoundExpression
    from grassmoth.models.ssm.execution.contracts import ObservationLaws


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


def _law_link(law: Law[BoundExpression]) -> LinkFunction:
    """Every operand of a bound law carries the law's parsed link."""
    return law.operands()[0][1].link
