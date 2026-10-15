"""Compile the independent coordinates of a possibly degenerate state process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sympy import Float, Integer, Poly

from nof1_causal_lab.artifacts.model.expressions import (
    CoefficientExpression,
    StateExpression,
    symbolic_expression,
)

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.structure.blocks import DiffusionBlockSpec, T0CholBlockSpec


@dataclass(frozen=True)
class ParallelRecurrences:
    """Coordinates whose exact recurrence is affine in the deterministic subsystem."""

    indices: tuple[int, ...]


@dataclass(frozen=True)
class SequentialRecurrences:
    """Coordinates without a compiled finite-dimensional associative recurrence."""

    indices: tuple[int, ...]


@dataclass(frozen=True)
class StateSpace:
    """Gaussian density coordinates; the remaining coordinates are exact recurrences."""

    size: int
    inputs: tuple[int, ...]
    initial: tuple[int, ...]
    transition: tuple[int, ...]
    static: tuple[int, ...]
    recurrence: ParallelRecurrences | SequentialRecurrences

    @property
    def initial_path_coordinates(self) -> tuple[int, ...]:
        """Initial random coordinates after promoting static states to global variables."""
        return tuple(index for index in self.initial if index not in self.static)

    @property
    def replayed_coordinates(self) -> tuple[int, ...]:
        """Endogenous coordinates reconstructed initially or at later time points."""
        return tuple(
            index
            for index in range(self.size)
            if index not in (*self.inputs, *self.static)
            and (index not in self.initial or index not in self.transition)
        )

    @property
    def requires_replay(self) -> bool:
        """Whether endogenous coordinates must be reconstructed along a whole path."""
        return bool(self.replayed_coordinates)


def compile_state_space(
    diffusion: DiffusionBlockSpec,
    initial: T0CholBlockSpec,
    static_loadings: jax.Array,
    inputs: tuple[int, ...],
    static: tuple[int, ...],
    dynamics: DynamicsSpec,
) -> StateSpace:
    """Resolve support from declared fixed values and free coefficient positions once."""
    initial_random = np.any(np.asarray(initial.template) != 0, axis=1) | np.asarray(
        initial.diag_support
    )
    initial_random |= np.any(np.asarray(static_loadings) != 0, axis=1)
    transition_random = np.any(
        (np.asarray(diffusion.diffusion_chol_template) != 0)
        | np.asarray(diffusion.diffusion_chol_support),
        axis=1,
    )
    transition = tuple(int(i) for i in np.flatnonzero(transition_random) if i not in inputs)
    deterministic = tuple(
        i for i in range(diffusion.n_latent) if i not in (*inputs, *static, *transition)
    )
    return StateSpace(
        size=diffusion.n_latent,
        inputs=inputs,
        initial=tuple(int(i) for i in np.flatnonzero(initial_random) if i not in inputs),
        transition=transition,
        static=static,
        recurrence=_compile_recurrence(dynamics, deterministic),
    )


def _compile_recurrence(
    dynamics: DynamicsSpec, indices: tuple[int, ...]
) -> ParallelRecurrences | SequentialRecurrences:
    """Prove affine closure symbolically; numerical execution never linearizes a recurrence."""
    for component in dynamics.components:
        if component.target not in indices:
            continue
        symbolic = symbolic_expression(component.expression)
        expression = symbolic.root.subs(
            {
                symbol: operand.value
                for symbol, operand in symbolic.operands.items()
                if isinstance(operand, CoefficientExpression)
                and isinstance(operand.value, (int, float))
            }
        )
        # Authored scalar literals are floats. An exact integral exponent such
        # as 2.0 must remain a polynomial for the symbolic affine-closure proof.
        expression = expression.subs(
            {
                number: Integer(int(number))
                for number in expression.atoms(Float)
                if float(number).is_integer()
            }
        )
        variables = tuple(
            symbol
            for symbol, operand in symbolic.operands.items()
            if isinstance(operand, StateExpression)
            and component.state_index[operand.construct_id] in indices
        )
        if component.kind == "potential":
            own = next(
                (
                    symbol
                    for symbol, operand in symbolic.operands.items()
                    if isinstance(operand, StateExpression)
                    and component.state_index[operand.construct_id] == component.target
                ),
                None,
            )
            if own is None:
                continue
            expression = -expression.diff(own)
        if variables and (
            not expression.is_polynomial(*variables)
            or Poly(expression, *variables).total_degree() > 1
        ):
            return SequentialRecurrences(indices)
    return ParallelRecurrences(indices)
