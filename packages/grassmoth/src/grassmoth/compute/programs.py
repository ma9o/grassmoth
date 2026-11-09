"""Explicit compilation dependencies for pure numerical setup programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled
    from jax.typing import DTypeLike

    from grassmoth.compute.budget import MemoryDemand


type ProgramShapes = Mapping[jax.tree_util.KeyPath, jax.ShapeDtypeStruct]


def scalar_operand(*, value: bool | int | float, dtype: DTypeLike | None = None) -> jax.Array:
    """Keep scalar controls as runtime operands until traced code selects ``[0]``.

    JAX embeds closed-over zero-dimensional arrays as literals. A one-element
    array is captured by closure conversion instead, so its value can change
    without changing the numerical program.
    """
    return jnp.asarray([value], dtype=dtype)


@dataclass(frozen=True)
class PreparedProgram[**P, Result]:
    """A callable with bound operands and its executable for memory accounting."""

    executable: Compiled
    call: Callable[P, Result]


@runtime_checkable
class ProgramCompiler(Protocol):
    """Compile a pure function under an explicitly supplied execution policy."""

    def __call__[**P, Result](
        self, function: Callable[P, Result], *args: P.args, **kwargs: P.kwargs
    ) -> PreparedProgram[P, Result]:
        """Bind one program without executing its numerical computation."""
        ...


@runtime_checkable
class ProgramPreparer(Protocol):
    """Prepare a JAX program and expose its sizing evidence."""

    def __call__(
        self, function: Callable[..., object], *args: object, **kwargs: object
    ) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], ProgramShapes, MemoryDemand]:
        """Trace abstract or concrete operands and return output leaves keyed by their tree paths."""
        ...
