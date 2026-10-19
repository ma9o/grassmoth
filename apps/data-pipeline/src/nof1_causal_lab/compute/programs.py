"""Explicit compilation dependencies for pure numerical setup programs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import jax
from jax.stages import Compiled


@dataclass(frozen=True)
class PreparedProgram[Result]:
    """A callable with bound operands and its executable for memory accounting."""

    executable: Compiled
    call: Callable[..., Result]


class ProgramCompiler(Protocol):
    """Compile a pure function under an explicitly supplied execution policy."""

    def __call__[Result](
        self, function: Callable[..., Result], *args: object, **kwargs: object
    ) -> PreparedProgram[Result]:
        """Bind one program without executing its numerical computation."""
        ...


def compile_jit[Result](
    function: Callable[..., Result], *args: object, **kwargs: object
) -> PreparedProgram[Result]:
    """Compile a numerical program directly without persistent staging effects."""
    executable = jax.jit(function).lower(*args, **kwargs).compile()
    return PreparedProgram(executable, cast("Callable[..., Result]", executable))
