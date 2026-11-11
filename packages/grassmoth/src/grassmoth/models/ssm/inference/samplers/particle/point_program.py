"""Compile exact native point expressions and statically select covered GPU programs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.typing import ArrayLike, DTypeLike

    from grassmoth.models.ssm.inference.targets.sequential import SequentialObservationRuntime

from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from pydantic import ConfigDict

from grassmoth.artifacts.base import Value
from grassmoth.models.ssm.inference.samplers.particle import scalar_program as scalar

type PointValue = jax.Array | tuple[jax.Array, ...]


@runtime_checkable
class PointOperandReference(Protocol):
    """Read-only Pallas operand interface used during kernel tracing."""

    def __getitem__(self, index: tuple[jax.Array | int, ...]) -> jax.Array:
        """Load one scalar from the captured operand."""
        ...


class ExistingPointKernels(Value):
    """Keep the current evaluator for a model outside the compiled point support."""

    reason: tuple[str, ...]


class CompiledPointProgram[Result: PointValue](Value):
    """Original state arithmetic, logical input shape and time-indexed operands."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_variable: core.Var
    operands: tuple[core.Var, ...]
    equations: tuple[core.JaxprEqn, ...]
    outputs: tuple[core.Var | core.Literal, ...]
    buffers: tuple[jax.Array, ...]
    output_is_tuple: bool

    def evaluate(self, state: jax.Array, values: tuple[scalar.ScalarTensor, ...]) -> Result:
        """Evaluate original model arithmetic on logical coordinates, then pad outputs."""
        environment = {
            **dict(zip(self.operands, values, strict=True)),
            self.state_variable: scalar.unpack(state, scalar.array_aval(self.state_variable).shape),
        }
        for equation in self.equations:
            environment = scalar.evaluate(equation, environment)
        result = tuple(
            scalar.pack(scalar.literal(v.val) if isinstance(v, core.Literal) else environment[v])
            for v in self.outputs
        )
        return cast("Result", result if self.output_is_tuple else result[0])

    def load(
        self, references: tuple[PointOperandReference, ...], time: jax.Array | int, chain: jax.Array
    ) -> tuple[scalar.ScalarTensor, ...]:
        """Load logical operands for one chain and time without reading padding."""
        return tuple(
            scalar.tensor(
                scalar.array_aval(variable).shape,
                (
                    reference[(time, chain, *index)]
                    for index in np.ndindex(scalar.array_aval(variable).shape)
                ),
            )
            for reference, variable in zip(references, self.operands, strict=True)
        )


class CompiledPointKernels(Value):
    """One admitted set of exact model expressions shared by both time loops."""

    observation: CompiledPointProgram[jax.Array]
    transition: CompiledPointProgram[jax.Array]
    state: CompiledPointProgram[jax.Array]


type PointKernelRoute = CompiledPointKernels | ExistingPointKernels


def _read(
    environment: Mapping[core.Var, ArrayLike], variable: core.Var | core.Literal
) -> ArrayLike:
    return variable.val if isinstance(variable, core.Literal) else environment[variable]


def inspect_point_program[Result: PointValue](
    function: Callable[[jax.Array, jax.Array], Result],
    dimension: int,
    steps: int,
    dtype: DTypeLike,
) -> CompiledPointProgram[Result] | ExistingPointKernels:
    """Partition by exact state dependence and reject unsupported dynamic primitives."""
    with jax.disable_jit():
        program, shape = jax.make_jaxpr(function, return_shape=True)(
            jax.ShapeDtypeStruct((dimension,), dtype), jax.ShapeDtypeStruct((), jnp.int32)
        )
    state_variable, time_variable = program.invars
    dependent = frozenset((state_variable,))
    dynamic: tuple[core.JaxprEqn, ...] = ()
    invariant: tuple[core.JaxprEqn, ...] = ()
    for equation in program.eqns:
        if any(not isinstance(v, core.Literal) and v in dependent for v in equation.invars):
            dependent = dependent | frozenset(equation.outvars)
            dynamic = (*dynamic, equation)
        else:
            invariant = (*invariant, equation)
    unsupported = tuple(
        sorted(
            {
                reason
                for equation in dynamic
                if (reason := scalar.unsupported_primitive(equation)) is not None
            }
        )
    )
    if unsupported:
        return ExistingPointKernels(reason=unsupported)
    external: tuple[core.Var, ...] = ()
    for variable in (*tuple(v for eqn in dynamic for v in eqn.invars), *program.outvars):
        if (
            not isinstance(variable, core.Literal)
            and variable not in dependent
            and variable not in external
        ):
            external = (*external, variable)

    def prepare(time: jax.Array) -> tuple[jax.Array, ...]:
        environment = {
            **dict(zip(program.constvars, program.consts, strict=True)),
            time_variable: time,
        }
        for equation in invariant:
            # These are original native operations outside Pallas, not scalar substitutions.
            values = equation.primitive.bind(
                *(_read(environment, v) for v in equation.invars),
                **equation.primitive.get_bind_params(equation.params),
            )
            values = values if equation.primitive.multiple_results else (values,)
            environment = {**environment, **dict(zip(equation.outvars, values, strict=True))}
        return cast("tuple[jax.Array, ...]", tuple(environment[v] for v in external))

    buffers = jax.vmap(prepare)(jnp.arange(steps, dtype=jnp.int32))
    # Empty logical operands are never read; Pallas CPU blocks need nonzero extents.
    buffers = tuple(
        buffer if buffer.size else jnp.zeros(tuple(max(1, n) for n in buffer.shape), buffer.dtype)
        for buffer in buffers
    )
    return CompiledPointProgram[Result](
        state_variable=state_variable,
        operands=external,
        equations=dynamic,
        outputs=tuple(program.outvars),
        buffers=tuple(buffers),
        output_is_tuple=isinstance(shape, tuple),
    )


def existing_point_kernels() -> ExistingPointKernels:
    """Keep native evaluation for a model or workload outside the compiled route."""
    return ExistingPointKernels(reason=("existing-model-or-workload-route",))


def compile_point_kernels(
    memory: SequentialObservationRuntime,
    steps: int,
    dtype: DTypeLike,
) -> PointKernelRoute:
    """Resolve coverage once; unsupported expressions keep their existing typed route."""
    observation = inspect_point_program(memory.log_probs, memory.dimension, steps, dtype)
    transition = inspect_point_program(
        lambda value, index: jax.vmap(
            lambda context: memory.target.transition_moments(context, memory.state(value), index)[0]
        )(memory.contexts),
        memory.dimension,
        steps,
        dtype,
    )
    state = inspect_point_program(
        lambda value, _index: memory.state(value), memory.dimension, steps, dtype
    )
    programs = (observation, transition, state)
    failures = tuple(
        reason for p in programs if isinstance(p, ExistingPointKernels) for reason in p.reason
    )
    if failures:
        return ExistingPointKernels(reason=tuple(sorted(set(failures))))
    assert isinstance(observation, CompiledPointProgram)
    assert isinstance(transition, CompiledPointProgram)
    assert isinstance(state, CompiledPointProgram)
    return CompiledPointKernels(observation=observation, transition=transition, state=state)
