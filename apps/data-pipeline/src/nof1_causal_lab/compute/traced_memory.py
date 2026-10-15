"""Arithmetic workspace estimates from unfused JAX array lifetimes."""

from __future__ import annotations

from functools import cache
from itertools import accumulate
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax.extend.core import AbstractToken, Var, jaxprs_in_params

from nof1_causal_lab.compute.budget import MemoryDemand

if TYPE_CHECKING:
    from jax.core import AbstractValue, ShapedArray
    from jax.extend.core import Jaxpr


def traced_memory(graph: Jaxpr) -> MemoryDemand:
    """Estimate live array bytes without lowering, compiling, or executing code.

    Count arguments and outputs once, release temporary arrays after their last
    use, and take the largest nested workspace. Loop bodies and conditional
    branches reuse workspace; their full stacked outputs remain resident.
    Array shapes already include all vectorized chains and parameter labels.

    This deliberately gives no credit for fusion, aliases, or buffer donation.
    It estimates array storage, not backend-specific workspaces: the selected
    executable must still pass the compiler's memory check before execution.
    Caches are scoped to this estimate so traced graphs are not retained.
    """

    @cache
    def array_bytes(aval: AbstractValue) -> int:
        if isinstance(aval, AbstractToken):
            return 0
        array = cast("ShapedArray", aval)
        shape = jax.ShapeDtypeStruct(array.shape, array.dtype)
        if jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
            shape = jax.eval_shape(jax.random.key_data, shape)
        return int(shape.size * shape.dtype.itemsize)

    @cache
    def temporary_bytes(current: Jaxpr) -> int:
        external = frozenset(
            value for value in (*current.invars, *current.outvars) if isinstance(value, Var)
        )
        constants = frozenset(current.constvars)
        temporary = (
            frozenset(value for equation in current.eqns for value in equation.outvars)
            - external
            - constants
        )
        last_use = {
            value: index
            for index, equation in enumerate(current.eqns)
            for value in (*equation.invars, *equation.outvars)
            if isinstance(value, Var) and value in temporary
        }
        allocations = tuple(
            sum(array_bytes(value.aval) for value in equation.outvars if value in temporary)
            for equation in current.eqns
        )
        releases = tuple(
            sum(
                array_bytes(value.aval)
                for value in frozenset(
                    value
                    for value in (*equation.invars, *equation.outvars)
                    if isinstance(value, Var)
                )
                if value in temporary and last_use[value] == index
            )
            for index, equation in enumerate(current.eqns)
        )
        live = tuple(
            accumulate(
                (
                    allocated - released
                    for allocated, released in zip(allocations, releases, strict=True)
                ),
                initial=0,
            )
        )
        nested = tuple(
            max((temporary_bytes(child) for child in jaxprs_in_params(equation.params)), default=0)
            for equation in current.eqns
        )
        peak = max(
            (
                live[index] + allocated + workspace
                for index, (allocated, workspace) in enumerate(
                    zip(allocations, nested, strict=True)
                )
            ),
            default=0,
        )
        return sum(array_bytes(value.aval) for value in constants) + peak

    external = frozenset(
        value for value in (*graph.invars, *graph.outvars) if isinstance(value, Var)
    )
    return MemoryDemand(
        host_bytes=0,
        device_bytes=sum(array_bytes(value.aval) for value in external) + temporary_bytes(graph),
    )
