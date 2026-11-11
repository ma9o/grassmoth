"""Compile small native tensor expressions as scalar SSA for Pallas.

Python object arrays describe compiler structure only. All numerical values remain
JAX scalar tracers and execute on the GPU. This avoids unsupported slice/gather
lowerings and non-power-of-two temporary vectors without padding model states.
"""

from collections.abc import Iterable, Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jax.extend import core
from numpy.typing import NDArray

type ScalarTensor = NDArray[np.object_]
ELEMENTWISE_PRIMITIVES = frozenset(
    (
        "abs",
        "acos",
        "acosh",
        "add",
        "and",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bitcast_convert_type",
        "cbrt",
        "ceil",
        "clamp",
        "convert_element_type",
        "cos",
        "cosh",
        "div",
        "eq",
        "erf_inv",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "ge",
        "gt",
        "integer_pow",
        "is_finite",
        "le",
        "log",
        "log1p",
        "logistic",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "pow",
        "rem",
        "rsqrt",
        "select_n",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "one_minus_square",
        "stop_gradient",
        "sub",
        "tan",
        "tanh",
        "xor",
    )
)
STRUCTURAL_PRIMITIVES = frozenset(
    (
        "split",
        "slice",
        "reshape",
        "squeeze",
        "transpose",
        "broadcast_in_dim",
        "stack",
        "concatenate",
        "iota",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_or",
        "reduce_and",
        "dot_general",
        "gather",
        "dynamic_slice",
        "pad",
        "unstack",
        "add_any",
    )
)
SUPPORTED_PRIMITIVES = ELEMENTWISE_PRIMITIVES | STRUCTURAL_PRIMITIVES


def tensor(shape: tuple[int, ...], values: Iterable[jax.Array]) -> ScalarTensor:
    """Build a read-only compiler tensor from scalar tracers."""
    result = np.empty(shape, dtype=object)
    for index, value in zip(np.ndindex(shape), values, strict=True):
        result[index] = value
    result.setflags(write=False)
    return result


def literal(value: object) -> ScalarTensor:
    """Expand a native literal into exact typed scalar constants."""
    array = np.asarray(value)
    return tensor(
        array.shape,
        (jnp.asarray(array[index].item(), dtype=array.dtype) for index in np.ndindex(array.shape)),
    )


def unpack(value: jax.Array, shape: tuple[int, ...] | None = None) -> ScalarTensor:
    """Extract only logical coordinates from a padded kernel tensor."""
    shape = value.shape if shape is None else shape
    # Only state/output axes are unpacked; invariant operands are loaded as scalars.
    axes = jnp.ix_(*(jnp.arange(size) for size in value.shape))
    return (
        tensor(
            shape,
            (
                jnp.sum(
                    jnp.where(
                        sum(
                            (axis != i).astype(jnp.int32)
                            for axis, i in zip(axes, index, strict=True)
                        )
                        == 0,
                        value,
                        0.0,
                    )
                )
                for index in np.ndindex(shape)
            ),
        )
        if value.ndim
        else tensor((), (value,))
    )


def pack(array: ScalarTensor) -> jax.Array:
    """Pad each logical axis to a power of two; structural arithmetic stays logical."""
    if array.shape == ():
        return cast("jax.Array", array[()])
    padded = tuple(1 << max(size - 1, 0).bit_length() for size in array.shape)
    if array.size == 0:
        return jnp.zeros(padded)
    result = jnp.zeros(padded, dtype=array.flat[0].dtype)
    axes = jnp.ix_(*(jnp.arange(size) for size in padded))
    for index in np.ndindex(array.shape):
        match = jnp.ones(padded, dtype=jnp.bool_)
        for grid, coordinate in zip(axes, index, strict=True):
            match = match & (grid == coordinate)
        result = jnp.where(match, array[index], result)
    return result


def array_aval(variable: core.Var | core.Literal) -> jax_core.ShapedArray:
    """Expose shaped numerical avals at the native expression compiler boundary."""
    return cast("jax_core.ShapedArray", variable.aval)


def unsupported_primitive(equation: core.JaxprEqn) -> str | None:
    """Admit only primitive variants whose exact scalar lowering is implemented."""
    name = equation.primitive.name
    if name not in SUPPORTED_PRIMITIVES:
        return name
    if any(
        np.issubdtype(array_aval(v).dtype, np.complexfloating)
        for v in (*equation.invars, *equation.outvars)
    ):
        return name + "[complex]"
    if name == "gather":
        dims = equation.params["dimension_numbers"]
        if dims.operand_batching_dims or dims.start_indices_batching_dims:
            return "gather[batching]"
        if equation.params["mode"] not in (
            jax.lax.GatherScatterMode.CLIP,
            jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        ):
            return "gather[mode]"
    if name == "select_n" and (
        len(equation.invars) != 3 or array_aval(equation.invars[0]).dtype != np.dtype(bool)
    ):
        return "select_n[nonboolean-or-multiple-cases]"
    if name == "bitcast_convert_type" and (
        np.dtype(array_aval(equation.invars[0]).dtype).itemsize
        != np.dtype(equation.params["new_dtype"]).itemsize
    ):
        return "bitcast_convert_type[different-widths]"
    return None


def evaluate(
    equation: core.JaxprEqn, environment: Mapping[core.Var, ScalarTensor]
) -> Mapping[core.Var, ScalarTensor]:
    """Interpret an explicit structural or elementwise primitive, never a guessed one."""
    ins = [
        literal(v.val) if isinstance(v, core.Literal) else environment[v] for v in equation.invars
    ]
    name, params = equation.primitive.name, equation.params
    if name not in SUPPORTED_PRIMITIVES:
        raise ValueError(f"Unsupported point-expression primitive: {name}")
    if name == "unstack":
        axis = params["axis"]
        parts = tuple(
            tensor(
                array_aval(variable).shape,
                (
                    ins[0][(*index[:axis], i, *index[axis:])]
                    for index in np.ndindex(array_aval(variable).shape)
                ),
            )
            for i, variable in enumerate(equation.outvars)
        )
        return {**environment, **dict(zip(equation.outvars, parts, strict=True))}
    if name == "split":
        parts = np.split(ins[0], np.cumsum(params["sizes"])[:-1], axis=params["axis"])
        return {**environment, **dict(zip(equation.outvars, parts, strict=True))}
    shape = array_aval(equation.outvars[0]).shape
    if name == "add_any":
        out = tensor(
            shape, (jax.lax.add(ins[0][index], ins[1][index]) for index in np.ndindex(shape))
        )
    elif name == "slice":
        strides = params.get("strides") or (1,) * len(params["start_indices"])
        out = ins[0][
            tuple(
                slice(a, b, c)
                for a, b, c in zip(
                    params["start_indices"], params["limit_indices"], strides, strict=True
                )
            )
        ]
    elif name == "reshape":
        value = (
            ins[0] if params.get("dimensions") is None else ins[0].transpose(params["dimensions"])
        )
        out = value.reshape(shape)
    elif name == "squeeze":
        out = np.squeeze(ins[0], axis=params["dimensions"])
    elif name == "transpose":
        out = ins[0].transpose(params["permutation"])
    elif name == "broadcast_in_dim":
        dims = params["broadcast_dimensions"]
        source_shape = tuple(
            ins[0].shape[dims.index(i)] if i in dims else 1 for i in range(len(shape))
        )
        out = np.broadcast_to(ins[0].reshape(source_shape), shape)
    elif name == "stack":
        out = np.stack(ins, axis=params["axis"])
    elif name == "concatenate":
        out = np.concatenate(ins, axis=params["dimension"])
    elif name == "iota":
        out = tensor(
            shape,
            (jnp.asarray(i[params["dimension"]], dtype=params["dtype"]) for i in np.ndindex(shape)),
        )
    elif name.startswith("reduce_") and name in (
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_or",
        "reduce_and",
    ):
        axes = params["axes"]
        keep = [i for i in range(ins[0].ndim) if i not in axes]
        reduction = {
            "reduce_sum": lambda a, b: a + b,
            "reduce_max": jnp.maximum,
            "reduce_min": jnp.minimum,
            "reduce_prod": lambda a, b: a * b,
            "reduce_or": jnp.logical_or,
            "reduce_and": jnp.logical_and,
        }[name]
        identity = {
            "reduce_sum": 0,
            "reduce_max": -jnp.inf,
            "reduce_min": jnp.inf,
            "reduce_prod": 1,
            "reduce_or": False,
            "reduce_and": True,
        }[name]
        dtype = array_aval(equation.outvars[0]).dtype
        if name in ("reduce_max", "reduce_min"):
            if np.issubdtype(dtype, np.integer):
                identity = np.iinfo(dtype).min if name == "reduce_max" else np.iinfo(dtype).max
            elif np.issubdtype(dtype, np.bool_):
                identity = name == "reduce_min"
        vals = []
        for index in np.ndindex(shape):
            acc = jnp.asarray(identity, dtype=dtype)
            for red in np.ndindex(tuple(ins[0].shape[a] for a in axes)):
                full = tuple(
                    index[keep.index(d)] if d in keep else red[axes.index(d)]
                    for d in range(ins[0].ndim)
                )
                acc = reduction(acc, ins[0][full])
            vals.append(acc)
        out = tensor(shape, vals)
    elif name == "dot_general":
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        lf = [a for a in range(ins[0].ndim) if a not in lc + lb]
        rf = [a for a in range(ins[1].ndim) if a not in rc + rb]
        vals = []
        for idx in np.ndindex(shape):
            left = [0] * ins[0].ndim
            right = [0] * ins[1].ndim
            for n, (a, b) in enumerate(zip(lb, rb, strict=True)):
                left[a] = right[b] = idx[n]
            for n, a in enumerate(lf):
                left[a] = idx[len(lb) + n]
            for n, a in enumerate(rf):
                right[a] = idx[len(lb) + len(lf) + n]
            value = jnp.asarray(0.0, dtype=array_aval(equation.outvars[0]).dtype)
            for contracted in np.ndindex(tuple(ins[0].shape[a] for a in lc)):
                for n, (a, b) in enumerate(zip(lc, rc, strict=True)):
                    left[a] = right[b] = contracted[n]
                value = value + ins[0][tuple(left)] * ins[1][tuple(right)]
            vals.append(value)
        out = tensor(shape, vals)
    elif name == "gather":
        d = params["dimension_numbers"]
        sizes = params["slice_sizes"]
        assert not d.operand_batching_dims
        assert not d.start_indices_batching_dims
        batch = [a for a in range(len(shape)) if a not in d.offset_dims]
        operand_axes = [a for a in range(ins[0].ndim) if a not in d.collapsed_slice_dims]
        vals = []
        for idx in np.ndindex(shape):
            start_index = tuple(idx[a] for a in batch)
            starts = [jnp.asarray(0, jnp.int32) for _ in ins[0].shape]
            for axis, a in enumerate(d.start_index_map):
                starts[a] = jnp.clip(ins[1][(*start_index, axis)], 0, ins[0].shape[a] - sizes[a])
            for a, offset in zip(operand_axes, d.offset_dims, strict=True):
                starts[a] = starts[a] + idx[offset]
            value = jnp.asarray(0.0, dtype=array_aval(equation.outvars[0]).dtype)
            for src in np.ndindex(ins[0].shape):
                match = jnp.asarray(True)
                for a, s in enumerate(src):
                    match = match & (starts[a] == s)
                value = jnp.where(match, ins[0][src], value)
            vals.append(value)
        out = tensor(shape, vals)
    elif name == "dynamic_slice":
        starts = [
            jnp.clip(v[()], 0, n - size)
            for v, n, size in zip(ins[1:], ins[0].shape, params["slice_sizes"], strict=True)
        ]
        vals = []
        for idx in np.ndindex(shape):
            value = jnp.asarray(0.0, dtype=array_aval(equation.outvars[0]).dtype)
            for src in np.ndindex(ins[0].shape):
                match = jnp.asarray(True)
                for a, s in enumerate(src):
                    match = match & (starts[a] + idx[a] == s)
                value = jnp.where(match, ins[0][src], value)
            vals.append(value)
        out = tensor(shape, vals)
    elif name == "pad":
        out = np.empty(shape, dtype=object)
        out.fill(ins[1][()])
        for src in np.ndindex(ins[0].shape):
            dest = tuple(
                lo + i * (interior + 1)
                for i, (lo, hi, interior) in zip(src, params["padding_config"], strict=True)
            )
            if all(0 <= i < n for i, n in zip(dest, shape, strict=True)):
                out[dest] = ins[0][src]
    else:
        # Only explicitly admitted elementwise primitives reach this branch.
        assert name in ELEMENTWISE_PRIMITIVES
        assert not equation.primitive.multiple_results, (name, params)
        arrays = [np.broadcast_to(value, shape) for value in ins]
        vals = []
        for index in np.ndindex(shape):
            vals.append(
                equation.primitive.bind(
                    *(a[index] for a in arrays), **equation.primitive.get_bind_params(params)
                )
            )
        out = tensor(shape, vals)
    for value in out.flat:
        assert getattr(value, "shape", ()) == (), (
            name,
            shape,
            getattr(value, "shape", None),
            [[(getattr(v, "shape", None)) for v in a.flat] for a in ins],
            params,
        )
    out.setflags(write=False)
    return {**environment, equation.outvars[0]: out}
