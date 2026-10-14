"""JSON/Pydantic boundary for native NumPyro distributions and their constructor trees.

The Python field is a NumPyro Distribution. JSON records native constructors,
arrays, transforms, and constraints; no probability behavior is implemented here.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from pydantic import GetCoreSchemaHandler, GetPydanticSchema, SerializationInfo, ValidationInfo
from pydantic_core import core_schema

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.json_types import JsonScalar, JsonValue

if TYPE_CHECKING:
    from jax.typing import ArrayLike

_NATIVE_MODULES = {
    "distribution": (dist, dist.Distribution),
    "transform": (transforms, transforms.Transform),
    "constraint": (constraints, constraints.Constraint),
}

type ArrayLoader = Callable[[str], np.ndarray]
type NumPyroValue = JsonScalar | NumericalArray | NumPyroArray | NumPyroObject
type NumPyroArray = Sequence[NumPyroValue]
type NumPyroObject = Mapping[str, NumPyroValue]


def _stored_constructor(value: NumPyroValue, arrays: dict[str, NumericalArray]) -> JsonValue:
    """Parse embedded law arguments into the lazy native constructor representation."""
    if isinstance(value, NumericalArray):
        arrays[value.identity] = value
        shape, dtype = value.layout
        return {
            "array_ref": value.identity,
            "shape": list(shape),
            "dtype": str(dtype),
            "index": list[int](),
        }
    if isinstance(value, Mapping):
        return {key: _stored_constructor(item, arrays) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_stored_constructor(item, arrays) for item in value]
    return value


def serialize_constructor(
    value: NumPyroValue, *, binary: bool, array_loader: ArrayLoader | None = None
) -> NumPyroValue:
    """Keep storage references internal; published native arguments own exact NPY values."""
    if isinstance(value, NumericalArray):
        if binary:
            return value
        return _stored_constructor(value, {})
    if isinstance(value, Mapping):
        if binary and "array_ref" in value:
            if array_loader is None:
                raise ValueError("Publishing a stored law requires its numerical values")
            return NumericalArray.from_numpy(
                np.asarray(_decode(_stored_constructor(value, {}), array_loader))
            )
        return {
            key: serialize_constructor(item, binary=binary, array_loader=array_loader)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            serialize_constructor(item, binary=binary, array_loader=array_loader) for item in value
        ]
    return value


def serialize_distribution(value: dist.Distribution, info: SerializationInfo) -> NumPyroObject:
    """Use the native constructor schema in both metadata and binary transport."""
    binary = bool((info.context or {}).get("binary_arrays"))
    loader = value.loader if isinstance(value, _StoredDistribution) else None
    if loader is None:
        loader = (info.context or {}).get("distribution_array_loader")
    return cast(
        "NumPyroObject",
        serialize_constructor(encode_distribution(value), binary=binary, array_loader=loader),
    )


class _StoredDistribution(dist.Distribution):
    """Lazy I/O for a native constructor; never a new probability family.

    Reading a model or its schema must not fetch hundreds of MB of numerical
    values. Numerical consumers materialize the native law before JAX tracing.
    """

    constructor: dict[str, JsonValue]
    loader: ArrayLoader | None

    def __init__(self, constructor: dict[str, JsonValue], loader: ArrayLoader | None) -> None:
        self.constructor = constructor
        self.loader = loader

    @cached_property
    def native(self) -> dist.Distribution:
        value = _decode(self.constructor, self.loader)
        if not isinstance(value, dist.Distribution):
            raise ValueError("Expected a native distribution")
        return value

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.native, name)

    def sample(self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        return self.native.sample(key, sample_shape)

    def log_prob(self, value: ArrayLike, intermediates: list[object] | None = None) -> ArrayLike:
        return self.native.log_prob(value, intermediates)

    @property
    def support(self) -> constraints.Constraint | None:
        return self.native.support

    @property
    def mean(self) -> jax.Array:
        return jnp.asarray(self.native.mean)

    @property
    def variance(self) -> jax.Array:
        return jnp.asarray(self.native.variance)


def materialize_distribution(value: dist.Distribution) -> dist.Distribution:
    """Load referenced values and return the actual native NumPyro instance."""
    return value.native if isinstance(value, _StoredDistribution) else value


def distribution_shape(value: dist.Distribution) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive native shapes, reading only array metadata for stored empirical laws."""
    match value:
        case _StoredDistribution(
            constructor={
                "distribution": "MixtureSameFamily",
                "params": {
                    "component_distribution": {
                        "distribution": "Delta",
                        "params": {"v": {"array_ref": _, "shape": stored, "index": index}} as delta,
                    },
                },
            }
        ):
            # Decoding validated every array reference's shape and index.
            shape = tuple(cast("Sequence[int]", stored))[len(cast("Sequence[int]", index)) :]
            event_dim = delta["event_dim"]
            if not isinstance(event_dim, int) or event_dim < 0 or event_dim >= len(shape):
                raise ValueError("Invalid native Delta event dimensions")
            # The component's rightmost batch axis enumerates the mixture's atoms.
            return shape[: len(shape) - event_dim - 1], shape[len(shape) - event_dim :]
        case _:
            native = materialize_distribution(value)
            return tuple(int(n) for n in native.batch_shape), tuple(
                int(n) for n in native.event_shape
            )


def empirical_distribution(
    values: np.ndarray,
    *,
    array_writer: Callable[[np.ndarray], str] | None = None,
    array_loader: ArrayLoader | None = None,
) -> dist.Distribution:
    """Express aligned finite samples as a native categorical mixture of point masses.

    One batched Delta holds every draw and its leading axis enumerates the equally
    weighted atoms, so the law stores two arrays however many draws it retains.
    """
    values = np.asarray(values)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Empirical laws require finite, nonempty draw and event axes")

    def argument(array: np.ndarray) -> JsonValue:
        if array_writer is None:
            return _encode(array)
        return {
            "array_ref": array_writer(array),
            "shape": [int(n) for n in array.shape],
            "dtype": str(array.dtype),
            "index": list[int](),
        }

    constructor: dict[str, JsonValue] = {
        "distribution": "MixtureSameFamily",
        "params": {
            "mixing_distribution": {
                "distribution": "CategoricalProbs",
                "params": {"probs": argument(np.full(values.shape[0], 1.0 / values.shape[0]))},
            },
            "component_distribution": {
                "distribution": "Delta",
                "params": {"v": argument(values), "event_dim": 1},
            },
        },
    }
    return decode_distribution(constructor, array_loader=array_loader)


def empirical_atoms(value: dist.Distribution) -> np.ndarray:
    """Read the equal-weight atoms of a native empirical law as one array.

    This extracts retained particle draws, not fitted marginal approximations.
    Other probability families should be sampled through their native methods.
    """
    law = materialize_distribution(value)
    if not isinstance(law, dist.MixtureSameFamily) or not isinstance(
        law.component_distribution, dist.Delta
    ):
        raise ValueError("Retained draws require a native point-mixture distribution")
    weights = np.asarray(law.mixing_distribution.probs)
    atoms = law.component_distribution
    if not np.all(weights == weights[0]):
        raise ValueError("Retained particle draws must have equal weights")
    if atoms.event_dim != 1 or np.any(np.asarray(atoms.log_density) != 0.0):
        raise ValueError("Retained draws require vector point masses")
    return np.asarray(atoms.v)


def _has_array_refs(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return "array_ref" in value or any(_has_array_refs(item) for item in value.values())
    return isinstance(value, list) and any(_has_array_refs(item) for item in value)


def _validate_stored_tree(value: JsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_stored_tree(item)
    elif isinstance(value, dict):
        if "array_ref" in value:
            if (
                set(value) != {"array_ref", "shape", "dtype", "index"}
                or re.fullmatch(r"[0-9a-f]{64}", str(value["array_ref"])) is None
                or not isinstance(value["shape"], list)
                or not all(isinstance(n, int) and n >= 0 for n in value["shape"])
                or not isinstance(value["index"], list)
                or not all(isinstance(n, int) and n >= 0 for n in value["index"])
                or np.dtype(str(value["dtype"])).hasobject
            ):
                raise ValueError("Invalid numerical array reference")
            shape = cast("Sequence[int]", value["shape"])
            index = cast("Sequence[int]", value["index"])
            if len(index) > len(shape) or any(i >= n for i, n in zip(index, shape, strict=False)):
                raise ValueError("Numerical array reference index is outside its declared shape")
        for tag, (module, base) in _NATIVE_MODULES.items():
            if tag in value:
                cls = getattr(module, str(value[tag]), None)
                if (
                    not isinstance(cls, type)
                    or not issubclass(cls, base)
                    or set(value) != {tag, "params"}
                    or not isinstance(value["params"], dict)
                ):
                    raise ValueError(f"Invalid native {tag} constructor")
        for item in value.values():
            _validate_stored_tree(item)


def _constructor_arguments(value: object) -> dict[str, object]:
    cls = type(value)
    arguments: dict[str, object] = value.get_args() if isinstance(value, dist.Distribution) else {}
    # NumPyro caches equivalent matrix parameterizations. A constructor accepts
    # just one; its stored Cholesky factor preserves the native law directly.
    if "scale_tril" in arguments:
        arguments.pop("covariance_matrix", None)
        arguments.pop("precision_matrix", None)
    aliases = {"base_distribution": "base_dist", "mask": "_mask", "transform": "_inv"}
    if isinstance(value, dist.MixtureGeneral):
        aliases["support"] = "_support"
    for name, parameter in inspect.signature(cls).parameters.items():
        if name in arguments or name == "validate_args":
            continue
        if name in getattr(cls, "arg_constraints", {}):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        attribute = aliases.get(name, name)
        if hasattr(value, attribute):
            arguments[name] = getattr(value, attribute)
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"Cannot serialize native constructor argument {cls.__name__}.{name}")
    return arguments


def rebuild_distribution(distribution: dist.Distribution) -> dist.Distribution:
    """Reconstruct native shape metadata after batching a distribution's PyTree.

    JAX stacks parameter leaves but retains static batch shapes and cached child
    distributions. Native constructors infer those again from the new arguments.
    Arrays stay in JAX; this operation does not serialize or detach gradients.
    """

    def rebuild[NativeValue](value: NativeValue) -> NativeValue:
        if isinstance(value, (dist.Distribution, transforms.Transform, constraints.Constraint)):
            return cast(
                "NativeValue",
                type(value)(
                    **{name: rebuild(item) for name, item in _constructor_arguments(value).items()}
                ),
            )
        if isinstance(value, (tuple, list)):
            return cast("NativeValue", type(value)(rebuild(item) for item in value))
        return value

    return rebuild(distribution)


@overload
def _encode(value: dist.Distribution) -> dict[str, JsonValue]: ...


@overload
def _encode(value: object) -> JsonValue: ...


def _encode(value: object) -> JsonValue:
    if isinstance(value, _StoredDistribution):
        return value.constructor
    for tag, (module, base) in _NATIVE_MODULES.items():
        if isinstance(value, base):
            cls = type(value)
            constructor_name: str = cls.__name__
            if getattr(module, constructor_name, None) is not cls:
                raise ValueError(f"{cls.__name__} is not a native NumPyro {tag} constructor")
            arguments = _constructor_arguments(value)
            if (
                isinstance(value, dist.Distribution)
                and "validate_args" in inspect.signature(cls).parameters
            ):
                # Preserve native execution settings. In particular, exact
                # categorical zeros use -inf logits, outside real_vector's
                # finite-only argument check despite defining a valid law.
                arguments["validate_args"] = value._validate_args
            return {
                tag: constructor_name,
                "params": {name: _encode(item) for name, item in arguments.items()},
            }
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError("Complex distribution arguments are not JSON values")
        return {"array": _encode(array.tolist()), "dtype": str(array.dtype)}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": str(value)}
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise ValueError(f"Cannot serialize native distribution argument {type(value).__name__}")


def _decode(value: JsonValue, array_loader: ArrayLoader | None = None) -> object:
    if isinstance(value, list):
        return [_decode(item, array_loader) for item in value]
    if not isinstance(value, dict):
        return value
    if "array_ref" in value:
        _validate_stored_tree(value)
        if array_loader is None:
            raise ValueError(
                "Materializing a stored distribution requires its workspace array loader"
            )
        array = array_loader(str(value["array_ref"]))
        if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
            raise ValueError("Stored numerical array does not match its declared shape and dtype")
        return array[tuple(cast("Sequence[int]", value["index"]))]
    for tag, (module, base) in _NATIVE_MODULES.items():
        if tag in value:
            cls = getattr(module, str(value[tag]), None)
            if not isinstance(cls, type) or not issubclass(cls, base):
                raise ValueError(f"Unknown native NumPyro {tag} constructor {value[tag]!r}")
            if set(value) != {tag, "params"} or not isinstance(value["params"], dict):
                raise ValueError(f"A native {tag} requires exactly its name and constructor params")
            arguments = {
                name: _decode(item, array_loader) for name, item in value["params"].items()
            }
            if "validate_args" in inspect.signature(cls).parameters:
                arguments.setdefault("validate_args", True)
            try:
                decoded = cls(**arguments)
            except TypeError as exc:  # noqa: ERR001 -- Native NumPyro constructors document TypeError for argument mismatch; only that call is wrapped.
                raise ValueError(f"Invalid {cls.__name__} constructor arguments: {exc}") from exc
            return (
                _validate_distribution(decoded)
                if isinstance(decoded, dist.Distribution)
                else decoded
            )
    if set(value) == {"array", "dtype"}:
        return np.asarray(_decode(value["array"], array_loader), dtype=str(value["dtype"]))
    if set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(_decode(item, array_loader) for item in value["tuple"])
    if set(value) == {"float"} and value["float"] in {"inf", "-inf", "nan"}:
        return float(str(value["float"]))
    return {name: _decode(item, array_loader) for name, item in value.items()}


def encode_distribution(value: dist.Distribution) -> dict[str, JsonValue]:
    """Serialize a native law without flattening batch/event dimensions or mixtures."""
    return _encode(value)


def decode_distribution(
    value: dict[str, JsonValue], *, array_loader: ArrayLoader | None = None
) -> dist.Distribution:
    """Restore and validate the exact native constructor tree."""
    if _has_array_refs(value):
        _validate_stored_tree(value)
        if "distribution" not in value:
            raise ValueError("Expected a NumPyro distribution constructor")
        return _StoredDistribution(value, array_loader)
    decoded = _decode(value, array_loader)
    if not isinstance(decoded, dist.Distribution):
        raise ValueError("Expected a NumPyro distribution constructor")
    return decoded


def _validate_distribution(value: dist.Distribution) -> dist.Distribution:
    if isinstance(value, _StoredDistribution):
        return value
    if isinstance(value, dist.CategoricalLogits):
        # Validate the same categorical law in probability coordinates so exact
        # zero weights remain legal. NaN weights and an all-zero law still fail
        # NumPyro's simplex constraint. This does not change the stored logits.
        dist.Categorical(probs=value.probs, validate_args=True)
    else:
        value.validate_args()
    return value


def _distribution_schema(_source: object, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    wire = core_schema.typed_dict_schema(
        {
            "distribution": core_schema.typed_dict_field(core_schema.str_schema()),
            "params": core_schema.typed_dict_field(
                handler.generate_schema(dict[str, NumPyroValue])
            ),
        },
        extra_behavior="forbid",
        ref="NumPyroDistribution",
    )

    def decode(value: NumPyroObject, info: ValidationInfo) -> dist.Distribution:
        arrays: dict[str, NumericalArray] = {}
        constructor = {key: _stored_constructor(item, arrays) for key, item in value.items()}
        external: ArrayLoader | None = (info.context or {}).get("distribution_array_loader")

        def load(identity: str) -> np.ndarray:
            if identity in arrays:
                return arrays[identity].values
            if external is None:
                raise ValueError("Stored law is missing its numerical value")
            return external(identity)

        return decode_distribution(constructor, array_loader=load if arrays else external)

    decoder = core_schema.with_info_after_validator_function(decode, wire)
    return core_schema.json_or_python_schema(
        json_schema=decoder,
        python_schema=core_schema.union_schema(
            [
                core_schema.no_info_after_validator_function(
                    _validate_distribution, core_schema.is_instance_schema(dist.Distribution)
                ),
                decoder,
            ]
        ),
        serialization=core_schema.plain_serializer_function_ser_schema(
            serialize_distribution, info_arg=True, when_used="always", return_schema=wire
        ),
    )


NumPyroDistribution = Annotated[dist.Distribution, GetPydanticSchema(_distribution_schema)]
