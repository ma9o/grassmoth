"""Persistent authored entity IDs, independent of names and model revisions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, get_args, overload

from pydantic import Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema

from nof1_causal_lab.artifacts.base import Value

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue


class _IdentityString(str):
    """A nominal ID validates the same grammar on public and parsed construction."""

    _pattern: ClassVar[str]
    _prefix: ClassVar[str | None] = None

    def __new__(cls, value: str) -> Self:
        if re.fullmatch(cls._pattern, value) is None:
            raise ValueError(f"Invalid {cls.__name__}: {value!r}")
        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(pattern=cls._pattern),
            ref=cls.__name__,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler(schema)
        if cls._prefix is not None:
            result["tsType"] = f"`{cls._prefix}:${{string}}`"
        return result


class GitOid(_IdentityString):
    """A full Git object identity spelled as forty lowercase hexadecimal characters."""

    _pattern = r"^[0-9a-f]{40}$"


class CallId(_IdentityString):
    """Content-derived identity of a pinned action request, prefixed with ``call:``."""

    _prefix = "call"
    _pattern = r"^call:[0-9a-f]{64}$"


type RevisionSelector = Annotated[
    GitOid | Literal["latest"],
    Field(
        description="An exact Git hash, or 'latest': the current non-stale model/panel or most recent applied simulation, according to the input type. Resolved once before cache lookup and execution; repeat the returned hashes to poll the same call."
    ),
]


class ConstructId(_IdentityString):
    """Namespaced identity of a scientific construct, independent of its display label."""

    _prefix = "construct"
    _pattern = r"^construct:[A-Za-z0-9_.-]+$"


class EdgeId(_IdentityString):
    """Namespaced identity of a directed causal edge, independent of endpoint labels."""

    _prefix = "edge"
    _pattern = r"^edge:[A-Za-z0-9_.-]+$"


class IndicatorId(_IdentityString):
    """Namespaced observation identity shared by model definitions and recorded measurements."""

    _prefix = "indicator"
    _pattern = r"^indicator:[A-Za-z0-9_.-]+$"


class MechanismId(_IdentityString):
    """Namespaced identity of an authored dynamics mechanism."""

    _prefix = "mechanism"
    _pattern = r"^mechanism:[A-Za-z0-9_.-]+$"


class DistributionId(_IdentityString):
    """Namespaced identity used to refer to a distribution owned by a model."""

    _prefix = "distribution"
    _pattern = r"^distribution:[A-Za-z0-9_.-]+$"


class ParameterId(_IdentityString):
    """Content-derived identity of a parameter's scientific definition."""

    _prefix = "parameter"
    _pattern = r"^parameter:[0-9a-f]{64}$"


class ParameterElementId(_IdentityString):
    """Content-derived identity of one coordinate within a parameter value."""

    _prefix = "element"
    _pattern = r"^element:[0-9a-f]{64}$"


type ArtifactId = Literal["question", "raw_data", "model", "panel"]

# Actions name work; several actions can enrich the same model artifact.
type ScientificActionId = Literal["edit_question", "edit_model", "prepare_data", "fit", "simulate"]
type ActionId = ScientificActionId | Literal["data_diff", "model_diff"]

ARTIFACT_IDS: tuple[ArtifactId, ...] = get_args(ArtifactId.__value__)


class GitRef(Value):
    """An exact file in a study's Git object database: repository, object, and path."""

    workspace_id: str = Field(min_length=1)
    revision: GitOid
    path: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


class ConstructRef(Value):
    """A construct identity independent of its current display name or model revision."""

    kind: Literal["construct"] = "construct"
    id: ConstructId


class EdgeRef(Value):
    """A causal-edge identity independent of edits to its scientific definition."""

    kind: Literal["edge"] = "edge"
    id: EdgeId


class IndicatorRef(Value):
    """An observation identity independent of its current display name or model revision."""

    kind: Literal["indicator"] = "indicator"
    id: IndicatorId


class MechanismRef(Value):
    """A particular additive term, independently of its position or coefficient values."""

    kind: Literal["mechanism"] = "mechanism"
    id: MechanismId


class ParameterRef(Value):
    """A scalar finding identifies its scientific parameter and declared logical component."""

    parameter_id: ParameterId
    element_id: ParameterElementId


type EntityRef = Annotated[
    ConstructRef | EdgeRef | IndicatorRef | MechanismRef, Field(discriminator="kind")
]


@overload
def scientific_id(prefix: Literal["construct"], payload: object) -> ConstructId: ...


@overload
def scientific_id(prefix: Literal["edge"], payload: object) -> EdgeId: ...


@overload
def scientific_id(prefix: Literal["indicator"], payload: object) -> IndicatorId: ...


@overload
def scientific_id(prefix: Literal["mechanism"], payload: object) -> MechanismId: ...


@overload
def scientific_id(prefix: Literal["distribution"], payload: object) -> DistributionId: ...


@overload
def scientific_id(prefix: Literal["parameter"], payload: object) -> ParameterId: ...


@overload
def scientific_id(prefix: Literal["element"], payload: object) -> ParameterElementId: ...


@overload
def scientific_id(prefix: Literal["call"], payload: object) -> CallId: ...


@overload
def scientific_id(prefix: str, payload: object) -> str: ...


def scientific_id(prefix: str, payload: object) -> str:
    """Hash canonical JSON into a namespaced identity and parse recognized identity kinds."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    value = f"{prefix}:{hashlib.sha256(encoded.encode()).hexdigest()}"
    match prefix:
        case "call":
            return CallId(value)
        case "construct":
            return ConstructId(value)
        case "edge":
            return EdgeId(value)
        case "indicator":
            return IndicatorId(value)
        case "mechanism":
            return MechanismId(value)
        case "distribution":
            return DistributionId(value)
        case "parameter":
            return ParameterId(value)
        case "element":
            return ParameterElementId(value)
        case _:
            return value
