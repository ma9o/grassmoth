"""Pinned call identities, revision selectors and the service artifact catalogue."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Annotated, Literal, Self, get_args

from pydantic import Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema

from nof1_causal_lab.public import GitOid

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue


class CallId(str):
    """Content-derived identity of a pinned action request, prefixed with ``call:``."""

    def __new__(cls, value: str) -> Self:
        """Parse the canonical content-derived call identity."""
        if re.fullmatch(r"call:[0-9a-f]{64}", value) is None:
            raise ValueError(f"Invalid {cls.__name__}: {value!r}")
        return str.__new__(cls, value)

    @classmethod
    def from_content(cls, payload: object) -> Self:
        """Hash the canonical pinned request into its service call identity."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return cls("call:" + hashlib.sha256(encoded.encode()).hexdigest())

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Validate both constructed and wire-format call identities."""
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(pattern=r"^call:[0-9a-f]{64}$"),
            ref=cls.__name__,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Retain the public string template for generated callers."""
        result = handler(schema)
        result["tsType"] = "`call:${string}`"
        return result


type RevisionSelector = Annotated[
    GitOid | Literal["latest"],
    Field(
        description="An exact Git hash, or 'latest': the current non-stale model/panel or most recent applied simulation, according to the input type. Resolved once before cache lookup and execution; repeat the returned hashes to poll the same call."
    ),
]


type ArtifactId = Literal["question", "raw_data", "model", "panel"]


type ScientificActionId = Literal["edit_question", "edit_model", "prepare_data", "fit", "simulate"]


type ActionId = ScientificActionId | Literal["data_diff", "model_diff"]


ARTIFACT_IDS: tuple[ArtifactId, ...] = get_args(ArtifactId.__value__)
