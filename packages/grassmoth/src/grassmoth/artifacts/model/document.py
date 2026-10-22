"""Identity-addressed scientific documents and their immutable merge operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue

    from grassmoth.json_types import JsonObject, JsonValue

_ENTITY_COLLECTIONS = frozenset({"constructs", "edges", "parameters", "dynamics", "mechanisms"})
_ENTRY_MAPS = _ENTITY_COLLECTIONS | {"indicators", "distributions", "law_layouts", "labels"}


def merge_fields(base: JsonObject, supplied: JsonObject) -> JsonObject:
    """Merge supplied fields; tagged scientific alternatives replace their entire value."""
    selected = _merge_base(base, supplied)
    return {
        key: _merge_field(key, selected.get(key), supplied[key]) if key in supplied else value
        for key, value in (dict(selected) | dict(supplied)).items()
    }


def _merge_base(base: JsonObject, supplied: JsonObject) -> JsonObject:
    family = supplied.get("distribution")
    discriminators = (
        ("kind", "distribution")
        if isinstance(family, str) and not family.startswith("distribution:")
        else ("kind",)
    )
    empty: dict[str, JsonValue] = {}
    return (
        empty
        if any(key in supplied and supplied[key] != base.get(key) for key in discriminators)
        else base
    )


def _merge_field(name: str, before: JsonValue, after: JsonValue) -> JsonValue:
    if not isinstance(after, Mapping):
        return after
    base = before if isinstance(before, Mapping) else {}
    if name in _ENTRY_MAPS:
        return {
            key: _merge_entry(base.get(key), value) if key in after else value
            for key, value in (
                dict(cast("JsonObject", base)) | dict(cast("JsonObject", after))
            ).items()
            if value is not None
        }
    return merge_fields(base, after)


def _merge_entry(before: JsonValue, after: JsonValue) -> JsonValue:
    return (
        merge_fields(before if isinstance(before, Mapping) else {}, after)
        if isinstance(after, Mapping)
        else after
    )


def diff_fields(before: JsonObject, after: JsonObject) -> JsonObject:
    """Return a deterministic patch between materialized scientific documents.

    ``merge_fields(before, diff_fields(before, after)) == after``. Compare exact
    stored values, including numerical buffer references; do not compare laws by
    mathematical equivalence or dereference their arrays. Map order is irrelevant,
    and emitted keys are sorted. Equal documents yield an empty mapping.

    Recurse through retained objects and match entity-map entries by identity.
    Omit unchanged fields, supply additions/updates, and emit null for deleted map
    entries. A nullable ordinary field becoming null is an update, not deletion.
    Added entities carry their whole definition. A changed kind or distribution
    family replaces its tagged alternative whole, using the same discriminator
    rules as merge_fields. Ordinary sequences are ordered atomic replacements.
    Changes to a referenced entity are reported only at that entity's owner, not
    repeated at unchanged references. Materialized ordinary object fields remain
    present; omission is an editing instruction, not an after-value to reconstruct.
    """
    base = _merge_base(before, after)
    return {
        key: _diff_field(key, base.get(key), value)
        for key, value in sorted(after.items())
        if key not in base or base[key] != value
    }


def _diff_field(name: str, before: JsonValue, after: JsonValue) -> JsonValue:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return after
    if name in _ENTRY_MAPS:
        return {
            key: _diff_field("", before.get(key), after[key]) if key in after else None
            for key in sorted(before.keys() | after.keys())
            if key not in before or key not in after or before[key] != after[key]
        }
    return diff_fields(before, after)


def entity_document(value: JsonObject) -> JsonObject:
    """Encode one resolved entity, with identities owned by its enclosing map."""
    return {
        key: {
            cast("str", item["id"]): entity_document(item)
            for item in cast("Sequence[JsonObject]", member)
        }
        if key in _ENTITY_COLLECTIONS
        and isinstance(member, Sequence)
        and not isinstance(member, str)
        else {
            cast("str", item["observation"]["id"]): entity_document(item)
            for item in cast("Sequence[Mapping[str, JsonObject]]", member)
        }
        if key == "indicators" and isinstance(member, Sequence) and not isinstance(member, str)
        else entity_document(member)
        if isinstance(member, Mapping)
        else member
        for key, member in value.items()
        if key != "id"
    }


def entity_fields(identity: str, value: JsonValue) -> JsonObject:
    """Expand map identities into the resolved scientific entity constructors."""
    if not isinstance(value, Mapping):
        raise ValueError("Entity definitions must be objects")
    fields = cast("JsonObject", value)
    if any(
        key in _ENTITY_COLLECTIONS | {"indicators"} and not isinstance(member, Mapping)
        for key, member in fields.items()
    ):
        raise ValueError("Entity collections, including indicators, must be ID-keyed objects")
    if "id" in fields:
        raise ValueError("Entity identities belong in their map keys, not their definitions")
    return {
        "id": identity,
        **{
            key: tuple(
                entity_fields(child_id, child)
                for child_id, child in cast("Mapping[str, JsonObject]", member).items()
            )
            if key in _ENTITY_COLLECTIONS and isinstance(member, Mapping)
            else tuple(
                _indicator_fields(child_id, child)
                for child_id, child in cast("Mapping[str, JsonValue]", member).items()
            )
            if key == "indicators" and isinstance(member, Mapping)
            else member
            for key, member in fields.items()
        },
    }


def _indicator_fields(identity: str, value: JsonValue) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError("Indicator definitions must be objects")
    fields = cast("JsonObject", value)
    return {**fields, "observation": entity_fields(identity, fields.get("observation", {}))}


def parameter_references(value: JsonValue) -> frozenset[str]:
    """Collect parameter operands from scientific components, excluding entity definitions."""
    if isinstance(value, Mapping):
        own = value.get("value") if value.get("kind") == "coefficient" else None
        return frozenset(
            {own} if isinstance(own, str) and own.startswith("parameter:") else ()
        ).union(*(parameter_references(item) for item in value.values()))
    if isinstance(value, Sequence) and not isinstance(value, str):
        return frozenset[str]().union(*(parameter_references(item) for item in value))
    return frozenset[str]()


def _entity_id(schema: JsonSchemaValue, handler: GetJsonSchemaHandler) -> JsonSchemaValue | None:
    schema = handler.resolve_ref_schema(schema)
    if "oneOf" in schema:
        return _entity_id(schema["oneOf"][0], handler)
    properties = schema.get("properties", {})
    if "observation" in properties:
        return _entity_id(properties["observation"], handler)
    return properties.get("id")


def document_schema(
    schema: JsonSchemaValue,
    handler: GetJsonSchemaHandler,
    *,
    active: frozenset[str] = frozenset(),
    recursive: Mapping[str, JsonSchemaValue] = MappingProxyType({}),
) -> JsonSchemaValue:
    """Derive document fields from their resolved owners, with optional input fields."""
    reference = schema.get("$ref")
    if reference is not None:
        if reference in recursive:
            return recursive[reference]
        if reference in active:
            if handler.mode == "validation":
                return _recursive_document_schema(schema, handler, active, recursive)
            return schema
        active = active | {reference}
        schema = handler.resolve_ref_schema(schema)
    result = dict(schema)
    result.pop("title", None)
    if handler.mode == "validation":
        result.pop("default", None)
    if "properties" in schema:
        properties = {
            name: document_schema(value, handler, active=active, recursive=recursive)
            for name, value in schema["properties"].items()
            if name != "id"
        }
        for name in _ENTITY_COLLECTIONS | {"indicators"}:
            if name in properties and "items" in properties[name]:
                identity = _entity_id(schema["properties"][name]["items"], handler)
                if identity is None:
                    continue
                item = properties[name]["items"]
                properties[name] = {
                    "type": "object",
                    "propertyNames": identity,
                    "additionalProperties": {"anyOf": [item, {"type": "null"}]}
                    if handler.mode == "validation"
                    else item,
                }
        for name in ("distributions", "law_layouts", "labels"):
            if name in properties and handler.mode == "validation":
                item = document_schema(
                    schema["properties"][name]["additionalProperties"],
                    handler,
                    active=active,
                    recursive=recursive,
                )
                properties[name]["additionalProperties"] = {"anyOf": [item, {"type": "null"}]}
        result["properties"] = properties
        required = [name for name in schema.get("required", ()) if name in properties]
        if handler.mode == "serialization" and required:
            result["required"] = required
        else:
            result.pop("required", None)
    if "items" in schema:
        result["items"] = document_schema(
            schema["items"], handler, active=active, recursive=recursive
        )
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            alternatives = [
                document_schema(item, handler, active=active, recursive=recursive)
                for item in schema[keyword]
            ]
            if keyword == "oneOf" and handler.mode == "validation":
                result.pop("oneOf", None)
                result.pop("discriminator", None)
                result["anyOf"] = alternatives
            else:
                result[keyword] = alternatives
    return result


def _recursive_document_schema(
    schema: JsonSchemaValue,
    handler: GetJsonSchemaHandler,
    active: frozenset[str],
    recursive: Mapping[str, JsonSchemaValue],
) -> JsonSchemaValue:
    """Keep recursively nested input fields partial without changing their complete owners."""
    reference = schema["$ref"]
    identity = f"{__name__}.DynamicalModelSpecDocument[{reference}]"
    projected = handler(
        core_schema.definitions_schema(
            core_schema.definition_reference_schema(identity),
            [core_schema.any_schema(ref=identity)],
        )
    )
    target = handler.resolve_ref_schema(projected)
    if not target:
        target.update(
            document_schema(
                handler.resolve_ref_schema(schema),
                handler,
                active=active,
                recursive={**recursive, reference: projected},
            )
        )
        target.update(
            {
                "x-python-module": "grassmoth.artifacts.model.dynamical_model_spec",
                "x-typescript-mode": "validation",
            }
        )
    return projected
