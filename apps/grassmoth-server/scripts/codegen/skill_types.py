"""Render action input and successful output documentation from their HTTP schemas."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue

_SOURCE_ROOTS = {
    "grassmoth": "../../../packages/grassmoth/src",
    "grassmoth_server": "../../../apps/grassmoth-server/src",
}


def _resolve(schema: JsonSchemaValue, components: JsonSchemaValue) -> JsonSchemaValue:
    return components[schema["$ref"].rsplit("/", 1)[-1]] if "$ref" in schema else schema


def _type(schema: JsonSchemaValue, components: JsonSchemaValue) -> str:
    if "$ref" in schema:
        component = schema["$ref"].rsplit("/", 1)[-1]
        definition = components[component]
        name = component.removesuffix("-Input").removesuffix("-Output")
        if "x-typescript-type" in definition:
            name = definition["x-typescript-type"]
        elif "title" in definition:
            name = definition["title"]
        module = definition["x-python-module"]
        source = f"{_SOURCE_ROOTS[module.partition('.')[0]]}/{module.replace('.', '/')}.py"
        result = f"[`{name}`]({source})"
        if "discriminator" in definition:
            discriminator = definition["discriminator"]
            variants = ", ".join(f"`{value}`" for value in discriminator["mapping"])
            result += f" (`{discriminator['propertyName']}`: {variants})"
        return result
    for keyword, separator in (("oneOf", " | "), ("anyOf", " | "), ("allOf", " & ")):
        if keyword in schema:
            return separator.join(_type(branch, components) for branch in schema[keyword])
    if "const" in schema:
        return f"`{json.dumps(schema['const'])}`"
    if "enum" in schema:
        return " | ".join(f"`{json.dumps(value)}`" for value in schema["enum"])
    if not schema:
        return "`unknown`"
    kind = schema["type"]
    if kind == "array":
        result = f"array<{_type(schema['items'], components)}>"
    elif kind == "object" and isinstance(schema.get("additionalProperties"), dict):
        key = (
            _type(schema["propertyNames"], components) if "propertyNames" in schema else "`string`"
        )
        result = f"map<{key}, {_type(schema['additionalProperties'], components)}>"
    else:
        result = f"`{kind}`"
    constraints = {
        "minimum": "≥",
        "exclusiveMinimum": ">",
        "maximum": "≤",
        "exclusiveMaximum": "<",
        "minItems": "min items",
        "maxItems": "max items",
        "minProperties": "min entries",
        "maxProperties": "max entries",
        "format": "format",
    }
    limits = [f"{label} {schema[key]}" for key, label in constraints.items() if key in schema]
    return result + (f" ({', '.join(limits)})" if limits else "")


def _fields(label: str, schema: JsonSchemaValue, components: JsonSchemaValue) -> str:
    definition = _resolve(schema, components)
    blocks = [f"**{label}: {_type(schema, components)}**"]
    if "description" in definition:
        blocks.append(definition["description"])
    rows = ["| Field | Type | Presence / default | Description |", "|---|---|---|---|"]
    for name, field in definition["properties"].items():
        presence = "required" if name in definition.get("required", []) else "optional"
        if "default" in field:
            presence += f"; default `{json.dumps(field['default'])}`"
        cells = (f"`{name}`", _type(field, components), presence, field.get("description", ""))
        rows.append(
            "| " + " | ".join(" ".join(cell.split()).replace("|", r"\|") for cell in cells) + " |"
        )
    blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def render_action_types(operation: JsonSchemaValue, components: JsonSchemaValue) -> str:
    """Select the action's own input and success body through declared discriminators."""
    action = operation["operationId"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    request_ref = request_schema["discriminator"]["mapping"][action]
    request = _resolve({"$ref": request_ref}, components)
    response_schema = operation["responses"]["200"]["content"]["application/msgpack"]["schema"]
    response = _resolve(response_schema, components)
    success = _resolve({"$ref": response["discriminator"]["mapping"]["success"]}, components)
    result = _resolve({"$ref": success["discriminator"]["mapping"][action]}, components)
    return "\n\n".join(
        (
            _fields("Input", request["properties"]["input"], components),
            _fields("Success body", result["properties"]["body"], components),
        )
    )
