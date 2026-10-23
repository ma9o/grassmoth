"""Derive JSON response schemas from the canonical binary response graph."""

from typing import override

from fastapi import FastAPI
from pydantic.json_schema import JsonSchemaValue

from grassmoth.public import NumericalArray

_COMPONENT = "#/components/schemas/"


def _references(value: object) -> frozenset[str]:
    if isinstance(value, str) and value.startswith(_COMPONENT):
        return frozenset((value.removeprefix(_COMPONENT),))
    if isinstance(value, dict):
        return frozenset[str]().union(*(_references(item) for item in value.values()))
    if isinstance(value, list):
        return frozenset[str]().union(*(_references(item) for item in value))
    return frozenset[str]()


def with_json_responses(openapi: JsonSchemaValue) -> JsonSchemaValue:
    """Project only array-dependent components; request and binary schemas stay intact."""
    definitions = openapi["components"]["schemas"]
    array_schema = NumericalArray.model_json_schema()
    arrays = frozenset(
        name for name, schema in definitions.items() if schema.get("title") == array_schema["title"]
    )
    dependencies = {name: _references(schema) for name, schema in definitions.items()}
    reachable = frozenset[str]().union(
        *(
            _references(
                operation.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/msgpack", {})
            )
            for operations in openapi["paths"].values()
            for operation in operations.values()
        )
    )
    while (
        descendants := frozenset[str]().union(*(dependencies[name] for name in reachable))
        - reachable
    ):
        reachable |= descendants
    affected = arrays & reachable
    while (
        expanded := frozenset(
            name for name, refs in dependencies.items() if name in reachable and refs & affected
        )
        - affected
    ):
        affected |= expanded

    def project(value: object) -> object:
        if (
            isinstance(value, str)
            and value.startswith(_COMPONENT)
            and value.removeprefix(_COMPONENT) in affected
        ):
            return f"{_COMPONENT}JsonResponse_{value.removeprefix(_COMPONENT)}"
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items()}
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    variants = {
        f"JsonResponse_{name}": (
            {
                "title": f"JsonResponse_{name}",
                "description": NumericalArray.model_fields["npy"].description,
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "number"},
                    {"type": "null"},
                    {"type": "array", "items": {"$ref": f"{_COMPONENT}JsonResponse_{name}"}},
                ],
            }
            if name in arrays
            else project(definitions[name])
        )
        for name in sorted(affected)
    }
    for operations in openapi["paths"].values():
        for operation in operations.values():
            content = operation.get("responses", {}).get("200", {}).get("content", {})
            if "application/msgpack" in content:
                content["application/json"] = project(content["application/msgpack"])
    definitions.update(variants)
    return openapi


class StudyAPI(FastAPI):
    """Expose the negotiated response representations in the live OpenAPI document."""

    @override
    def openapi(self) -> JsonSchemaValue:
        if self.openapi_schema is None:
            self.openapi_schema = with_json_responses(super().openapi())
        return self.openapi_schema
