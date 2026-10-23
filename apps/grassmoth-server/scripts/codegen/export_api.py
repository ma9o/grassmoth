"""Export one OpenAPI contract graph and the curl skill.

Run ``bun run codegen`` or ``bun run codegen:check`` from the repository root.
All Python exports are generated together before their TypeScript consumers.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAliasType

from fastapi import routing
from fastapi.openapi.utils import get_fields_from_routes, get_openapi_path
from pydantic import TypeAdapter

from grassmoth.public import (
    CausalEdgeSpec,
    CausalEffectResult,
    ConstructSpec,
    DataDiffOutput,
    EffectSummary,
    GitOid,
    IndicatorSpec,
    ModelDiffOutput,
    ObservationHistory,
    ParameterSpec,
)
from grassmoth_server.actions.contracts.polling import ActionPoll
from grassmoth_server.actions.contracts.progress import ProgressEvent
from grassmoth_server.actions.contracts.requests import DataDiffRequest, ModelDiffRequest
from grassmoth_server.actions.contracts.timeline import TimelineResponse, TimelineRevision
from grassmoth_server.api.schema import with_json_responses
from grassmoth_server.llm.traces import LLMTrace
from grassmoth_server.study.records import StudyRevision
from grassmoth_server.study.snapshot import ModelSnapshot
from scripts.artifact_contracts import ARTIFACT_CONTRACTS
from scripts.codegen.skill_types import render_action_types
from scripts.codegen.type_system_catalog import (
    ContractJsonSchema,
    annotate_definitions,
    generic_types,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pydantic import BaseModel
    from pydantic.json_schema import JsonSchemaMode, JsonSchemaValue
    from pydantic_core import CoreSchema

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "packages" / "api-types" / "schemas"
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "grassmoth-study-api" / "SKILL.md"

EXPORTED_API_MODELS: tuple[type[BaseModel] | TypeAliasType, ...] = (
    TimelineResponse,
    TimelineRevision,
    StudyRevision,
    ModelDiffOutput,
    ModelDiffRequest[GitOid],
    DataDiffOutput,
    DataDiffRequest[GitOid],
    LLMTrace,
    ModelSnapshot,
    ObservationHistory,
    ActionPoll,
    ProgressEvent,
    ConstructSpec,
    CausalEdgeSpec,
    IndicatorSpec,
    ParameterSpec,
)

EXPORTED_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    EffectSummary,
    CausalEffectResult,
)


def export_openapi() -> JsonSchemaValue:
    """Generate HTTP fields, stored roots and generic bodies together, once."""
    from grassmoth_server.api.app import app

    fields = get_fields_from_routes(app.routes)
    roots = (*ARTIFACT_CONTRACTS.values(), *EXPORTED_API_MODELS, *EXPORTED_RESULT_MODELS)
    generator = ContractJsonSchema()
    for field in fields:
        generator.register(field.field_info.annotation)
    for root in roots:
        generator.register(root)
    templates = generic_types(generator.generic_types)
    for template in templates.values():
        generator.register(template)
    inputs: list[tuple[Hashable, JsonSchemaMode, CoreSchema]] = (
        [(field, field.mode, field._type_adapter.core_schema) for field in fields]
        + [(root, "serialization", TypeAdapter(root).core_schema) for root in roots]
        + [
            (name, "serialization", TypeAdapter(template).core_schema)
            for name, template in templates.items()
        ]
    )
    field_mapping, generated = generator.generate_definitions(inputs=inputs)
    definitions = {str(name): definition for name, definition in generated.items()}
    root_refs: JsonSchemaValue = {}
    for root in roots:
        schema = field_mapping[root, "serialization"]
        if "$ref" not in schema:
            definitions[root.__name__] = schema
            schema = {"$ref": f"#/components/schemas/{root.__name__}"}
        root_refs[root.__name__] = schema
    generic_refs: JsonSchemaValue = {}
    for name, template in templates.items():
        schema = field_mapping[name, "serialization"]
        if "$ref" in schema:
            component = schema["$ref"].rsplit("/", 1)[-1]
        else:
            component = name
            definitions[component] = schema
        definition = definitions[component]
        definition.pop("x-typescript-type", None)
        definition["x-python-module"] = template.__module__
        definition["x-typescript-parameters"] = [
            p.__name__ for p in generator.generic_types[name].__type_params__
        ]
        generic_refs[name] = {"$ref": f"#/components/schemas/{component}"}
    annotate_definitions(
        {
            name: definition
            for name, definition in definitions.items()
            if not name.startswith("Body_")
        }
    )
    paths: dict[str, dict[str, Any]] = {}
    security_schemes: JsonSchemaValue = {}
    operation_ids: set[str] = set()
    http_fields = {(field, field.mode): field_mapping[field, field.mode] for field in fields}
    for route in routing.iter_route_contexts(app.routes):
        if not isinstance(route.original_route, routing.APIRoute):
            continue
        path, security, extra_definitions = get_openapi_path(
            route=route,  # ty: ignore[invalid-argument-type] -- FastAPI uses RouteContext here; its private protocol wrongly requires writable forwarded properties.
            operation_ids=operation_ids,
            model_name_map={},
            field_mapping=http_fields,
        )
        if path:
            route_path = route.path_format
            assert route_path is not None, "An API route context must have a path"
            paths.setdefault(route_path, {}).update(path)
        security_schemes.update(security)
        definitions.update(extra_definitions)
    components: JsonSchemaValue = {"schemas": dict(sorted(definitions.items()))}
    if security_schemes:
        components["securitySchemes"] = security_schemes
    return with_json_responses(
        {
            "openapi": app.openapi_version,
            "info": {"title": app.title, "version": app.version, "description": app.description},
            "paths": paths,
            "components": components,
            "x-contract-roots": root_refs,
            "x-typescript-generics": generic_refs,
        }
    )


_BASE_URL = "${TOOL_SERVER_URL:-http://localhost:8100}"
_MAX_EXAMPLE_DEPTH = 8


def _example_from_schema(
    schema: JsonSchemaValue,
    components: JsonSchemaValue,
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> Any:
    """A minimal JSON skeleton satisfying a schema, for a copy-pasteable body.

    Required object properties only, first branch of a union, first enum value.
    The authoritative, hand-written body examples live in the app description;
    this is just a shape hint next to each endpoint.
    """
    if depth > _MAX_EXAMPLE_DEPTH:
        return {}

    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            return {}
        return _example_from_schema(
            components.get(name, {}), components, depth=depth, seen=seen | {name}
        )

    for combinator in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(combinator)
        if branches:
            non_null = [b for b in branches if b.get("type") != "null"] or branches
            return _example_from_schema(non_null[0], components, depth=depth + 1, seen=seen)

    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if enum:
        return enum[0]

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        props: JsonSchemaValue = schema.get("properties", {})
        required = set(schema.get("required", []))
        # Include required fields plus any discriminator/fixed-value field (a
        # `const`, e.g. move `kind`) even when a default makes it non-required,
        # so the skeleton is a valid body. Property order is preserved.
        return {
            name: _example_from_schema(prop, components, depth=depth + 1, seen=seen)
            for name, prop in props.items()
            if name in required or "const" in prop
        }
    if schema_type == "array":
        items = schema.get("items")
        return (
            [_example_from_schema(items, components, depth=depth + 1, seen=seen)] if items else []
        )
    if schema_type == "integer" or schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "string"


def _path_with_placeholders(path: str, parameters: list[JsonSchemaValue]) -> str:
    """Substitute path params with uppercase placeholders for the curl example."""
    result = path
    for param in parameters:
        if param.get("in") == "path":
            name = param["name"]
            result = result.replace("{" + name + "}", name.upper())
    return result


def _curl_block(
    method: str,
    path: str,
    operation: JsonSchemaValue,
    components: JsonSchemaValue,
) -> str:
    parameters = operation.get("parameters", [])
    url = f"{_BASE_URL}{_path_with_placeholders(path, parameters)}"
    lines = [f'curl -s "{url}"']
    if "application/msgpack" in operation.get("responses", {}).get("200", {}).get("content", {}):
        lines[0] += " -H 'Accept: application/json' -o /tmp/action.json"
    if method != "get":
        lines[0] += " \\"
        lines.append(f"  -X {method.upper()} \\")
        contents = operation.get("requestBody", {}).get("content", {})
        if "multipart/form-data" in contents:
            schema = contents["multipart/form-data"]["schema"]
            if "$ref" in schema:
                schema = components[schema["$ref"].rsplit("/", 1)[-1]]
            fields = schema.get("properties", {})
            for index, (name, field) in enumerate(fields.items()):
                value = (
                    "@/path/to/file"
                    if field.get("contentMediaType") == "application/octet-stream"
                    else name.upper()
                )
                suffix = " \\" if index < len(fields) - 1 else ""
                lines.append(f'  -F "{name}={value}"{suffix}')
            return "```bash\n" + "\n".join(lines) + "\n```"
        lines.append("  -H 'Content-Type: application/json' \\")
        body_schema = (
            operation.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        example = _example_from_schema(body_schema, components) if body_schema else {}
        lines.append(f"  -d '{json.dumps(example)}'")
    return "```bash\n" + "\n".join(lines) + "\n```"


def _parameters_block(operation: JsonSchemaValue) -> str | None:
    parameters = operation.get("parameters", [])
    if not parameters:
        return None
    rows = ["**Parameters**", ""]
    for param in parameters:
        location = param.get("in", "query")
        required = "required" if param.get("required") else "optional"
        description = param.get("description", "").strip().replace("\n", " ")
        suffix = f" — {description}" if description else ""
        rows.append(f"- `{param['name']}` ({location}, {required}){suffix}")
    return "\n".join(rows)


def _skill_frontmatter() -> str:
    """YAML frontmatter for the Agent Skill.

    `name` and `description` satisfy both the Codex requirement and Claude Code's
    trigger heuristic; the description says what the skill does and when to reach
    for it. Kept as a controlled single-line double-quoted scalar for maximal
    cross-parser compatibility.
    """
    description = (
        "Run and inspect grassmoth studies through the HTTP API with curl. "
        "Use for model editing, data preparation, fitting, simulation and saved results."
    )
    return f'---\nname: grassmoth-study-api\ndescription: "{description}"\n---'


def render_skill(openapi: JsonSchemaValue) -> str:
    info = openapi.get("info", {})
    components = openapi.get("components", {}).get("schemas", {})
    title = info.get("title", "Agent API")

    blocks: list[str] = [
        _skill_frontmatter(),
        f"# {title} — curl skill",
        (
            "> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI "
            "OpenAPI spec) by `apps/grassmoth-server/scripts/codegen/export_api.py`. "
            "Edit Python type/field descriptions, the API description or route docstrings, "
            "not this file."
        ),
        inspect.cleandoc(info.get("description", "")),
        "## Endpoints",
    ]

    for path in sorted(openapi.get("paths", {})):
        methods = openapi["paths"][path]
        for method in sorted(methods):
            operation = methods[method]
            if not isinstance(operation, dict):
                continue
            blocks.append(f"### {method.upper()} `{path}`")
            description = inspect.cleandoc(
                operation.get("description") or operation.get("summary") or ""
            )
            if description:
                blocks.append(description)
            params = _parameters_block(operation)
            if params:
                blocks.append(params)
            if method == "post" and path.startswith("/api/studies/"):
                blocks.append(render_action_types(operation, components))
            blocks.append(_curl_block(method, path, operation, components))

    text = "\n\n".join(block for block in blocks if block)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n") + "\n"


def main(*, check: bool = False) -> bool:
    openapi = export_openapi()
    outputs = {
        OUTPUT_DIR / "openapi.json": json.dumps(openapi, indent=2) + "\n",
        SKILL_PATH: render_skill(openapi),
    }
    changed = []
    for path, rendered in outputs.items():
        if check:
            if not path.exists() or path.read_text() != rendered:
                changed.append(path.relative_to(REPO_ROOT))
        else:
            if path.exists() and path.read_text() == rendered:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
    if changed:
        print("API exports are out of date. Run `bun run codegen`.", file=sys.stderr)
        for path in changed:
            print(f"  {path}", file=sys.stderr)
        return True
    print(f"API exports {'checked' if check else 'written'}: {len(outputs)} files")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify generated files without writing."
    )
    raise SystemExit(main(check=parser.parse_args().check))
