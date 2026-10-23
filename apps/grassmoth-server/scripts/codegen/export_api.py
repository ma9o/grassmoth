"""Export one OpenAPI contract graph, the study API reference and the curl skill.

Run ``bun run codegen`` or ``bun run codegen:check`` from the repository root.
All Python exports are generated together before their TypeScript consumers.
"""

from __future__ import annotations

import argparse
import inspect
import json
import posixpath
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
from scripts.codegen.export_story_scenarios import examples
from scripts.codegen.reference_types import render_action_types
from scripts.codegen.type_system_catalog import (
    ContractJsonSchema,
    annotate_definitions,
    generic_types,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from pydantic import BaseModel
    from pydantic.json_schema import JsonSchemaMode, JsonSchemaValue
    from pydantic_core import CoreSchema

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "packages" / "api-types" / "schemas"
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "grassmoth-study-api" / "SKILL.md"
REFERENCE_PATH = REPO_ROOT / "docs" / "reference" / "study-api.md"

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


def _example_requests() -> Mapping[str, JsonSchemaValue]:
    """Each action's first synthetic scenario request, as an HTTP caller sends it.

    Saved preparation requests capture source files; callers name their folder.
    """
    requests: dict[str, JsonSchemaValue] = {}
    for example in examples():
        body = example.request.model_dump(mode="json")
        if body["action"] in requests:
            continue
        del body["reasoning"]
        if body["action"] == "prepare_data":
            files = body["input"]["source"]["files"]
            body["input"]["source"] = posixpath.commonpath(
                [posixpath.dirname(path) for path in files]
            )
        requests[body["action"]] = body
    return requests


def _enum(schema: JsonSchemaValue, components: JsonSchemaValue) -> tuple[str, ...]:
    """Enum values in declaration order, through references and unions."""
    if "$ref" in schema:
        return _enum(components[schema["$ref"].rsplit("/", 1)[-1]], components)
    if "anyOf" in schema:
        return tuple(value for branch in schema["anyOf"] for value in _enum(branch, components))
    return tuple(schema["enum"])


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
    requests: Mapping[str, JsonSchemaValue],
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
        lines.append("  --data-binary @- <<'JSON'")
        lines.append(json.dumps(requests[operation["operationId"]], indent=2))
        lines.append("JSON")
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


_GENERATED = (
    "> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI "
    "OpenAPI spec) by `apps/grassmoth-server/scripts/codegen/export_api.py`. "
    "Edit Python type/field descriptions, the API description or route docstrings, "
    "not this file."
)


def _document(blocks: Iterable[str | None]) -> str:
    text = "\n\n".join(block for block in blocks if block)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n") + "\n"


def _link(target: Path, document: Path) -> str:
    return target.relative_to(document.parent, walk_up=True).as_posix()


def render_skill(openapi: JsonSchemaValue) -> str:
    """Operating rules and pointers; per-endpoint detail lives in the reference."""
    info = openapi["info"]
    components = openapi["components"]["schemas"]
    docs = REPO_ROOT / "docs"
    flows = docs / "assets" / "action-flows"
    charts = ", ".join(
        f"[`{action}`]({_link(flows / (action.replace('_', '-') + '.svg'), SKILL_PATH)})"
        for action in _enum(components["ActionId"], components)
    )
    spec = docs / "reference" / "statistical-model-spec"
    references = (
        f"- The [study API reference]({_link(REFERENCE_PATH, SKILL_PATH)}) lists every "
        "endpoint's input and success-body fields with an example request. Read an action's "
        "section before calling it.",
        f"- Each action's flow chart shows its checks, branches and failure outcomes: {charts}.",
        f"- [Assumptions and limits]({_link(docs / 'assumptions.md', SKILL_PATH)}) state the "
        "modeling commitments behind every result.",
        f"- The [likelihood]({_link(spec / 'likelihoods.md', SKILL_PATH)}) and "
        f"[parameter and prior]({_link(spec / 'parameters.md', SKILL_PATH)}) references "
        "describe parts of the `dynamical_model_spec` language.",
    )
    return _document(
        (
            _skill_frontmatter(),
            f"# {info['title']}",
            _GENERATED,
            inspect.cleandoc(info["description"]),
            "## References",
            "\n".join(references),
        )
    )


def render_reference(openapi: JsonSchemaValue) -> str:
    """Every endpoint with its typed input, success body and an example request."""
    info = openapi["info"]
    components = openapi["components"]["schemas"]
    requests = _example_requests()
    scenarios = _link(Path(__file__).with_name("story_data.py"), REFERENCE_PATH)
    blocks: list[str | None] = [
        f"# {info['title']} reference",
        _GENERATED,
        inspect.cleandoc(info["description"]),
        "## Endpoints",
        f"Example requests come from the [synthetic interface scenarios]({scenarios}). Replace "
        "their revision hashes with `latest` or revisions from your study's timeline.",
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
            blocks.append(_curl_block(method, path, operation, components, requests))

    return _document(blocks)


def main(*, check: bool = False) -> bool:
    openapi = export_openapi()
    outputs = {
        OUTPUT_DIR / "openapi.json": json.dumps(openapi, indent=2) + "\n",
        REFERENCE_PATH: render_reference(openapi),
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
