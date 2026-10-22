"""Field proofs, ownership, and production TypeScript consumers."""

from __future__ import annotations

import ast
import json
import subprocess
from typing import override

import pytest

from scripts.checks import check_fields as checker

pytestmark = pytest.mark.contract


def _source(tmp_path, monkeypatch, text: str) -> checker.SourceFile:
    monkeypatch.setattr(checker, "SOURCE_ROOTS", (tmp_path,))
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    path = tmp_path / "domain.py"
    path.write_text(text)
    return checker._source_file(path)


def test_equality_proofs_are_scoped_to_the_validator_and_require_an_invariant(
    tmp_path, monkeypatch
):
    source = _source(
        tmp_path,
        monkeypatch,
        """
from pydantic import BaseModel, model_validator
class Report(BaseModel):
    data: tuple[int, ...]
    cached: tuple[int, ...]
    size: int
    health: int

    @model_validator(mode="after")
    def check_projection(self):
        expected = tuple(self.data)
        if self.cached != expected:
            raise ValueError("duplicate projection")
        assert self.size == len(self.data)
        return self

    @model_validator(mode="after")
    def check_health(self):
        expected = external_health()
        if self.health != expected:
            raise ValueError("independent input")
        if self.health != len(self.data) and self.size == 0:
            raise ValueError("conditional constraint")
        return self
""",
    )
    fields = checker.collect_fields((source,))
    assert {finding.field.name for finding in checker.recomputed_fields(fields)} == {
        "cached",
        "size",
    }
    cached = next(field for field in fields if field.name == "cached")
    validation_read = next(
        node
        for node in ast.walk(source.tree)
        if isinstance(node, ast.Attribute) and node.attr == "cached"
    )
    assert checker.classify_reference(cached, source, validation_read) == "validation"


def test_aliases_generics_and_inherited_exports_keep_the_declaring_owner(tmp_path, monkeypatch):
    source = _source(
        tmp_path,
        monkeypatch,
        """
from typing import Annotated, Literal
from pydantic import BaseModel, Field
type LeftTag = Literal["left"]
class Parent(BaseModel):
    label: Annotated[str, Field(alias="wire_label")]
class Left(Parent):
    kind: LeftTag = "left"
class Right(Parent):
    kind: Literal["right"] = "right"
LeftAlias = Left
type Choice = LeftAlias | Right
class Envelope[T](BaseModel):
    value: T
""",
    )
    exports = tmp_path / "exports.py"
    exports.write_text("from domain import Parent as ParentAlias\n")
    derived = tmp_path / "derived.py"
    derived.write_text("from exports import ParentAlias as Base\nclass Imported(Base):\n    pass\n")
    sources = (source, checker._source_file(exports), checker._source_file(derived))
    fields = checker.collect_fields(sources)
    schemas = {
        "Left-Output": {
            "title": "Left-Output",
            "x-python-module": "domain",
            "properties": {"kind": {}, "wire_label": {}},
        },
        "Envelope_Left_": {
            "title": "Envelope[Left]",
            "x-python-module": "domain",
            "properties": {"value": {}},
        },
        "Imported-Output": {
            "title": "Imported-Output",
            "x-python-module": "derived",
            "properties": {"wire_label": {}},
        },
    }
    api = {
        "components": {"schemas": schemas},
        "x-typescript-generics": {"Envelope": {"$ref": "#/components/schemas/Envelope_Left_"}},
    }
    assert checker.discriminators(fields, schemas, sources) == frozenset(
        {"domain.Left.kind", "domain.Right.kind"}
    )
    exported = checker.exported_fields(fields, api, sources)
    assert exported["domain.Parent.label"]["aliases"] == {"wire_label"}
    assert exported["domain.Envelope.value"]["components"] == {"Envelope_Left_"}
    assert "domain.Left.label" not in exported
    assert "Imported-Output" in exported["domain.Parent.label"]["components"]


class _TypeInformation(checker.PythonReferences):
    """Deterministic language-server responses for producer proof boundaries."""

    def __init__(self, source: checker.SourceFile) -> None:
        self.source = source

    @override  # noqa: V105 -- constructor_constants calls this language-server protocol method.
    def hover(self, source: checker.SourceFile, node: ast.AST) -> str:
        return "(variable) configured_method: Literal['exact']"

    @override
    def definitions(self, source: checker.SourceFile, node: ast.AST) -> tuple[()]:
        return ()

    @override  # noqa: V105 -- producer_inventory calls this language-server protocol method.
    def type_definitions(
        self, source: checker.SourceFile, node: ast.AST
    ) -> tuple[dict[str, object], ...]:
        owner = {"record": "Record", "changed": "Record"}.get(ast.unparse(node))
        if owner is None:
            return ()
        declaration = next(
            node
            for node in self.source.tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
        return (
            {
                "uri": self.source.path.as_uri(),
                "range": {"start": {"line": declaration.lineno - 1, "character": 6}},
            },
        )


def test_producer_aliases_defaults_factories_and_revision_writes(tmp_path, monkeypatch):
    source = _source(
        tmp_path,
        monkeypatch,
        """
from pydantic import BaseModel, Field, TypeAdapter
class Record(BaseModel):
    method: str
    count: int = 0
class Closed(BaseModel):
    method: str
    count: int = 0
class FactoryRecord(BaseModel):
    count: int = 0
class Escaped(BaseModel):
    count: int = 0
class Container(BaseModel):
    child: FactoryRecord = Field(default_factory=FactoryRecord)
Alias = Closed
factory = Escaped
closed = Alias(method=configured_method)
record = Record(method=configured_method)
record.revised(count=user_count)
TypeAdapter(Record).validate_python(external_input)
""",
    )
    fields = checker.collect_fields((source,))
    server = _TypeInformation(source)
    producers, open_fields = checker.producer_inventory(fields, (source,), server)
    by_name = {field.identity: field for field in fields}
    assert "domain.Closed.count" not in open_fields
    count = checker.constructor_constants(
        by_name["domain.Closed.count"], producers["domain.Closed.count"], server
    )
    method = checker.constructor_constants(
        by_name["domain.Closed.method"], producers["domain.Closed.method"], server
    )
    assert count is not None
    assert count[0] == "0"
    assert method is not None
    assert method[0] == "'exact'"
    assert {
        "domain.Record.method",
        "domain.Record.count",
        "domain.FactoryRecord.count",
        "domain.Escaped.count",
    } <= open_fields


def test_typescript_reads_are_owned_and_only_production_reads_count(tmp_path):
    web = tmp_path / "apps/grassmoth-client"
    generated = tmp_path / "packages/api-types/src/generated"
    (web / "src").mkdir(parents=True)
    generated.mkdir(parents=True)
    (web / "tsconfig.json").write_text(
        json.dumps({"compilerOptions": {"strict": True, "target": "ESNext"}, "include": ["src"]})
    )
    (tmp_path / "knip.json").write_text(
        json.dumps(
            {
                "workspaces": {
                    "apps/grassmoth-client": {"project": ["src/**/*.ts", "!src/**/*.test.ts"]}
                }
            }
        )
    )
    (generated / "models.ts").write_text(
        "export interface Alpha { label: string; }\n"
        "export interface Beta { label: string; }\n"
        "export interface Aliased { wire_name: string; }\n"
        "export interface Opaque { raw: number; }\n"
        "export interface Structural { source: number; }\n"
    )
    imports = 'import type { Alpha, Beta, Aliased, Opaque, Structural } from "../../../packages/api-types/src/generated/models";\n'
    (web / "src/readers.ts").write_text(
        imports
        + "export function render(a: Alpha, b: Beta, alias: Aliased, opaque: Opaque) {\n"
        + '  a.label = "written";\n'
        + "  const erased = (value: unknown) => String(value);\n"
        + "  erased(opaque);\n"
        + "  return b.label + alias.wire_name;\n}\n"
        + "export function structural(s: Structural) {\n"
        + "  const render = (entry: { source: number }) => entry.source;\n"
        + "  return render(s);\n}\n"
    )
    (web / "src/readers.test.ts").write_text(
        imports + "export const onlyInTest = (a: Alpha) => a.label;\n"
    )
    requested = [
        {
            "identity": f"domain.{owner}.{field}",
            "name": field,
            "owner": owner,
            "aliases": [alias],
            "components": [owner],
        }
        for owner, field, alias in (
            ("Alpha", "label", "label"),
            ("Beta", "label", "label"),
            ("Aliased", "python_name", "wire_name"),
            ("Opaque", "raw", "raw"),
            ("Structural", "source", "source"),
        )
    ]
    completed = subprocess.run(
        ["bun", str(checker.WEB_READERS), str(tmp_path)],
        input=json.dumps(requested),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert set(result["readers"]) == {"domain.Beta.label", "domain.Aliased.python_name"}
    assert set(result["opaque"]) == {"domain.Opaque.raw", "domain.Structural.source"}
