"""Reject duplicated, constant, and unread production fields.

FIELD001
    Stored fields cannot be validated equal to another owned field or a pure
    recomputation of owned fields. Compute the projection at its owner. When
    equality applies to only one alternative, split the alternatives instead.

FIELD002
    A field cannot have a single-valued Literal unless it distinguishes union
    alternatives. Nor can every resolved production constructor supply the same
    constant. Defaults participate when a constructor omits the field. Literal
    types on producer expressions are resolved by basedpyright, without a second
    dataflow engine. Delete the root constant and its copies, or compute an output.

FIELD003
    An exported field (an OpenAPI component with x-python-module) needs a
    production reader in Python src or knip's production web project. Constructor
    keywords, revisions, declarations, serialization, and reads in the owner's
    validators or __post_init__ are not readers. Python references come from
    basedpyright's language server; web references come from TypeScript's language
    service. Aliases, generic owners and input/output schemas retain their owner.
    Explicit reflective reads and erased model dumps are opaque, not findings.

Diagnostics include the recomputation, producer sites, or "no reader". The only
suppression is an inline ``# noqa: FIELD00N -- genuine false-positive reason``;
suppressed findings and their reasons are printed. There is no baseline.

Usage: bun run --cwd apps/grassmoth-server lint:fields
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOTS = (ROOT / "packages/grassmoth/src", ROOT / "apps/grassmoth-server/src")
OPENAPI = ROOT / "packages/api-types/schemas/openapi.json"
WEB_READERS = Path(__file__).with_name("field_web_readers.ts")
RULES = ("FIELD001", "FIELD002", "FIELD003")
VALIDATORS = frozenset({"model_validator", "field_validator", "root_validator", "validator"})
MODEL_BASES = frozenset(
    {"BaseModel", "RootModel", "GenericModel", "TypedDict", "NamedTuple", "Protocol"}
)
_SUPPRESSION = re.compile(r"#\s*noqa:\s*([^#]+?)\s+--\s+(.+)$")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    module: str
    text: str
    tree: ast.Module
    parents: Mapping[ast.AST, ast.AST]
    bindings: Mapping[str, str]


@dataclass(frozen=True)
class Field:
    source: SourceFile
    owner: ast.ClassDef
    name: str
    node: ast.AnnAssign | ast.FunctionDef | ast.AsyncFunctionDef
    annotation: ast.expr | None
    default: ast.expr | None
    computed: bool = False

    @property
    def identity(self) -> str:
        return f"{self.source.module}.{self.owner.name}.{self.name}"

    @property
    def position(self) -> tuple[int, int]:
        if isinstance(self.node, ast.AnnAssign):
            return _lsp_position(self.source, self.node.target)
        line = self.source.text.splitlines()[self.node.lineno - 1]
        column = line.index(self.name, self.node.col_offset)
        return self.node.lineno - 1, len(line[:column].encode("utf-16-le")) // 2


@dataclass(frozen=True)
class Finding:
    field: Field
    code: str
    evidence: str

    def diagnostic(self) -> str:
        line, column = self.field.position
        relative = self.field.source.path.relative_to(ROOT)
        return (
            f"{relative}:{line + 1}:{column + 1}: {self.code} {self.evidence} "
            f"[{self.field.identity}]"
        )


@dataclass(frozen=True)
class FieldOutcome:
    field: Field
    status: str
    evidence: str


@dataclass(frozen=True)
class CheckResult:
    findings: tuple[Finding, ...]
    suppressed: tuple[tuple[Finding, str], ...]
    outcomes: tuple[FieldOutcome, ...]


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _ancestors(source: SourceFile, node: ast.AST) -> Iterable[ast.AST]:
    while node in source.parents:
        node = source.parents[node]
        yield node


def _lsp_position(source: SourceFile, node: ast.AST) -> tuple[int, int]:
    assert isinstance(node, (ast.expr, ast.stmt, ast.arg, ast.keyword, ast.alias, ast.pattern))
    if isinstance(node, ast.Attribute):
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        line = node.end_lineno - 1
        column = node.end_col_offset - len(node.attr.encode("utf-8"))
    else:
        line, column = node.lineno - 1, node.col_offset
    prefix = source.text.splitlines()[line].encode("utf-8")[:column].decode("utf-8")
    return line, len(prefix.encode("utf-16-le")) // 2


def _source_file(path: Path) -> SourceFile:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    module = ".".join(
        path.relative_to(next(root for root in SOURCE_ROOTS if path.is_relative_to(root)))
        .with_suffix("")
        .parts
    )
    if module.endswith(".__init__"):
        module = module.removesuffix(".__init__")
    bindings: dict[str, str] = {}
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                prefix = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1])
                if node.module:
                    prefix += "." + node.module
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{prefix}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    return SourceFile(path, module, text, tree, parents, bindings)


def _resolve(source: SourceFile, node: ast.AST) -> str:
    if isinstance(node, ast.Subscript):
        return _resolve(source, node.value)
    if isinstance(node, ast.Name):
        return source.bindings.get(node.id, f"{source.module}.{node.id}")
    if isinstance(node, ast.Attribute):
        return f"{_resolve(source, node.value)}.{node.attr}"
    return ""


def _type_aliases(sources: Sequence[SourceFile]) -> dict[str, tuple[SourceFile, ast.expr]]:
    aliases: dict[str, tuple[SourceFile, ast.expr]] = {}
    for source in sources:
        for name in source.bindings:
            if name != "*":
                aliases[f"{source.module}.{name}"] = source, ast.Name(id=name, ctx=ast.Load())
        for node in source.tree.body:
            if isinstance(node, ast.TypeAlias):
                aliases[f"{source.module}.{node.name.id}"] = source, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id[:1].isupper()
                    and isinstance(node.value, ast.Name | ast.Attribute | ast.Subscript | ast.BinOp)
                ):
                    aliases[f"{source.module}.{target.id}"] = source, node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and _name(node.annotation) == "TypeAlias"
                and node.value is not None
            ):
                aliases[f"{source.module}.{node.target.id}"] = source, node.value
    return aliases


def _type_owners(
    source: SourceFile,
    node: ast.AST,
    aliases: Mapping[str, tuple[SourceFile, ast.expr]],
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Expand declared type aliases, including generic union alternatives."""
    identity = _resolve(source, node)
    if identity in aliases and identity not in seen:
        origin, expression = aliases[identity]
        return _type_owners(origin, expression, aliases, seen | {identity})
    if isinstance(node, ast.Name | ast.Attribute):
        return frozenset({identity})
    if isinstance(node, ast.Subscript):
        return _type_owners(source, node.value, aliases, seen) | _type_owners(
            source, node.slice, aliases, seen
        )
    return frozenset().union(
        *(_type_owners(source, child, aliases, seen) for child in ast.iter_child_nodes(node))
    )


def _fields_by_owner(
    fields: Sequence[Field], sources: Sequence[SourceFile]
) -> dict[str, dict[str, Field]]:
    """Inherited exports still refer to the field's original declaration."""
    result: dict[str, dict[str, Field]] = {}
    aliases = _type_aliases(sources)
    for field in fields:
        result.setdefault(field.identity.rpartition(".")[0], {})[field.name] = field
    classes = {
        f"{source.module}.{node.name}": (source, node)
        for source in sources
        for node in ast.walk(source.tree)
        if isinstance(node, ast.ClassDef)
    }
    changed = True
    while changed:
        changed = False
        for identity, (source, owner) in classes.items():
            for base in owner.bases:
                parent = _constructor_owner(source, base, aliases)
                for name, field in tuple(result.get(parent, {}).items()):
                    if name not in result.setdefault(identity, {}):
                        result[identity][name] = field
                        changed = True
    return result


def collect_fields(sources: Sequence[SourceFile]) -> tuple[Field, ...]:
    aliases = _type_aliases(sources)
    classes = {
        f"{source.module}.{node.name}": (source, node)
        for source in sources
        for node in ast.walk(source.tree)
        if isinstance(node, ast.ClassDef)
    }
    known: set[str] = set()
    while True:
        added = {
            identity
            for identity, (source, node) in classes.items()
            if identity not in known
            and (
                any(
                    _name(dec) in {"dataclass", "pydantic_dataclass"} for dec in node.decorator_list
                )
                or any(
                    _name(base) in MODEL_BASES or _constructor_owner(source, base, aliases) in known
                    for base in node.bases
                )
            )
        }
        if not added:
            break
        known.update(added)
    fields: list[Field] = []
    for identity in sorted(known):
        source, owner = classes[identity]
        for node in owner.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "model_config" or _name(node.annotation) == "ClassVar":
                    continue
                fields.append(
                    Field(source, owner, node.target.id, node, node.annotation, node.value)
                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
                _name(dec) == "computed_field" for dec in node.decorator_list
            ):
                fields.append(Field(source, owner, node.name, node, node.returns, None, True))
    return tuple(fields)


def _owned_access(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
    ):
        return node.attr
    return None


def _validated(source: SourceFile, owner: ast.ClassDef, node: ast.AST) -> bool:
    for ancestor in _ancestors(source, node):
        if isinstance(ancestor, ast.FunctionDef | ast.AsyncFunctionDef):
            return source.parents.get(ancestor) is owner and (
                ancestor.name == "__post_init__"
                or any(_name(dec) in VALIDATORS for dec in ancestor.decorator_list)
            )
        if isinstance(ancestor, ast.ClassDef):
            return False
    return False


def recomputed_fields(fields: Sequence[Field]) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    by_owner: dict[tuple[Path, str], dict[str, Field]] = {}
    for field in fields:
        if not field.computed:
            by_owner.setdefault((field.source.path, field.owner.name), {})[field.name] = field
    for owned in by_owner.values():
        first = next(iter(owned.values()))
        for node in ast.walk(first.owner):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            if not isinstance(node.ops[0], ast.Eq | ast.NotEq):
                continue
            if not _validated(first.source, first.owner, node):
                continue
            validation = first.source.parents.get(node)
            if isinstance(validation, ast.Assert):
                if not isinstance(node.ops[0], ast.Eq):
                    continue
            elif isinstance(validation, ast.If) and validation.test is node:
                rejecting = (
                    validation.body if isinstance(node.ops[0], ast.NotEq) else validation.orelse
                )
                if not any(isinstance(statement, ast.Raise) for statement in rejecting):
                    continue
            else:
                # A conditional rejection of a compound predicate does not prove
                # that equality is an invariant of this field.
                continue
            method = next(
                ancestor
                for ancestor in _ancestors(first.source, node)
                if isinstance(ancestor, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            assignments: dict[str, list[ast.Assign | ast.AnnAssign]] = {}
            for statement in ast.walk(method):
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                elif isinstance(statement, ast.AnnAssign):
                    target = statement.target
                else:
                    continue
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(statement)
            aliases = {
                name: statements[0].value
                for name, statements in assignments.items()
                if len(statements) == 1
                and statements[0].lineno < node.lineno
                and statements[0].value is not None
            }
            left, right = node.left, node.comparators[0]
            if isinstance(right, ast.Name) and right.id in aliases:
                right = aliases[right.id]
            if isinstance(left, ast.Name) and left.id in aliases:
                left = aliases[left.id]
            candidates = ((left, right),) if _owned_access(left) else ((right, left),)
            for target, expression in candidates:
                name = _owned_access(target)
                dependencies = {
                    access
                    for child in ast.walk(expression)
                    if (access := _owned_access(child)) is not None
                }
                if name not in owned or not dependencies or name in dependencies:
                    continue
                if not dependencies <= owned.keys():
                    continue
                external = {
                    child.id
                    for child in ast.walk(expression)
                    if isinstance(child, ast.Name) and child.id not in {"self", "cls"}
                }
                if external - {"len", "tuple", "frozenset", "set", "sorted", "sum", "min", "max"}:
                    continue
                findings.append(
                    Finding(
                        owned[name],
                        "FIELD001",
                        f"compute field validated equal to {ast.unparse(expression)} (validator line {node.lineno})",
                    )
                )
    return tuple(findings)


def _literal(
    annotation: ast.expr | None,
    source: SourceFile | None = None,
    aliases: Mapping[str, tuple[SourceFile, ast.expr]] | None = None,
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if annotation is not None and source is not None and aliases is not None:
        identity = _resolve(source, annotation)
        if identity in aliases and identity not in seen:
            origin, expression = aliases[identity]
            return _literal(expression, origin, aliases, seen | {identity})
    if (
        isinstance(annotation, ast.Subscript)
        and _name(annotation.value) == "Annotated"
        and isinstance(annotation.slice, ast.Tuple)
    ):
        return _literal(annotation.slice.elts[0], source, aliases, seen)
    if not isinstance(annotation, ast.Subscript) or not (
        _name(annotation.value) == "Literal"
        or (
            source is not None
            and _resolve(source, annotation.value)
            in {"typing.Literal", "typing_extensions.Literal"}
        )
    ):
        return None
    values = (
        annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
    )
    return ast.unparse(values[0]) if len(values) == 1 else None


def discriminators(
    fields: Sequence[Field], schemas: Mapping[str, Any], sources: Sequence[SourceFile]
) -> frozenset[str]:
    result: set[str] = set()
    by_owner = _fields_by_owner(fields, sources)
    aliases = _type_aliases(sources)
    for source in sources:
        for node in ast.walk(source.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                pending: list[ast.expr] = [node]
            elif (
                isinstance(node, ast.Subscript)
                and _name(node.value) == "Union"
                and isinstance(node.slice, ast.Tuple)
            ):
                pending = list(node.slice.elts)
            else:
                continue
            members: list[str] = []
            while pending:
                member = pending.pop()
                if isinstance(member, ast.BinOp) and isinstance(member.op, ast.BitOr):
                    pending.extend((member.left, member.right))
                else:
                    members.extend(_type_owners(source, member, aliases))
            owners = [by_owner[member] for member in members if member in by_owner]
            if len(owners) < 2:
                continue
            for name in set.intersection(*(set(owner) for owner in owners)):
                variants = [owner[name] for owner in owners]
                values = [
                    _literal(variant.annotation, variant.source, aliases) for variant in variants
                ]
                if None not in values and len(set(values)) == len(values):
                    result.update(variant.identity for variant in variants)
    for schema in schemas.values():
        discriminator = schema.get("discriminator")
        if not discriminator:
            continue
        name = discriminator["propertyName"]
        for reference in discriminator.get("mapping", {}).values():
            component = schemas[reference.rsplit("/", 1)[-1]]
            owner = _schema_owner(component)
            if owner in by_owner and name in by_owner[owner]:
                result.add(by_owner[owner][name].identity)
    return frozenset(result)


def _schema_owner(schema: Mapping[str, Any]) -> str:
    module = schema.get("x-python-module", "")
    declaration = schema.get("x-typescript-type", schema.get("title", ""))
    name = declaration.split("<", 1)[0].split("[", 1)[0]
    name = name.removesuffix("-Input").removesuffix("-Output")
    return f"{module}.{name}" if module and name else ""


def exported_fields(
    fields: Sequence[Field], api: Mapping[str, Any], sources: Sequence[SourceFile]
) -> dict[str, dict[str, Any]]:
    by_owner = _fields_by_owner(fields, sources)
    result: dict[str, dict[str, Any]] = {}
    generic_components = {
        reference["$ref"].rsplit("/", 1)[-1]: name
        for name, reference in api.get("x-typescript-generics", {}).items()
    }
    for component_name, schema in api["components"]["schemas"].items():
        if "x-python-module" not in schema:
            continue
        if component_name in generic_components:
            owner = f"{schema['x-python-module']}.{generic_components[component_name]}"
        else:
            owner = _schema_owner(schema)
        owned = by_owner.get(owner, {})
        for alias in schema.get("properties", {}):
            for field in owned.values():
                if alias in _field_aliases(field):
                    entry = result.setdefault(
                        field.identity, {"aliases": set(), "components": set()}
                    )
                    entry["aliases"].add(alias)
                    entry["components"].add(component_name)
    return result


def _field_aliases(field: Field) -> frozenset[str]:
    aliases = {field.name}
    nodes = [field.default]
    if isinstance(field.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nodes.extend(field.node.decorator_list)
    elif (
        isinstance(field.annotation, ast.Subscript) and _name(field.annotation.value) == "Annotated"
    ) and isinstance(field.annotation.slice, ast.Tuple):
        nodes.extend(field.annotation.slice.elts[1:])
    for node in nodes:
        if isinstance(node, ast.Call):
            aliases.update(
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg in {"alias", "serialization_alias", "validation_alias"}
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            )
    return frozenset(aliases)


class PythonReferences:
    """A single basedpyright workspace; await analysis before querying symbols."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(ROOT / ".venv/bin/basedpyright-langserver"), "--stdio"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.messages: Queue[dict[str, Any]] = Queue()
        self.sequence = 0
        self.finished_analysis: bool = False
        self.opened: set[Path] = set()
        self.type_cache: dict[tuple[Path, int, int], Sequence[dict[str, Any]]] = {}
        Thread(target=self._read, daemon=True).start()
        try:
            self.request(
                "initialize",
                {
                    "processId": None,
                    "rootPath": str(ROOT),
                    "rootUri": ROOT.as_uri(),
                    "workspaceFolders": [{"uri": ROOT.as_uri(), "name": "grassmoth"}],
                    "capabilities": {
                        "window": {"workDoneProgress": True},
                        "workspace": {"configuration": True},
                    },
                },
            )
            self.send({"method": "initialized", "params": {}})
            self._wait_for_analysis(time.monotonic() + 180)
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=10)
            raise

    def _wait_for_analysis(self, deadline: float) -> None:
        while not self.finished_analysis:
            self._receive(deadline)

    def _read(self) -> None:
        assert self.process.stdout is not None
        while True:
            headers: dict[str, str] = {}
            while line := self.process.stdout.readline():
                if line == b"\r\n":
                    break
                key, value = line.decode().split(":", 1)
                headers[key.lower()] = value.strip()
            if not line:
                self.messages.put({"error": {"message": "basedpyright language server exited"}})
                return
            body = self.process.stdout.read(int(headers["content-length"]))
            self.messages.put(json.loads(body))

    def send(self, message: Mapping[str, Any]) -> None:
        assert self.process.stdin is not None
        payload = json.dumps({"jsonrpc": "2.0", **message}).encode()
        self.process.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        self.process.stdin.flush()

    def _receive(self, deadline: float) -> dict[str, Any]:
        try:
            message = self.messages.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty as exc:
            raise RuntimeError("basedpyright language server timed out") from exc
        method = message.get("method")
        if method == "workspace/configuration":
            self.send(
                {
                    "id": message["id"],
                    "result": [
                        {"analysis": {"diagnosticMode": "workspace"}}
                        if item.get("section") == "basedpyright"
                        else None
                        for item in message["params"]["items"]
                    ],
                }
            )
        elif "id" in message and method:
            self.send({"id": message["id"], "result": None})
        if method == "pyright/endProgress" or (
            method == "$/progress" and message["params"]["value"].get("kind") == "end"
        ):
            self.finished_analysis = True
        if "error" in message:
            raise RuntimeError(str(message["error"]))
        return message

    def request(self, method: str, params: Mapping[str, Any]) -> Any:
        self.sequence += 1
        identity = self.sequence
        self.send({"id": identity, "method": method, "params": params})
        deadline = time.monotonic() + 180
        while True:
            message = self._receive(deadline)
            if message.get("id") == identity and "method" not in message:
                return message.get("result")

    def _open(self, source: SourceFile) -> None:
        if source.path not in self.opened:
            self.send(
                {
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": source.path.as_uri(),
                            "languageId": "python",
                            "version": 1,
                            "text": source.text,
                        }
                    },
                }
            )
            self.opened.add(source.path)

    def references(self, source: SourceFile, line: int, column: int) -> Sequence[dict[str, Any]]:
        self._open(source)
        return (
            self.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": source.path.as_uri()},
                    "position": {"line": line, "character": column},
                    "context": {"includeDeclaration": False},
                },
            )
            or ()
        )

    def hover(self, source: SourceFile, node: ast.AST) -> str:
        self._open(source)
        line, column = _lsp_position(source, node)
        result = self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": source.path.as_uri()},
                "position": {"line": line, "character": column},
            },
        )
        if result is None:
            return ""
        contents = result["contents"]
        return contents if isinstance(contents, str) else contents.get("value", "")

    def type_definitions(self, source: SourceFile, node: ast.AST) -> Sequence[dict[str, Any]]:
        self._open(source)
        line, column = _lsp_position(source, node)
        key = source.path, line, column
        if key not in self.type_cache:
            self.type_cache[key] = (
                self.request(
                    "textDocument/typeDefinition",
                    {
                        "textDocument": {"uri": source.path.as_uri()},
                        "position": {"line": line, "character": column},
                    },
                )
                or ()
            )
        return self.type_cache[key]

    def definitions(self, source: SourceFile, node: ast.AST) -> Sequence[dict[str, Any]]:
        self._open(source)
        line, column = _lsp_position(source, node)
        return (
            self.request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": source.path.as_uri()},
                    "position": {"line": line, "character": column},
                },
            )
            or ()
        )

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request("shutdown", {})
                self.send({"method": "exit", "params": {}})
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)


def _at_position(source: SourceFile, line: int, column: int) -> ast.AST | None:
    text = source.text.splitlines()[line]
    column = len(text.encode("utf-16-le")[: column * 2].decode("utf-16-le").encode("utf-8"))
    matches: list[tuple[tuple[int, int], ast.AST]] = []
    for node in ast.walk(source.tree):
        if not isinstance(node, (ast.expr, ast.stmt, ast.arg, ast.keyword, ast.alias, ast.pattern)):
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            continue
        if (
            (node.lineno, node.col_offset)
            <= (line + 1, column)
            < (node.end_lineno, node.end_col_offset)
        ):
            matches.append(
                ((node.end_lineno - node.lineno, node.end_col_offset - node.col_offset), node)
            )
    return min(matches, key=lambda match: match[0])[1] if matches else None


def classify_reference(field: Field, source: SourceFile, node: ast.AST) -> str:
    ancestors = tuple(_ancestors(source, node))
    if isinstance(node, ast.keyword) or any(
        isinstance(parent, ast.keyword) and parent.value is not node for parent in ancestors[:1]
    ):
        return "write"
    if source.path == field.source.path and field.owner in ancestors:
        if _validated(source, field.owner, node):
            return "validation"
        if field.node in ancestors or node is field.node:
            return "declaration"
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return "read"
    if isinstance(node, ast.Attribute):
        return "write"
    if isinstance(node, ast.MatchClass):
        return "read"
    if isinstance(node, ast.Constant) and any(
        isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Load)
        for parent in ancestors[:2]
    ):
        return "read"
    if isinstance(node, ast.Constant) and ancestors:
        parent = ancestors[0]
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr == "get"
            and parent.args
            and parent.args[0] is node
        ):
            return "read"
    return "declaration"


def _constant(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Call) and _name(node.func) in {"Field", "field"}:
        if node.args:
            return _constant(node.args[0])
        for keyword in node.keywords:
            if keyword.arg == "default":
                return _constant(keyword.value)
            if keyword.arg == "default_factory" and _name(keyword.value) in {"tuple", "frozenset"}:
                return "()" if _name(keyword.value) == "tuple" else "frozenset()"
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    pending = [value]
    while pending:
        member = pending.pop()
        if isinstance(member, tuple):
            pending.extend(member)
        elif isinstance(member, list | dict | set):
            # Equal initial contents do not prove a mutable field is constant.
            return None
    return repr(value)


def constructor_constants(
    field: Field,
    producers: Sequence[tuple[SourceFile, ast.Call]],
    server: PythonReferences,
) -> tuple[str, tuple[str, ...]] | None:
    values: list[str] = []
    sites: list[str] = []
    for source, node in producers:
        if any(keyword.arg is None for keyword in node.keywords):
            return None
        keyword = next(
            (keyword for keyword in node.keywords if keyword.arg in _field_aliases(field)),
            None,
        )
        value = keyword.value if keyword is not None else field.default
        supplied = keyword is not None
        if not supplied and node.args:
            # Positional dataclass / NamedTuple constructors are producers too.
            # A starred tuple or TypedDict mapping has unknown field values.
            positional = [
                member.target.id
                for member in field.owner.body
                if isinstance(member, ast.AnnAssign)
                and isinstance(member.target, ast.Name)
                and _name(member.annotation) != "ClassVar"
            ]
            if any(isinstance(argument, ast.Starred) for argument in node.args):
                return None
            if any(_name(base) == "TypedDict" for base in field.owner.bases):
                return None
            if _resolve(source, node.func) != field.identity.rpartition(".")[0]:
                # Inherited and aliased positional constructors need their actual
                # signature, rather than the field declaration's local order.
                return None
            index = positional.index(field.name)
            if index < len(node.args):
                value = node.args[index]
                supplied = True
        constant = _constant(value)
        if constant is None and supplied and isinstance(value, ast.Name | ast.Attribute):
            hover = server.hover(source, value)
            match = re.search(r"(?m)^(?:\([^\n)]+\)\s+)?\w+: Literal\[([^\]\n]+)\]\s*$", hover)
            if match:
                annotation = ast.parse(f"Literal[{match.group(1)}]", mode="eval").body
                literal = _literal(annotation)
                if literal is not None:
                    constant = _constant(ast.parse(literal, mode="eval").body)
        if constant is None:
            return None
        values.append(constant)
        sites.append(f"{source.path.relative_to(ROOT)}:{node.lineno}")
    if values and len(set(values)) == 1:
        return values[0], tuple(sites)
    return None


def _constructor_owner(
    source: SourceFile,
    node: ast.AST,
    aliases: Mapping[str, tuple[SourceFile, ast.expr]],
    seen: frozenset[str] = frozenset(),
) -> str:
    identity = _resolve(source, node)
    if identity in aliases and identity not in seen:
        origin, expression = aliases[identity]
        return _constructor_owner(origin, expression, aliases, seen | {identity})
    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript):
        return identity
    return ""


def _receiver_owners(
    source: SourceFile,
    receiver: ast.expr,
    sources: Mapping[Path, SourceFile],
    server: PythonReferences,
) -> frozenset[str]:
    result: set[str] = set()
    aliases = _type_aliases(tuple(sources.values()))
    for definition in server.type_definitions(source, receiver):
        path = Path(unquote(urlparse(definition["uri"]).path))
        origin = sources.get(path)
        if origin is None:
            continue
        position = definition["range"]["start"]
        node = _at_position(origin, position["line"], position["character"])
        if node is None:
            continue
        owner = (
            node
            if isinstance(node, ast.ClassDef)
            else next(
                (
                    ancestor
                    for ancestor in _ancestors(origin, node)
                    if isinstance(ancestor, ast.ClassDef)
                ),
                None,
            )
        )
        if owner is not None:
            result.add(f"{origin.module}.{owner.name}")
    # LSP typeDefinition(list[Record]) points to builtins.list. The declaration's
    # annotation retains the owning identity of Record, including imported aliases.
    for definition in server.definitions(source, receiver):
        path = Path(unquote(urlparse(definition["uri"]).path))
        origin = sources.get(path)
        if origin is None:
            continue
        position = definition["range"]["start"]
        node = _at_position(origin, position["line"], position["character"])
        if isinstance(node, ast.arg) and node.annotation is not None:
            result.update(_type_owners(origin, node.annotation, aliases))
        parent = origin.parents.get(node) if node is not None else None
        if isinstance(parent, ast.AnnAssign):
            result.update(_type_owners(origin, parent.annotation, aliases))
    return frozenset(result)


def producer_inventory(
    fields: Sequence[Field], sources: Sequence[SourceFile], server: PythonReferences
) -> tuple[dict[str, list[tuple[SourceFile, ast.Call]]], frozenset[str]]:
    """Declared inputs and revisions prevent a closed-producer constant proof."""
    by_owner = _fields_by_owner(fields, sources)
    by_path = {source.path: source for source in sources}
    aliases = _type_aliases(sources)
    producers: dict[str, list[tuple[SourceFile, ast.Call]]] = {}
    open_fields: set[str] = set()
    parsed: set[str] = set()
    for source in sources:
        for node in ast.walk(source.tree):
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.Return) and node.value is not None:
                declared_alias = (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and f"{source.module}.{node.targets[0].id}" in aliases
                ) or (isinstance(node, ast.AnnAssign) and _name(node.annotation) == "TypeAlias")
                if not declared_alias:
                    for value in ast.walk(node.value):
                        parent = source.parents.get(value)
                        if isinstance(value, ast.Name | ast.Attribute) and not (
                            isinstance(parent, ast.Call) and parent.func is value
                        ):
                            owner = _constructor_owner(source, value, aliases)
                            if owner in by_owner:
                                parsed.add(owner)
            if not isinstance(node, ast.Call):
                continue
            identity = _constructor_owner(source, node.func, aliases)
            if isinstance(node.func, ast.Name) and node.func.id == "cls":
                owner = next(
                    (
                        ancestor
                        for ancestor in _ancestors(source, node)
                        if isinstance(ancestor, ast.ClassDef)
                    ),
                    None,
                )
                if owner is not None:
                    identity = f"{source.module}.{owner.name}"
            if identity in by_owner:
                for field in by_owner[identity].values():
                    producers.setdefault(field.identity, []).append((source, node))
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "model_validate",
                "model_validate_json",
                "model_construct",
                "parse_obj",
                "from_orm",
            }:
                parsed.update(_type_owners(source, node.func.value, aliases))
            if _name(node.func) == "TypeAdapter" and node.args:
                parsed.update(_type_owners(source, node.args[0], aliases))
            if _name(node.func) not in {
                "isinstance",
                "issubclass",
                "cast",
                "assert_type",
                "reveal_type",
            }:
                # Passing a constructor to another function, including a
                # default_factory, creates producers whose values are unknown.
                for value in (*node.args, *(keyword.value for keyword in node.keywords)):
                    owner = _constructor_owner(source, value, aliases)
                    if owner in by_owner:
                        parsed.add(owner)
            receiver = None
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "revised",
                "_replace",
                "model_copy",
                "update",
                "setdefault",
            }:
                receiver = node.func.value
            elif (_resolve(source, node.func) == "dataclasses.replace" and node.args) or (
                _name(node.func) in {"setattr", "__setattr__"} and node.args
            ):
                receiver = node.args[0]
            if receiver is not None:
                for owner in _receiver_owners(source, receiver, by_path, server):
                    for name, field in by_owner.get(owner, {}).items():
                        if (
                            any(keyword.arg in {name, None, "update"} for keyword in node.keywords)
                            or _name(node.func) in {"update", "setdefault"}
                            or (
                                _name(node.func) in {"setattr", "__setattr__"}
                                and len(node.args) > 1
                                and (
                                    not isinstance(node.args[1], ast.Constant)
                                    or node.args[1].value == name
                                )
                            )
                        ):
                            open_fields.add(field.identity)
    # TypeAdapter(PipelineConfig), for example, also constructs its nested sampler
    # options. Their default is not evidence of a production constant.
    pending = list(parsed)
    visited: set[str] = set()
    while pending:
        owner = pending.pop()
        if owner in visited:
            continue
        visited.add(owner)
        for field in by_owner.get(owner, {}).values():
            open_fields.add(field.identity)
            if field.annotation is not None:
                pending.extend(_type_owners(field.source, field.annotation, aliases))
    return producers, frozenset(open_fields)


def _opaque_reads(
    fields: Sequence[Field], sources: Sequence[SourceFile], server: PythonReferences
) -> frozenset[str]:
    by_owner = _fields_by_owner(fields, sources)
    by_path = {source.path: source for source in sources}
    opaque: set[str] = set()
    for source in sources:
        for node in ast.walk(source.tree):
            if not isinstance(node, ast.Call):
                continue
            receiver = None
            member = None
            if _name(node.func) == "getattr" and len(node.args) >= 2:
                receiver = node.args[0]
                member = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "model_dump",
                "model_dump_json",
            }:
                receiver = node.func.value
            elif (
                _resolve(source, node.func)
                in {
                    "polars.DataFrame",
                    "polars.from_dicts",
                    "pandas.DataFrame",
                    "dataclasses.asdict",
                }
                and node.args
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"dump_python", "dump_json"}
                and node.args
            ):
                receiver = node.args[0]
            if receiver is None:
                continue
            if isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}:
                owner = next(
                    (
                        ancestor
                        for ancestor in _ancestors(source, node)
                        if isinstance(ancestor, ast.ClassDef)
                    ),
                    None,
                )
                owners = {f"{source.module}.{owner.name}"} if owner is not None else set()
            else:
                owners = _receiver_owners(source, receiver, by_path, server)
            for owner in owners:
                for field in by_owner.get(owner, {}).values():
                    if source.path == field.source.path and _validated(source, field.owner, node):
                        continue
                    if member is None or member == field.name:
                        opaque.add(field.identity)
    return frozenset(opaque)


def web_readers(
    fields: Sequence[Field], exported: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    payload = [
        {
            "identity": field.identity,
            "name": field.name,
            "owner": field.owner.name,
            "aliases": sorted(exported[field.identity]["aliases"]),
            "components": sorted(exported[field.identity]["components"]),
        }
        for field in fields
        if field.identity in exported
    ]
    try:
        result = subprocess.run(
            ["bun", str(WEB_READERS)],
            cwd=ROOT,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"TypeScript field analysis failed:\n{exc.stderr}") from exc
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr, flush=True)
    return json.loads(result.stdout)


def _suppression(finding: Finding) -> str | None:
    line = finding.field.source.text.splitlines()[finding.field.position[0]]
    match = _SUPPRESSION.search(line)
    if match and finding.code in {code.strip() for code in match.group(1).split(",")}:
        return match.group(2).strip()
    return None


def check(selected: frozenset[str]) -> CheckResult:
    sources = tuple(
        _source_file(path)
        for path in sorted(path for root in SOURCE_ROOTS for path in root.rglob("*.py"))
    )
    by_path = {source.path: source for source in sources}
    fields = collect_fields(sources)
    print(f"fields: collected {len(fields)} declarations", file=sys.stderr, flush=True)
    api_text = OPENAPI.read_text()
    api = json.loads(api_text)
    exported = exported_fields(fields, api, sources)
    tags = discriminators(fields, api["components"]["schemas"], sources)
    findings = list(recomputed_fields(fields)) if "FIELD001" in selected else []
    web = web_readers(fields, exported) if "FIELD003" in selected else {"readers": {}, "opaque": {}}
    print(
        f"fields: {len(exported)} exported; {len(web['readers'])} web readers",
        file=sys.stderr,
        flush=True,
    )
    server = PythonReferences() if selected & {"FIELD002", "FIELD003"} else None
    outcomes: dict[str, FieldOutcome] = {}
    aliases = _type_aliases(sources)
    try:
        producers, open_producers = (
            producer_inventory(fields, sources, server)
            if server is not None and "FIELD002" in selected
            else ({}, frozenset())
        )
        opaque = (
            _opaque_reads(fields, sources, server)
            if server is not None and "FIELD003" in selected
            else frozenset()
        )
        reference_cache: dict[str, tuple[tuple[SourceFile, ast.AST], ...]] = {}

        def locations(field: Field) -> tuple[tuple[SourceFile, ast.AST], ...]:
            if field.identity not in reference_cache:
                assert server is not None
                resolved = []
                for reference in server.references(field.source, *field.position):
                    path = Path(unquote(urlparse(reference["uri"]).path))
                    source = by_path.get(path)
                    if source is None:
                        continue
                    start = reference["range"]["start"]
                    node = _at_position(source, start["line"], start["character"])
                    if node is not None:
                        resolved.append((source, node))
                reference_cache[field.identity] = tuple(resolved)
            return reference_cache[field.identity]

        for index, field in enumerate(fields):
            if index % 100 == 0:
                print(f"fields: {index}/{len(fields)}", file=sys.stderr, flush=True)
            if "FIELD002" in selected and not field.computed and field.identity not in tags:
                literal = _literal(field.annotation, field.source, aliases)
                if literal is not None:
                    findings.append(
                        Finding(
                            field,
                            "FIELD002",
                            f"delete constant field with single-valued Literal[{literal}]; it discriminates no union",
                        )
                    )
                else:
                    assert server is not None
                    constant = constructor_constants(
                        field, producers.get(field.identity, ()), server
                    )
                    if constant is not None and field.identity in open_producers:
                        constant = None
                    if constant is not None and any(
                        (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store))
                        or (
                            isinstance(node, ast.Constant)
                            and any(
                                isinstance(parent, ast.Subscript)
                                and isinstance(parent.ctx, ast.Store)
                                for parent in _ancestors(source, node)
                            )
                        )
                        for source, node in locations(field)
                    ):
                        constant = None
                    if constant is not None:
                        value, sites = constant
                        findings.append(
                            Finding(
                                field,
                                "FIELD002",
                                f"every production constructor supplies {value}; producers: {', '.join(sites)}",
                            )
                        )
            if "FIELD003" not in selected or field.identity not in exported:
                continue
            if field.identity in tags:
                outcomes[field.identity] = FieldOutcome(field, "consumed", "union discriminator")
                continue
            if field.identity in web["readers"]:
                outcomes[field.identity] = FieldOutcome(
                    field, "consumed", f"web reader {web['readers'][field.identity]}"
                )
                continue
            assert server is not None
            consumed = False
            for source, node in locations(field):
                if classify_reference(field, source, node) == "read":
                    consumed = True
                    outcomes[field.identity] = FieldOutcome(
                        field,
                        "consumed",
                        f"Python reader {source.path.relative_to(ROOT)}:{_lsp_position(source, node)[0] + 1}",
                    )
                    break
            if consumed:
                continue
            if field.identity in web["opaque"] or field.identity in opaque:
                evidence = (
                    f"web access erases properties at {web['opaque'][field.identity]}"
                    if field.identity in web["opaque"]
                    else "typed reflective access or generic serialization in Python"
                )
                outcomes[field.identity] = FieldOutcome(field, "opaque", evidence)
                continue
            evidence = "no reader in Python src or knip's production web project"
            findings.append(Finding(field, "FIELD003", evidence))
            outcomes[field.identity] = FieldOutcome(field, "error", evidence)
    finally:
        if server is not None:
            server.close()
    unique = sorted(
        {(finding.field.identity, finding.code): finding for finding in findings}.values(),
        key=lambda finding: (finding.field.source.path, finding.field.position, finding.code),
    )
    suppressed = tuple(
        (finding, reason) for finding in unique if (reason := _suppression(finding)) is not None
    )
    active = tuple(finding for finding in unique if _suppression(finding) is None)
    for finding in active:
        outcomes[finding.field.identity] = FieldOutcome(finding.field, "error", finding.evidence)
    for finding, reason in suppressed:
        outcomes[finding.field.identity] = FieldOutcome(
            finding.field, "opaque", f"inline exception: {reason}"
        )
    for field in fields:
        outcomes.setdefault(
            field.identity,
            FieldOutcome(
                field,
                "opaque",
                "no closed redundancy proof; internal readers are checked by Vulture",
            ),
        )
    if (
        tuple(sorted(path for root in SOURCE_ROOTS for path in root.rglob("*.py")))
        != tuple(source.path for source in sources)
        or any(source.path.read_text() != source.text for source in sources)
        or OPENAPI.read_text() != api_text
    ):
        raise RuntimeError(
            "source files or OpenAPI changed during field analysis; rerun the checker"
        )
    return CheckResult(active, suppressed, tuple(outcomes.values()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="append", choices=RULES)
    parser.add_argument(
        "--explain", action="store_true", help="print each field's consumed/error/opaque outcome"
    )
    args = parser.parse_args(argv)
    try:
        result = check(frozenset(args.select or RULES))
    except (RuntimeError, OSError, SyntaxError, json.JSONDecodeError) as exc:
        print(f"Field checks failed: {exc}", file=sys.stderr)
        return 2
    findings, suppressed = result.findings, result.suppressed
    for finding in findings:
        print(finding.diagnostic())
    for finding, reason in suppressed:
        print(f"suppressed: {finding.diagnostic()} -- {reason}")
    if args.explain:
        for outcome in sorted(result.outcomes, key=lambda outcome: outcome.field.identity):
            print(f"{outcome.status}: {outcome.field.identity} -- {outcome.evidence}")
    counts = {
        status: sum(outcome.status == status for outcome in result.outcomes)
        for status in ("consumed", "error", "opaque")
    }
    print(
        f"Field outcomes: {counts['consumed']} consumed, {counts['error']} errors, {counts['opaque']} opaque"
    )
    print(f"Field checks: {len(findings)} finding(s), {len(suppressed)} inline suppression(s)")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
