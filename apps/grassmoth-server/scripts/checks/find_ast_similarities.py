"""Compare Python function bodies for shared structure and binding differences.

Run explicitly, like ``complexity``; this command is not part of ``lint``::

    bun run similarity
    bun run similarity packages/grassmoth/src/grassmoth/models/ssm
    bun run similarity --threshold 0.8 --limit 10
    bun run similarity --json
    bun run similarity --workers 1

Existing duplicate-audit candidates and a bounded AST-pattern neighbor search
seed ordered tree comparisons. Local bindings are normalized consistently;
captures, shadowing, comprehension scopes, literals and nested implementations
remain visible. Statement sequences are aligned before anti-unification, which
records consistent substitutions as shared holes. Ranking favors substantial
shared bodies. Scores measure retained syntax, never semantic confidence.
Only bodies determine the score; signatures are reported separately. Attribute
and keyword names remain literal. Candidate retrieval is bounded, not exhaustive.
Source parsing and candidate retrieval use CPU worker processes by default; ``--workers``
controls their count. Binding-aware trees are built only for retrieved candidates.
Detailed comparisons and report ordering stay deterministic.
Text reports bound difference groups and snippet length per match; JSON retains
every evidence location.

Findings are advisory and exit successfully. Source/argument failures exit with
an error. This script imports no application modules, installs no dependencies,
and keeps no baseline or suppression registry. Python source is the only input;
dynamic dispatch, alias equivalence and runtime contracts are not resolved.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import chain
from multiprocessing import cpu_count, get_context
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, override

from scripts.checks.find_duplicates import (
    DEFAULT_MIN_FUNCTION_NODES,
    REPO_ROOT,
    DuplicateAuditError,
    LineRange,
    PythonDefinition,
    SourceSelection,
    _function_body,
    ast_candidates,
    collect_python_definitions,
    explicit_selection,
    source_paths,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
type ScopeNode = FunctionNode | ast.Lambda | ast.ClassDef
type Pattern = tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]
type PatternCounts = tuple[tuple[Pattern, int], ...]

DEFAULT_THRESHOLD = 0.55
DEFAULT_RETRIEVAL_THRESHOLD = 0.995
DEFAULT_NEIGHBORS = 5
DEFAULT_LIMIT = 50
_ALIGNED_SEQUENCES = frozenset({"sequence:body", "sequence:orelse", "sequence:finalbody"})


@dataclass(frozen=True, slots=True)
class BindingScope:
    kind: Literal["function", "class", "comprehension"]
    bindings: Mapping[str, str]
    globals: frozenset[str]


class _BindingCollector(ast.NodeVisitor):
    """Collect one scope's declarations without leaking nested bindings."""

    def __init__(self) -> None:
        self.names: dict[str, int] = {}
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def bind(self, name: str) -> None:
        self.names.setdefault(name, len(self.names))

    @override
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id)

    @override
    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    @override
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def _definition(self, node: FunctionNode) -> None:
        self.bind(node.name)
        for expression in (*node.decorator_list, *node.args.defaults):
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                self.visit(expression)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._definition(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._definition(node)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in node.args.defaults:
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                self.visit(expression)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bind(node.name)
        for expression in (*node.bases, *node.decorator_list):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bind(alias.asname or alias.name.split(".", 1)[0])

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bind(alias.asname or alias.name)

    @override
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.bind(node.name)
        self.generic_visit(node)

    @override
    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            self.bind(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            self.bind(node.rest)
        super().generic_visit(node)

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        # Comprehension targets have their own scope. Assignment expressions in
        # its expressions bind the surrounding scope (PEP 572).
        for generator in node.generators:
            self.visit(generator.iter)
            for expression in generator.ifs:
                self.visit(expression)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node)

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node)

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node)


def _scope(node: ScopeNode) -> BindingScope:
    collector = _BindingCollector()
    parameters: list[ast.arg] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        parameters = [*args.posonlyargs, *args.args]
        if args.vararg is not None:
            parameters.append(args.vararg)
        parameters.extend(args.kwonlyargs)
        if args.kwarg is not None:
            parameters.append(args.kwarg)
        for parameter in parameters:
            collector.bind(parameter.arg)
    if isinstance(node, ast.Lambda):
        collector.visit(node.body)
    else:
        for statement in node.body:
            collector.visit(statement)
    parameter_names = {
        parameter.arg: f"parameter:{slot}" for slot, parameter in enumerate(parameters)
    }
    local_names = tuple(
        name
        for name in collector.names
        if name not in parameter_names and name not in collector.globals | collector.nonlocals
    )
    names = {**parameter_names, **{name: f"local:{slot}" for slot, name in enumerate(local_names)}}
    return BindingScope(
        "class" if isinstance(node, ast.ClassDef) else "function",
        MappingProxyType(names),
        frozenset(collector.globals),
    )


@dataclass(frozen=True, slots=True)
class AstTerm:
    label: str
    children: tuple[AstTerm, ...]
    size: int
    fingerprint: str
    text: str = field(compare=False)
    line: int = field(compare=False)


def _term(
    label: str,
    children: tuple[AstTerm, ...] = (),
    *,
    text: str = "",
    line: int = 0,
    weight: int = 1,
) -> AstTerm:
    fingerprint = hashlib.sha256(
        repr((label, tuple(c.fingerprint for c in children))).encode()
    ).hexdigest()
    return AstTerm(
        label, children, weight + sum(c.size for c in children), fingerprint, text or label, line
    )


class _Normalizer:
    """Normalize lexical identities while preserving operations and literals."""

    def __init__(self, outer: tuple[BindingScope, ...]) -> None:
        self.root_depth = len(outer)

    def _name(self, name: str, scopes: tuple[BindingScope, ...], line: int) -> AstTerm:
        for index in range(len(scopes) - 1, -1, -1):
            scope = scopes[index]
            if name in scope.globals:
                break
            if name in scope.bindings:
                if scope.kind == "class":
                    return _term(f"class-name:{name}", text=name, line=line)
                identity = f"binding:{index - self.root_depth}:{scope.bindings[name]}"
                return _term(identity, text=f"{name} [{identity}]", line=line)
        return _term(f"global:{name}", text=f"{name} [global/free]", line=line)

    def _field(self, name: str, child: AstTerm) -> AstTerm:
        return _term(f"field:{name}", (child,), weight=0, line=child.line)

    def _arguments(
        self, args: ast.arguments, outer: tuple[BindingScope, ...], inner: tuple[BindingScope, ...]
    ) -> AstTerm:
        def parameter(arg: ast.arg | None) -> AstTerm:
            if arg is None:
                return self.normalize(None, inner)
            return _term(
                "arg",
                (self._name(arg.arg, inner, arg.lineno), self.normalize(arg.annotation, outer)),
                line=arg.lineno,
            )

        fields = []
        for name, value in ast.iter_fields(args):
            if name in {"defaults", "kw_defaults"}:
                normalized = self.normalize(value, outer, field_name=name)
            elif isinstance(value, list):
                normalized = _term(
                    f"sequence:{name}", tuple(parameter(arg) for arg in value), weight=0
                )
            else:
                normalized = parameter(value)
            fields.append(self._field(name, normalized))
        return _term("arguments", tuple(fields))

    def _comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        scopes: tuple[BindingScope, ...],
    ) -> AstTerm:
        collector = _BindingCollector()
        for generator in node.generators:
            collector.visit(generator.target)
        inner = (
            *(s for s in scopes if s.kind != "class"),
            BindingScope(
                "comprehension",
                MappingProxyType({name: f"local:{slot}" for name, slot in collector.names.items()}),
                frozenset(),
            ),
        )
        generators = []
        for index, generator in enumerate(node.generators):
            fields = (
                self._field("target", self.normalize(generator.target, inner)),
                self._field(
                    "iter", self.normalize(generator.iter, scopes if index == 0 else inner)
                ),
                self._field("ifs", self.normalize(generator.ifs, inner, field_name="ifs")),
                self._field("is_async", self.normalize(generator.is_async, inner)),
            )
            generators.append(_term("comprehension", fields))
        values = (
            (("key", node.key), ("value", node.value))
            if isinstance(node, ast.DictComp)
            else (("elt", node.elt),)
        )
        fields = [self._field(name, self.normalize(value, inner)) for name, value in values]
        fields.append(
            self._field("generators", _term("sequence:generators", tuple(generators), weight=0))
        )
        return _term(type(node).__name__, tuple(fields), line=node.lineno)

    def normalize(
        self,
        value: object,
        scopes: tuple[BindingScope, ...],
        *,
        field_name: str = "",
        line: int = 0,
    ) -> AstTerm:
        if isinstance(value, ast.Name):
            return _term(
                "Name",
                (self._name(value.id, scopes, value.lineno), _term(type(value.ctx).__name__)),
                line=value.lineno,
            )
        if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            outer = tuple(s for s in scopes if s.kind != "class")
            inner = (*outer, _scope(value))
            fields = []
            for name, child in ast.iter_fields(value):
                if name == "type_comment":
                    continue
                if name == "name" and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    normalized = self._name(value.name, scopes, value.lineno)
                elif name == "args":
                    normalized = self._arguments(value.args, scopes, inner)
                elif name == "body":
                    body = (
                        value.body if isinstance(value, ast.Lambda) else list(_function_body(value))
                    )
                    normalized = self.normalize(body, inner, field_name="body")
                else:
                    normalized = self.normalize(child, scopes, field_name=name)
                fields.append(self._field(name, normalized))
            return _term(type(value).__name__, tuple(fields), line=value.lineno)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return self._comprehension(value, scopes)
        if isinstance(value, ast.NamedExpr):
            # Skip only the comprehension scopes between the assignment and
            # its owner. A lambda inside a comprehension owns its own walrus
            # bindings and must retain the same identity at stores and loads.
            scope_end = len(scopes)
            while scope_end and scopes[scope_end - 1].kind == "comprehension":
                scope_end -= 1
            surrounding = scopes[:scope_end]
            return _term(
                "NamedExpr",
                (
                    self._field("target", self.normalize(value.target, surrounding)),
                    self._field("value", self.normalize(value.value, scopes)),
                ),
                line=value.lineno,
            )
        if isinstance(value, ast.ClassDef):
            inner = (*scopes, _scope(value))
            fields = tuple(
                self._field(
                    name,
                    self._name(value.name, scopes, value.lineno)
                    if name == "name"
                    else self.normalize(
                        child, inner if name == "body" else scopes, field_name=name
                    ),
                )
                for name, child in ast.iter_fields(value)
            )
            return _term("ClassDef", fields, line=value.lineno)
        if isinstance(value, ast.alias):
            # The imported symbol/module is an external contract; only its
            # local alias can be consistently renamed.
            return _term(
                "alias",
                (
                    self._field("name", self.normalize(value.name, scopes)),
                    self._field(
                        "binding",
                        self._name(
                            value.asname or value.name.split(".", 1)[0], scopes, value.lineno
                        ),
                    ),
                ),
                line=value.lineno,
            )
        if isinstance(value, (ast.Global, ast.Nonlocal)):
            return _term(
                type(value).__name__,
                tuple(self._name(name, scopes, value.lineno) for name in value.names),
                line=value.lineno,
            )
        if isinstance(value, ast.AST):
            node_line = getattr(value, "lineno", line)
            fields = []
            for name, child in ast.iter_fields(value):
                if name == "type_comment":
                    continue
                binder = (
                    isinstance(value, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar))
                    and name == "name"
                ) or (isinstance(value, ast.MatchMapping) and name == "rest")
                normalized = (
                    self._name(child, scopes, node_line)
                    if binder and isinstance(child, str)
                    else self.normalize(child, scopes, field_name=name, line=node_line)
                )
                fields.append(self._field(name, normalized))
            return _term(type(value).__name__, tuple(fields), line=node_line)
        if isinstance(value, list):
            return _term(
                f"sequence:{field_name}",
                tuple(self.normalize(child, scopes, line=line) for child in value),
                weight=0,
                line=line,
            )
        return _term(f"{type(value).__name__}:{value!r}", text=repr(value), line=line)


def normalize_function(node: FunctionNode, outer: tuple[BindingScope, ...] = ()) -> AstTerm:
    normalizer = _Normalizer(outer)
    scopes = (*outer, _scope(node))
    body = normalizer.normalize(list(_function_body(node)), scopes, field_name="body")
    return _term(type(node).__name__, (body,), weight=0, line=node.lineno)


@dataclass(frozen=True, slots=True)
class SyntaxDifference:
    kind: str
    path: str
    hole: int
    left: str
    right: str
    left_line: int
    right_line: int


@dataclass(frozen=True, slots=True)
class AstComparison:
    score: float
    shared_size: float
    differences: tuple[SyntaxDifference, ...]


def _describe(term: AstTerm) -> str:
    def describe(child: AstTerm, depth: int) -> str:
        if not child.children:
            return child.text
        if child.label.startswith("field:"):
            return f"{child.label.removeprefix('field:')}={describe(child.children[0], depth)}"
        if depth == 0:
            return child.label
        parts = [describe(part, depth - 1) for part in child.children[:4]]
        if len(child.children) > 4:
            parts.append("...")
        return f"{child.label}({', '.join(parts)})"

    return describe(term, 4)[:160]


def compare_terms(first: AstTerm, second: AstTerm) -> AstComparison:
    """Ordered anti-unification with alignment at statement-list boundaries."""
    costs: dict[tuple[int, int], int] = {}
    alignments: dict[tuple[int, int], tuple[tuple[int | None, int | None], ...]] = {}

    def cost(left: AstTerm, right: AstTerm) -> int:
        identity = (id(left), id(right))
        if identity in costs:
            return costs[identity]
        if left.fingerprint == right.fingerprint:
            value = 0
        elif left.label != right.label:
            value = left.size + right.size
        elif left.label in _ALIGNED_SEQUENCES:
            value = align(left, right)
        elif len(left.children) != len(right.children):
            value = left.size + right.size
        else:
            value = sum(cost(a, b) for a, b in zip(left.children, right.children, strict=True))
        costs[identity] = value
        return value

    def align(left: AstTerm, right: AstTerm) -> int:
        rows = [[0] * (len(right.children) + 1) for _ in range(len(left.children) + 1)]
        for i, child in enumerate(left.children, 1):
            rows[i][0] = rows[i - 1][0] + child.size
        for j, child in enumerate(right.children, 1):
            rows[0][j] = rows[0][j - 1] + child.size
        for i, a in enumerate(left.children, 1):
            for j, b in enumerate(right.children, 1):
                rows[i][j] = min(
                    rows[i - 1][j - 1] + cost(a, b),
                    rows[i - 1][j] + a.size,
                    rows[i][j - 1] + b.size,
                )
        steps: list[tuple[int | None, int | None]] = []
        i, j = len(left.children), len(right.children)
        while i or j:
            if (
                i
                and j
                and rows[i][j]
                == rows[i - 1][j - 1] + cost(left.children[i - 1], right.children[j - 1])
            ):
                steps.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i and rows[i][j] == rows[i - 1][j] + left.children[i - 1].size:
                steps.append((i - 1, None))
                i -= 1
            else:
                steps.append((None, j - 1))
                j -= 1
        alignments[(id(left), id(right))] = tuple(reversed(steps))
        return rows[-1][-1]

    differences: list[SyntaxDifference] = []
    holes: dict[tuple[str, str], int] = {}

    def emit(left: AstTerm, right: AstTerm, path: str) -> None:
        if left.fingerprint == right.fingerprint:
            return
        if left.label == right.label and left.label in _ALIGNED_SEQUENCES:
            for i, j in alignments[(id(left), id(right))]:
                if i is None:
                    assert j is not None
                    child = right.children[j]
                    differences.append(
                        SyntaxDifference(
                            "inserted statement",
                            path,
                            -1,
                            "",
                            _describe(child),
                            left.line,
                            child.line,
                        )
                    )
                elif j is None:
                    child = left.children[i]
                    differences.append(
                        SyntaxDifference(
                            "removed statement",
                            path,
                            -1,
                            _describe(child),
                            "",
                            child.line,
                            right.line,
                        )
                    )
                else:
                    emit(left.children[i], right.children[j], f"{path}[{i}:{j}]")
        elif left.label != right.label or len(left.children) != len(right.children):
            identity = (left.fingerprint, right.fingerprint)
            hole = holes.setdefault(identity, len(holes))
            kind = (
                "binding"
                if left.label.startswith("binding:") or right.label.startswith("binding:")
                else "syntax"
            )
            differences.append(
                SyntaxDifference(
                    kind, path, hole, _describe(left), _describe(right), left.line, right.line
                )
            )
        else:
            for index, (a, b) in enumerate(zip(left.children, right.children, strict=True)):
                suffix = (
                    a.label.removeprefix("field:") if a.label.startswith("field:") else str(index)
                )
                emit(a, b, f"{path}.{suffix}")

    distance = cost(first, second)
    emit(first, second, "body")
    total = first.size + second.size
    shared = (total - distance) / 2
    return AstComparison((total - distance) / total if total else 1.0, shared, tuple(differences))


@dataclass(frozen=True, slots=True)
class IndexedFunction:
    definition: PythonDefinition
    patterns: PatternCounts
    node_count: int
    signature: str
    enclosing: tuple[PythonDefinition, ...]


def _patterns(node: FunctionNode) -> PatternCounts:
    wrapper = ast.Module(body=list(_function_body(node)), type_ignores=[])
    patterns: Counter[Pattern] = Counter()
    for child in ast.walk(wrapper):
        if child is wrapper:
            continue
        fields = []
        for name, value in ast.iter_fields(child):
            if isinstance(value, ast.AST):
                fields.append((name, (type(value).__name__,)))
            elif isinstance(value, list):
                fields.append(
                    (
                        name,
                        tuple(type(item).__name__ for item in value if isinstance(item, ast.AST)),
                    )
                )
        patterns[(type(child).__name__, tuple(fields))] += 1
    return tuple(patterns.items())


def index_functions(
    repo_root: Path, paths: Sequence[Path], *, min_nodes: int
) -> tuple[IndexedFunction, ...]:
    definitions = collect_python_definitions(repo_root, paths)
    return _index_definitions(definitions, min_nodes=min_nodes)


def _index_definitions(
    definitions: Sequence[PythonDefinition], *, min_nodes: int
) -> tuple[IndexedFunction, ...]:
    by_path: dict[str, list[PythonDefinition]] = defaultdict(list)
    for definition in definitions:
        if definition.kind in {"class", "function"}:
            by_path[definition.path].append(definition)
    indexed = []
    for definition in definitions:
        if definition.kind != "function":
            continue
        node = cast("FunctionNode", definition.node)
        node_count = sum(1 for statement in _function_body(node) for _ in ast.walk(statement))
        if node_count < min_nodes:
            continue
        parents = sorted(
            (
                parent
                for parent in by_path[definition.path]
                if parent.kind == "function"
                and parent.start < definition.start
                and definition.end <= parent.end
            ),
            key=lambda parent: parent.start,
        )
        signature = f"({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        indexed.append(
            IndexedFunction(definition, _patterns(node), node_count, signature, tuple(parents))
        )
    return tuple(indexed)


def _comparison_terms(
    functions: Sequence[IndexedFunction], pairs: Sequence[tuple[int, int]]
) -> Mapping[int, AstTerm]:
    participating = {index for pair in pairs for index in pair}
    terms = {}
    for index in sorted(participating):
        function = functions[index]
        outer = tuple(_scope(cast("FunctionNode", parent.node)) for parent in function.enclosing)
        terms[index] = normalize_function(cast("FunctionNode", function.definition.node), outer)
    return MappingProxyType(terms)


def _nested_pair(first: PythonDefinition, second: PythonDefinition) -> bool:
    return first.path == second.path and (
        (first.start <= second.start and second.end <= first.end)
        or (second.start <= first.start and first.end <= second.end)
    )


@dataclass(frozen=True, slots=True)
class _RetrievalFunction:
    """Worker metadata without source ASTs or normalized comparison trees."""

    definition: PythonDefinition
    patterns: PatternCounts
    node_count: int


def candidate_pairs(
    functions: Sequence[IndexedFunction | _RetrievalFunction],
    selection: SourceSelection,
    *,
    neighbors: int,
    retrieval_threshold: float,
) -> tuple[tuple[int, int], ...]:
    selected = {
        i
        for i, f in enumerate(functions)
        if selection.includes(f.definition.path, f.definition.start, f.definition.end)
    }
    locations = {
        (f.definition.path, f.definition.start, f.definition.qualname): i
        for i, f in enumerate(functions)
    }
    pairs: set[tuple[int, int]] = set()
    for candidate in ast_candidates(
        tuple(f.definition for f in functions), selection=selection, deep=False
    ):
        first = locations[(candidate.first.path, candidate.first.start, candidate.first.label)]
        second = locations[(candidate.second.path, candidate.second.start, candidate.second.label)]
        if not _nested_pair(functions[first].definition, functions[second].definition):
            pairs.add((min(first, second), max(first, second)))
    postings: dict[Pattern, list[tuple[int, int]]] = defaultdict(list)
    norms = tuple(math.sqrt(sum(v * v for _, v in f.patterns)) for f in functions)
    for j, function in enumerate(functions):
        for pattern, count in function.patterns:
            postings[pattern].append((j, count))
    for i in sorted(selected):
        first = functions[i]
        eligible = frozenset(
            j
            for j, second in enumerate(functions)
            if i != j
            and not _nested_pair(first.definition, second.definition)
            and min(first.node_count, second.node_count) / max(first.node_count, second.node_count)
            >= 0.6
        )
        dots: dict[int, int] = defaultdict(int)
        for pattern, count in first.patterns:
            for j, other_count in postings[pattern]:
                if j in eligible:
                    dots[j] += count * other_count
        ranked = heapq.nlargest(
            neighbors, ((dot / (norms[i] * norms[j]), j) for j, dot in dots.items())
        )
        for score, j in ranked:
            if score >= retrieval_threshold:
                pairs.add((min(i, j), max(i, j)))
    return tuple(sorted(pairs))


@dataclass(frozen=True, slots=True)
class SimilarityFinding:
    first: IndexedFunction
    second: IndexedFunction
    comparison: AstComparison


def find_similarities(
    functions: Sequence[IndexedFunction],
    pairs: Sequence[tuple[int, int]],
    *,
    threshold: float,
) -> tuple[SimilarityFinding, ...]:
    terms = _comparison_terms(functions, pairs)
    findings = []
    for i, j in pairs:
        comparison = compare_terms(terms[i], terms[j])
        if comparison.score >= threshold:
            findings.append(SimilarityFinding(functions[i], functions[j], comparison))
    return tuple(
        sorted(
            findings,
            key=lambda f: (
                -f.comparison.shared_size,
                -f.comparison.score,
                f.first.definition.identity,
                f.second.definition.identity,
            ),
        )
    )


def _selection_chunks(
    functions: Sequence[IndexedFunction], selection: SourceSelection, *, workers: int
) -> tuple[SourceSelection, ...]:
    # Keep each file together so enclosing and nested ranges cannot select the
    # same function in multiple workers. Balance by the number of query functions.
    by_path: dict[str, list[PythonDefinition]] = defaultdict(list)
    for function in functions:
        definition = function.definition
        if selection.includes(definition.path, definition.start, definition.end):
            by_path[definition.path].append(definition)
    if not by_path:
        return ()
    groups: list[dict[str, tuple[LineRange, ...]]] = [{} for _ in range(min(workers, len(by_path)))]
    sizes = [0] * len(groups)
    for path in sorted(by_path, key=lambda path: (-len(by_path[path]), path)):
        index = min(range(len(groups)), key=sizes.__getitem__)
        groups[index][path] = (
            (LineRange(1, max(definition.end for definition in by_path[path])),)
            if selection.ranges is None
            else selection.ranges[path]
        )
        sizes[index] += len(by_path[path])
    return tuple(SourceSelection(group, selection.description) for group in groups)


def search_sources(
    repo_root: Path,
    paths: Sequence[Path],
    selection: SourceSelection,
    *,
    min_nodes: int,
    threshold: float,
    neighbors: int,
    retrieval_threshold: float,
    workers: int,
) -> tuple[tuple[IndexedFunction, ...], tuple[SimilarityFinding, ...]]:
    worker_count = min(workers, len(paths))
    if worker_count <= 1:
        functions = index_functions(repo_root, paths, min_nodes=min_nodes)
        pairs = candidate_pairs(
            functions, selection, neighbors=neighbors, retrieval_threshold=retrieval_threshold
        )
    else:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=get_context("spawn")) as pool:
            # Parse files in workers; defer binding-aware normalization until
            # candidate retrieval has identified the functions to compare.
            parsed = pool.map(
                partial(collect_python_definitions, repo_root),
                ((path,) for path in sorted(paths)),
                chunksize=max(1, len(paths) // (4 * worker_count)),
            )
            definitions = tuple(chain.from_iterable(parsed))
            functions = _index_definitions(definitions, min_nodes=min_nodes)
            retrieval = tuple(
                _RetrievalFunction(replace(f.definition, node=None), f.patterns, f.node_count)
                for f in functions
            )
            retrieved = pool.map(
                partial(
                    candidate_pairs,
                    retrieval,
                    neighbors=neighbors,
                    retrieval_threshold=retrieval_threshold,
                ),
                _selection_chunks(functions, selection, workers=worker_count),
            )
            pairs = tuple(sorted(set(chain.from_iterable(retrieved))))
    return functions, find_similarities(functions, pairs, threshold=threshold)


def render_report(
    functions: Sequence[IndexedFunction],
    findings: Sequence[SimilarityFinding],
    *,
    limit: int,
    selection: SourceSelection,
) -> str:
    lines = [
        "AST similarity review (advisory; scores describe syntax)",
        f"Scope: {selection.description}",
        f"Indexed {len(functions)} Python functions; showing {min(limit, len(findings))} of {len(findings)} pairs.",
    ]
    for finding in findings[:limit]:
        comparison = finding.comparison
        lines.extend(
            (
                "",
                f"Retained syntax {comparison.score:.2f}; shared weight {comparison.shared_size:.0f}",
            )
        )
        for function in (finding.first, finding.second):
            definition = function.definition
            lines.append(
                f"  {definition.path}:{definition.start} {definition.qualname}{function.signature}"
            )
        if not comparison.differences:
            lines.append("  Same normalized body; inspect signatures and domain contracts.")
        groups: dict[tuple[int, int], list[SyntaxDifference]] = defaultdict(list)
        for index, difference in enumerate(comparison.differences):
            # A shared hole records a consistent substitution. Show it once
            # in text; JSON retains every location.
            identity = (difference.hole, -1 if difference.hole >= 0 else index)
            groups[identity].append(difference)
        for group in tuple(groups.values())[:8]:
            difference = group[0]
            hole = f"H{difference.hole} " if difference.hole >= 0 else ""
            count = f" ({len(group)} occurrences)" if len(group) > 1 else ""
            lines.append(
                f"  {hole}{difference.kind}{count}: {difference.left or '(absent)'} -> {difference.right or '(absent)'}"
            )
            lines.append(
                f"    lines {difference.left_line}/{difference.right_line}; {difference.path}"
            )
        if len(groups) > 8:
            lines.append(
                f"  ... {len(groups) - 8} more difference groups; use --json for all evidence."
            )
    lines.append(
        "\nReview common helpers and invariant owners; similarity does not establish interchangeable behavior."
    )
    return "\n".join(lines)


def _json_report(
    functions: Sequence[IndexedFunction],
    findings: Sequence[SimilarityFinding],
    *,
    limit: int,
    selection: SourceSelection,
) -> dict[str, object]:
    def location(function: IndexedFunction) -> dict[str, object]:
        definition = function.definition
        return {
            "path": definition.path,
            "line": definition.start,
            "end": definition.end,
            "name": definition.qualname,
            "signature": function.signature,
            "nodes": function.node_count,
        }

    return {
        "advisory": True,
        "scope": selection.description,
        "indexed_functions": len(functions),
        "matching_pairs": len(findings),
        "findings": [
            {
                "first": location(f.first),
                "second": location(f.second),
                "score": f.comparison.score,
                "shared_weight": f.comparison.shared_size,
                "differences": [
                    {
                        "kind": d.kind,
                        "path": d.path,
                        "hole": d.hole,
                        "left": d.left,
                        "right": d.right,
                        "left_line": d.left_line,
                        "right_line": d.right_line,
                    }
                    for d in f.comparison.differences
                ],
            }
            for f in findings[:limit]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="repository-relative Python source files/directories; compare against all production Python",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="minimum retained syntax score (default: %(default)s)",
    )
    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=DEFAULT_RETRIEVAL_THRESHOLD,
        help="minimum AST-pattern cosine for additional candidates (default: %(default)s)",
    )
    parser.add_argument("--min-nodes", type=int, default=DEFAULT_MIN_FUNCTION_NODES)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="maximum ranked pairs to display (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="CPU worker processes; use 1 for serial execution (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable evidence for the displayed pairs"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or not 0 <= args.retrieval_threshold <= 1:
        parser.error("thresholds must be between 0 and 1")
    if min(args.min_nodes, args.neighbors, args.limit, args.workers) < 1:
        parser.error("--min-nodes, --neighbors, --limit and --workers must be positive")
    try:
        selection = (
            explicit_selection(REPO_ROOT, args.targets)
            if args.targets
            else SourceSelection(None, "all production Python")
        )
        if args.targets:
            assert selection.ranges is not None
            if not any(path.endswith(".py") for path in selection.ranges):
                raise DuplicateAuditError("targets contain no production Python source")
        paths = tuple(path for path in source_paths(REPO_ROOT) if path.suffix == ".py")
        functions, findings = search_sources(
            REPO_ROOT,
            paths,
            selection,
            min_nodes=args.min_nodes,
            threshold=args.threshold,
            neighbors=args.neighbors,
            retrieval_threshold=args.retrieval_threshold,
            workers=args.workers,
        )
        if args.json:
            print(
                json.dumps(
                    _json_report(functions, findings, limit=args.limit, selection=selection),
                    indent=2,
                )
            )
        else:
            print(render_report(functions, findings, limit=args.limit, selection=selection))
    except (DuplicateAuditError, OSError) as exc:
        print(f"AST similarity review failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
