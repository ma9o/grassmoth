"""Rank likely duplicate classes and functions for local agent review.

The audit is deliberately advisory. By default it compares definitions touched
since the merge base with ``origin/master`` against the entire source tree. A
token-clone pass catches copied blocks in Python and TypeScript, while a Python
AST pass catches short alpha-equivalent functions and data models with strongly
overlapping fields. Source-only Python passes also report identical declared
field signatures, repeated rows ranked by distinct conversion sites, generic
families, presence-dependency evidence grouped by its validator owner, and
repeated closed vocabularies. No production package is imported. Matching
syntax is evidence for review, not full contract or semantic equivalence.

``bun run lint`` runs the default diff-aware audit.
Usage from the repository root::

    bun run --cwd apps/grassmoth-server lint:duplicates
    bun run --cwd apps/grassmoth-server lint:duplicates --deep
    bun run --cwd apps/grassmoth-server lint:duplicates --all
    bun run --cwd apps/grassmoth-server lint:duplicates --include-reviewed
    bun run --cwd apps/grassmoth-server lint:duplicates packages/grassmoth/src/grassmoth/models/ssm

Candidates are retrieval results, not claims of semantic equivalence. The
reviewing agent decides whether to consolidate the pair, preserve an intentional
boundary mirror, or leave related implementations separate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOTS = (
    Path("packages/grassmoth/src"),
    Path("apps/grassmoth-server/src"),
    Path("apps/grassmoth-client/src"),
    Path("packages/api-types/src"),
)
PYTHON_SOURCE_ROOTS = (Path("packages/grassmoth/src"), Path("apps/grassmoth-server/src"))
SUPPORTED_SUFFIXES = frozenset({".py", ".ts", ".tsx"})
IGNORED_PARTS = frozenset({"generated", "node_modules", "__pycache__", ".next"})
IGNORED_NAME_MARKERS = (".test.", ".spec.", ".stories.")

DEFAULT_LIMIT = 30
DEEP_LIMIT = 60
DEFAULT_CLASS_THRESHOLD = 0.78
DEEP_CLASS_THRESHOLD = 0.68
DEFAULT_FUNCTION_THRESHOLD = 0.82
DEEP_FUNCTION_THRESHOLD = 0.72
DEFAULT_MIN_FUNCTION_NODES = 35
DEEP_MIN_FUNCTION_NODES = 25
REVIEWED_PAIRS_PATH = REPO_ROOT / "apps/grassmoth-server/scripts/checks/duplicate_reviews.json"
TYPE_CATEGORIES = (
    "identical fields",
    "repeated row",
    "generic family",
    "presence dependency",
    "closed vocabulary",
)

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NAME_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "build",
        "by",
        "create",
        "for",
        "from",
        "get",
        "in",
        "is",
        "make",
        "of",
        "on",
        "or",
        "run",
        "set",
        "the",
        "to",
        "with",
    }
)
_UNINFORMATIVE_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "array",
        "asarray",
        "bool",
        "cast",
        "dict",
        "enumerate",
        "float",
        "get",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "items",
        "len",
        "list",
        "max",
        "min",
        "range",
        "set",
        "str",
        "sum",
        "tuple",
        "where",
        "zip",
    }
)
_CONTROL_NODE_NAMES = frozenset(
    {
        "Assert",
        "AsyncFor",
        "AsyncWith",
        "Await",
        "Break",
        "Continue",
        "For",
        "GeneratorExp",
        "If",
        "IfExp",
        "ListComp",
        "Match",
        "Raise",
        "Return",
        "SetComp",
        "Try",
        "TryStar",
        "While",
        "With",
        "Yield",
        "YieldFrom",
    }
)


class DuplicateAuditError(RuntimeError):
    """An operational failure that prevents a trustworthy audit."""


@dataclass(frozen=True, slots=True)
class LineRange:
    """Inclusive source-line range."""

    start: int
    end: int

    def overlaps(self, start: int, end: int) -> bool:
        return self.start <= end and start <= self.end


@dataclass(frozen=True)
class SourceSelection:
    """Definitions that seed comparisons against the full repository."""

    ranges: dict[str, tuple[LineRange, ...]] | None
    description: str

    @property
    def is_all(self) -> bool:
        return self.ranges is None

    @property
    def path_count(self) -> int:
        return len(self.ranges) if self.ranges is not None else 0

    def includes(self, path: str, start: int, end: int) -> bool:
        if self.ranges is None:
            return True
        return any(line_range.overlaps(start, end) for line_range in self.ranges.get(path, ()))


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    annotation: str
    default: str
    optional: bool = False
    singleton_tag: bool = False


@dataclass(frozen=True)
class PythonDefinition:
    kind: str
    path: str
    name: str
    qualname: str
    start: int
    end: int
    fields: tuple[FieldSpec, ...] = ()
    methods: frozenset[str] = frozenset()
    bases: frozenset[str] = frozenset()
    fingerprint: str = ""
    node_counts: tuple[tuple[str, int], ...] = ()
    control_counts: tuple[tuple[str, int], ...] = ()
    calls: frozenset[str] = frozenset()
    signature: tuple[str, ...] = ()
    node_count: int = 0
    is_method: bool = False
    parent_class: str | None = None
    declared_fields: tuple[FieldSpec, ...] = ()
    node: ast.AST | None = None
    module: str = ""
    imports: tuple[tuple[str, str], ...] = ()
    vocabulary: frozenset[tuple[str, str]] = frozenset()
    vocabulary_kind: str = ""

    @property
    def identity(self) -> tuple[str, int, str]:
        return (self.path, self.start, self.qualname)


@dataclass(frozen=True, slots=True)
class Location:
    path: str
    start: int
    end: int
    label: str = ""

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (self.path, self.start, self.end, self.label)


@dataclass(frozen=True)
class Candidate:
    category: str
    score: float
    first: Location
    second: Location
    reason: str
    first_selected: bool
    second_selected: bool
    review_fingerprint: str = ""
    evidence_locations: tuple[Location, ...] = ()

    @property
    def level(self) -> str:
        return "HIGH" if self.score >= 0.9 else "MEDIUM"

    @property
    def identity(self) -> tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]]:
        first, second = sorted((self.first.identity, self.second.identity))
        return (self.category, first, second)


@dataclass(frozen=True)
class ConversionWitness:
    target: PythonDefinition
    fields: frozenset[str]
    location: Location
    fingerprint: str
    position: tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class ReviewedPair:
    category: str
    first: str
    second: str
    fingerprint: str
    classification: str
    rationale: str

    @property
    def identity(self) -> tuple[str, str, str]:
        first, second = sorted((self.first, self.second))
        return (self.category, first, second)


def _run_command(args: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DuplicateAuditError(f"command failed ({' '.join(args)}): {detail}")
    return completed.stdout


def _git(args: Sequence[str], *, repo_root: Path) -> str:
    return _run_command(("git", *args), cwd=repo_root)


def _is_source_path(path: Path) -> bool:
    if path.suffix not in SUPPORTED_SUFFIXES:
        return False
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    if any(marker in path.name for marker in IGNORED_NAME_MARKERS):
        return False
    return not (path.suffix == ".py" and path.name.startswith("test_"))


def _is_below_source_root(path: Path) -> bool:
    return any(path == root or path.is_relative_to(root) for root in SOURCE_ROOTS)


def source_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return every production Python/TypeScript source path."""
    paths: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        paths.extend(
            path.relative_to(repo_root)
            for path in root.rglob("*")
            if path.is_file() and _is_source_path(path.relative_to(repo_root))
        )
    return tuple(sorted(set(paths)))


def parse_changed_ranges(diff_text: str) -> tuple[LineRange, ...]:
    """Extract inclusive new-file ranges from a zero-context unified diff."""
    ranges: list[LineRange] = []
    for line in diff_text.splitlines():
        match = _HUNK_HEADER.match(line)
        if match is None:
            continue
        start = int(match.group("start"))
        count_text = match.group("count")
        count = 1 if count_text is None else int(count_text)
        end = start if count == 0 else start + count - 1
        ranges.append(LineRange(start=max(1, start), end=max(1, end)))
    return tuple(ranges)


def changed_selection(repo_root: Path, *, base_ref: str) -> SourceSelection:
    """Select changed source lines relative to the merge base with ``base_ref``."""
    try:
        merge_base = _git(("merge-base", "HEAD", base_ref), repo_root=repo_root).strip()
    except DuplicateAuditError as exc:
        raise DuplicateAuditError(
            f"cannot resolve merge base with {base_ref!r}; pass --base or use --all"
        ) from exc
    if not merge_base:
        raise DuplicateAuditError(f"git returned no merge base for {base_ref!r}")

    tracked_text = _git(
        ("diff", "--name-only", "--diff-filter=ACMR", merge_base, "--"),
        repo_root=repo_root,
    )
    untracked_text = _git(
        ("ls-files", "--others", "--exclude-standard"),
        repo_root=repo_root,
    )
    tracked = {Path(line) for line in tracked_text.splitlines() if line}
    untracked = {Path(line) for line in untracked_text.splitlines() if line}
    selected: dict[str, tuple[LineRange, ...]] = {}

    for path in sorted(tracked | untracked):
        if not _is_below_source_root(path) or not _is_source_path(path):
            continue
        relative = path.as_posix()
        if path in untracked:
            selected[relative] = (LineRange(1, sys.maxsize),)
            continue
        diff_text = _git(
            ("diff", "--unified=0", "--no-color", merge_base, "--", relative),
            repo_root=repo_root,
        )
        ranges = parse_changed_ranges(diff_text)
        if ranges:
            selected[relative] = ranges

    return SourceSelection(
        ranges=selected,
        description=f"changed definitions since merge base with {base_ref}",
    )


def explicit_selection(repo_root: Path, targets: Sequence[str]) -> SourceSelection:
    """Select all lines in explicit source files or directories."""
    selected: dict[str, tuple[LineRange, ...]] = {}
    resolved_root = repo_root.resolve()
    for target_text in targets:
        target = (repo_root / target_text).resolve()
        try:
            relative_target = target.relative_to(resolved_root)
        except ValueError as exc:
            raise DuplicateAuditError(f"target is outside the repository: {target_text}") from exc
        if not target.exists():
            raise DuplicateAuditError(f"target does not exist: {target_text}")
        candidates = (target,) if target.is_file() else target.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(resolved_root)
            if _is_below_source_root(relative) and _is_source_path(relative):
                selected[relative.as_posix()] = (LineRange(1, sys.maxsize),)
        if target.is_file() and relative_target.as_posix() not in selected:
            raise DuplicateAuditError(f"target is not a production source file: {target_text}")
    return SourceSelection(
        ranges=selected,
        description=f"{len(selected)} explicitly selected source file(s)",
    )


def _terminal_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return ast.unparse(node)


def _name_words(name: str) -> frozenset[str]:
    snake_case = _WORD_BOUNDARY.sub(r"\1_\2", name).lower()
    return frozenset(
        word
        for word in re.split(r"[^a-z0-9]+", snake_case)
        if len(word) > 1 and word not in _NAME_STOP_WORDS
    )


def _jaccard(left: frozenset[str] | set[str], right: frozenset[str] | set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(
    left_items: tuple[tuple[str, int], ...],
    right_items: tuple[tuple[str, int], ...],
) -> float:
    left = dict(left_items)
    right = dict(right_items)
    dot = sum(left[key] * right[key] for key in left.keys() & right.keys())
    denominator = math.sqrt(
        sum(value * value for value in left.values())
        * sum(value * value for value in right.values())
    )
    return dot / denominator if denominator else 0.0


def _scope_nodes(root: ast.AST) -> Iterator[ast.AST]:
    """Walk one executable scope without descending into nested definitions."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if node is not root and isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _function_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    """Return executable statements, excluding a leading documentation string."""
    body = tuple(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _function_body_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Walk executable function bodies without counting signatures or annotations."""
    wrapper = ast.Module(body=list(_function_body(node)), type_ignores=[])
    for child in _scope_nodes(wrapper):
        if child is not wrapper:
            yield child


def _is_function_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a definition is an interface or abstract placeholder."""
    body = _function_body(node)
    if not body:
        return True
    if len(body) != 1:
        return False

    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    if isinstance(statement, ast.Return):
        return isinstance(statement.value, ast.Name) and statement.value.id == "NotImplemented"
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return False

    exception = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
    return _terminal_name(exception) == "NotImplementedError"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    def annotation(arg: ast.arg) -> str:
        return ast.unparse(arg.annotation) if arg.annotation is not None else ""

    args = node.args
    result = [annotation(arg) for arg in args.posonlyargs]
    result.append("/")
    result.extend(annotation(arg) for arg in args.args if arg.arg not in {"self", "cls"})
    if args.vararg is not None:
        result.append(f"*{annotation(args.vararg)}")
    else:
        result.append("*")
    result.extend(annotation(arg) for arg in args.kwonlyargs)
    if args.kwarg is not None:
        result.append(f"**{annotation(args.kwarg)}")
    result.append(f"->{ast.unparse(node.returns) if node.returns is not None else ''}")
    return tuple(result)


def _local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    ordered: list[str] = []

    def add(name: str) -> None:
        if name not in ordered:
            ordered.append(name)

    args = node.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        add(arg.arg)
    if args.vararg is not None:
        add(args.vararg.arg)
    if args.kwarg is not None:
        add(args.kwarg.arg)
    for child in _scope_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                add(alias.asname or alias.name.split(".", maxsplit=1)[0])
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            add(child.name)
    return {name: f"_local_{index}" for index, name in enumerate(ordered)}


def _normalized_ast_value(
    value: object,
    *,
    locals_by_name: dict[str, str],
    root: ast.AST,
) -> object:
    if isinstance(value, ast.Name):
        return (
            "Name",
            locals_by_name.get(value.id, value.id),
            type(value.ctx).__name__,
        )
    if isinstance(value, ast.arg):
        return (
            "arg",
            locals_by_name.get(value.arg, value.arg),
            _normalized_ast_value(value.annotation, locals_by_name=locals_by_name, root=root)
            if value.annotation is not None
            else None,
        )
    if isinstance(value, ast.Constant):
        normalized = (
            value.value
            if value.value is None or isinstance(value.value, bool)
            else type(value.value).__name__
        )
        return ("Constant", normalized)
    if (
        isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and value is not root
    ):
        return (type(value).__name__, "nested-definition")
    if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = value.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return (
            type(value).__name__,
            _normalized_ast_value(value.args, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(value.returns, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(value.decorator_list, locals_by_name=locals_by_name, root=root),
            _normalized_ast_value(body, locals_by_name=locals_by_name, root=root),
        )
    if isinstance(value, ast.AST):
        return (
            type(value).__name__,
            tuple(
                (
                    field,
                    _normalized_ast_value(child, locals_by_name=locals_by_name, root=root),
                )
                for field, child in ast.iter_fields(value)
                if field != "type_comment"
            ),
        )
    if isinstance(value, list):
        return tuple(
            _normalized_ast_value(child, locals_by_name=locals_by_name, root=root)
            for child in value
        )
    return value


def _function_fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = _normalized_ast_value(node, locals_by_name=_local_names(node), root=node)
    return hashlib.sha256(repr(normalized).encode()).hexdigest()


def _field_default(value: ast.expr | None) -> str:
    return ast.dump(value, include_attributes=False) if value is not None else ""


def _literal_values(node: ast.AST) -> frozenset[tuple[str, str]]:
    if not isinstance(node, ast.Subscript) or _terminal_name(node.value) != "Literal":
        return frozenset()
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
    if not all(isinstance(item, ast.Constant) for item in items):
        return frozenset()
    return frozenset(
        (type(item.value).__name__, repr(item.value))
        for item in items
        if isinstance(item, ast.Constant)
    )


def _annotation_ast(annotation: str) -> ast.expr:
    node = ast.parse(annotation, mode="eval").body
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    return node


def _field_spec(name: str, annotation: ast.expr, value: ast.expr | None) -> FieldSpec:
    expression = _annotation_ast(ast.unparse(annotation))
    optional = _annotation_admits_none(expression)
    if isinstance(value, ast.Constant) and value.value is None:
        optional = True
    if isinstance(value, ast.Call) and _terminal_name(value.func) in {"Field", "field"}:
        defaults = [kw.value for kw in value.keywords if kw.arg == "default"]
        if value.args:
            defaults.append(value.args[0])
        optional |= any(isinstance(item, ast.Constant) and item.value is None for item in defaults)
    tag = expression
    if isinstance(tag, ast.Subscript) and _terminal_name(tag.value) == "Annotated":
        tag = tag.slice.elts[0] if isinstance(tag.slice, ast.Tuple) else tag.slice
    return FieldSpec(
        name,
        ast.unparse(annotation),
        _field_default(value),
        optional=optional,
        singleton_tag=(
            isinstance(tag, ast.Subscript)
            and _terminal_name(tag.value) == "Literal"
            and (not isinstance(tag.slice, ast.Tuple) or len(tag.slice.elts) == 1)
        ),
    )


def _annotation_admits_none(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_admits_none(node.left) or _annotation_admits_none(node.right)
    if isinstance(node, ast.Subscript):
        name = _terminal_name(node.value)
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name == "Optional":
            return True
        if name == "Annotated":
            return _annotation_admits_none(items[0])
        if name in {"Union", "Literal"}:
            return any(_annotation_admits_none(item) for item in items)
    return False


def _module_name(path: str) -> str:
    relative = (
        Path(path)
        .relative_to(next(root for root in PYTHON_SOURCE_ROOTS if Path(path).is_relative_to(root)))
        .with_suffix("")
    )
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _module_imports(tree: ast.Module, path: str) -> tuple[tuple[str, str], ...]:
    module = _module_name(path)
    package = module.split(".") if Path(path).stem == "__init__" else module.split(".")[:-1]
    bindings: dict[str, str] = {}
    for node in _scope_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                prefix = ".".join((*package[: len(package) - node.level + 1], prefix)).rstrip(".")
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{prefix}.{alias.name}"
    return tuple(sorted(bindings.items()))


def _self_fields(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[FieldSpec]:
    for child in _scope_nodes(node):
        targets: Sequence[ast.expr]
        annotation = ""
        value: ast.expr | None
        if isinstance(child, ast.Assign):
            targets = child.targets
            value = child.value
        elif isinstance(child, ast.AnnAssign):
            targets = (child.target,)
            annotation = ast.unparse(child.annotation)
            value = child.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                yield FieldSpec(target.attr, annotation, _field_default(value))


def _class_definition(
    node: ast.ClassDef,
    *,
    path: str,
    qualname: str,
) -> PythonDefinition:
    fields: dict[str, FieldSpec] = {}
    declared: dict[str, FieldSpec] = {}
    methods: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            field = _field_spec(statement.target.id, statement.annotation, statement.value)
            fields[field.name] = declared[field.name] = field
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(statement.name)
            if statement.name == "__init__":
                for field in _self_fields(statement):
                    fields.setdefault(field.name, field)
    return PythonDefinition(
        kind="class",
        path=path,
        name=node.name,
        qualname=qualname,
        start=node.lineno,
        end=node.end_lineno or node.lineno,
        fields=tuple(sorted(fields.values(), key=lambda field: field.name)),
        methods=frozenset(methods),
        bases=frozenset(_terminal_name(base) for base in node.bases),
        declared_fields=tuple(sorted(declared.values(), key=lambda field: field.name)),
        node=node,
        module=_module_name(path),
    )


def _function_definition(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    path: str,
    qualname: str,
    is_method: bool,
    parent_class: str | None,
) -> PythonDefinition:
    nodes = tuple(_function_body_nodes(node))
    node_counts = Counter(type(child).__name__ for child in nodes)
    control_counts = Counter(
        type(child).__name__ for child in nodes if type(child).__name__ in _CONTROL_NODE_NAMES
    )
    calls = frozenset(
        name
        for child in nodes
        if isinstance(child, ast.Call)
        for name in (_call_name(child.func),)
        if name
    )
    return PythonDefinition(
        kind="function",
        path=path,
        name=node.name,
        qualname=qualname,
        start=node.lineno,
        end=node.end_lineno or node.lineno,
        fingerprint=_function_fingerprint(node),
        node_counts=tuple(sorted(node_counts.items())),
        control_counts=tuple(sorted(control_counts.items())),
        calls=calls,
        signature=_function_signature(node),
        node_count=len(nodes),
        is_method=is_method,
        parent_class=parent_class,
        node=node,
        module=_module_name(path),
    )


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self, *, path: str, imports: tuple[tuple[str, str], ...]) -> None:
        self.path = path
        self.scope: list[tuple[str, str]] = []
        self.definitions: list[PythonDefinition] = []
        self.imports = imports

    def _qualname(self, name: str) -> str:
        return ".".join((*[scope_name for _, scope_name in self.scope], name))

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        definition = _class_definition(node, path=self.path, qualname=self._qualname(node.name))
        vocabulary: frozenset[tuple[str, str]] = frozenset()
        is_enum = any(_terminal_name(base) in {"Enum", "StrEnum", "IntEnum"} for base in node.bases)
        if is_enum:
            vocabulary = frozenset(
                (type(statement.value.value).__name__, repr(statement.value.value))
                for statement in node.body
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
                and isinstance(statement.value, ast.Constant)
            )
        self.definitions.append(
            replace(
                definition,
                imports=self.imports,
                vocabulary=vocabulary,
                vocabulary_kind="enum" if is_enum else "",
            )
        )
        self.scope.append(("class", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self.scope and self.scope[-1][0] == "class")
        parent_class = self.scope[-1][1] if is_method else None
        if not _is_function_stub(node):
            self.definitions.append(
                replace(
                    _function_definition(
                        node,
                        path=self.path,
                        qualname=self._qualname(node.name),
                        is_method=is_method,
                        parent_class=parent_class,
                    ),
                    imports=self.imports,
                )
            )
        self.scope.append(("function", node.name))
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments.extend(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
        for arg in arguments:
            if arg.annotation is not None:
                self._annotation_vocabularies(arg.annotation, self._qualname(arg.arg))
        if node.returns is not None:
            self._annotation_vocabularies(node.returns, self._qualname("return"))
        self.generic_visit(node)
        self.scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _vocabulary(self, node: ast.expr, *, name: str, kind: str) -> None:
        values = _literal_values(node)
        if not values or (kind == "inline" and len(values) < 2):
            return
        self.definitions.append(
            PythonDefinition(
                kind="vocabulary",
                path=self.path,
                name=name,
                qualname=name,
                start=node.lineno,
                end=node.end_lineno or node.lineno,
                node=node,
                module=_module_name(self.path),
                imports=self.imports,
                vocabulary=values,
                vocabulary_kind=kind,
            )
        )

    def _annotation_vocabularies(self, node: ast.expr, label: str) -> None:
        literals = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Subscript) and len(_literal_values(child)) >= 2
        ]
        for index, literal in enumerate(literals):
            name = label if len(literals) == 1 else f"{label}[{index + 1}]"
            self._vocabulary(literal, name=name, kind="inline")

    @override  # noqa: V105 -- ast.NodeVisitor dispatches this hook dynamically.
    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        if _literal_values(node.value):
            self._vocabulary(node.value, name=self._qualname(node.name.id), kind="alias")
        else:
            self._annotation_vocabularies(node.value, self._qualname(node.name.id))

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self._vocabulary(node.value, name=node.targets[0].id, kind="alias")
        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            label = self._qualname(node.target.id)
            if (
                not self.scope
                and _terminal_name(node.annotation) == "TypeAlias"
                and node.value is not None
            ):
                self._vocabulary(node.value, name=label, kind="alias")
            else:
                self._annotation_vocabularies(node.annotation, label)
        self.generic_visit(node)


def collect_python_definitions(
    repo_root: Path,
    paths: Iterable[Path],
) -> tuple[PythonDefinition, ...]:
    """Parse definitions under the production Python root without importing them."""
    definitions: list[PythonDefinition] = []
    for relative_path in sorted(
        path
        for path in paths
        if path.suffix == ".py" and any(path.is_relative_to(root) for root in PYTHON_SOURCE_ROOTS)
    ):
        path = repo_root / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise DuplicateAuditError(f"cannot parse {relative_path}: {exc}") from exc
        collector = _DefinitionCollector(
            path=relative_path.as_posix(), imports=_module_imports(tree, relative_path.as_posix())
        )
        collector.visit(tree)
        definitions.extend(collector.definitions)
    return tuple(definitions)


def _ordered_locations(
    first: PythonDefinition,
    second: PythonDefinition,
) -> tuple[Location, Location]:
    first_location = Location(first.path, first.start, first.end, first.qualname)
    second_location = Location(second.path, second.start, second.end, second.qualname)
    if first_location.identity <= second_location.identity:
        return first_location, second_location
    return second_location, first_location


def _definition_content_fingerprint(definition: PythonDefinition) -> str:
    if definition.kind == "function":
        return definition.fingerprint
    payload = (
        tuple((field.name, field.annotation, field.default) for field in definition.fields),
        tuple(sorted(definition.methods)),
        tuple(sorted(definition.bases)),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _candidate_review_fingerprint(
    first: PythonDefinition,
    second: PythonDefinition,
) -> str:
    """Fingerprint both definition bodies so source edits invalidate reviews."""
    content = sorted(
        (_definition_content_fingerprint(first), _definition_content_fingerprint(second))
    )
    return hashlib.sha256(f"{first.kind}:{content[0]}:{content[1]}".encode()).hexdigest()


def _definition_location(definition: PythonDefinition) -> Location:
    return Location(definition.path, definition.start, definition.end, definition.qualname)


def _type_candidate(
    category: str,
    members: Sequence[PythonDefinition],
    *,
    reason: str,
    score: float,
    selection: SourceSelection,
    sites: Sequence[Location] = (),
    site_fingerprints: Sequence[str] = (),
) -> Candidate | None:
    """Keep group membership and all witness bodies in the existing review scheme."""
    locations = [_definition_location(member) for member in members]
    locations.extend(sites)
    locations = list({location.identity: location for location in locations}.values())
    if not any(selection.includes(loc.path, loc.start, loc.end) for loc in locations):
        return None
    first, second, *evidence = locations
    content = sorted(
        (
            f"{member.path}::{member.qualname}",
            ast.dump(member.node, include_attributes=False) if member.node is not None else "",
        )
        for member in members
    )
    fingerprint = hashlib.sha256(
        repr((category, content, sorted(site_fingerprints))).encode()
    ).hexdigest()
    return Candidate(
        category=category,
        score=score,
        first=first,
        second=second,
        reason=reason,
        first_selected=selection.includes(first.path, first.start, first.end),
        second_selected=selection.includes(second.path, second.start, second.end),
        review_fingerprint=fingerprint,
        evidence_locations=tuple(evidence),
    )


def _symbol_index(definitions: Sequence[PythonDefinition]) -> dict[str, PythonDefinition]:
    return {
        f"{definition.module}.{definition.qualname}": definition
        for definition in definitions
        if definition.kind == "class" or definition.vocabulary_kind == "alias"
    }


def _resolve_symbol(
    expression: ast.expr,
    context: PythonDefinition,
    symbols: dict[str, PythonDefinition],
) -> PythonDefinition | None:
    """Resolve declared local/import bindings; do not infer unbound names or run imports."""
    if isinstance(expression, ast.Subscript):
        expression = expression.value
    name = ast.unparse(expression)
    scope = context.qualname.split(".")[:-1]
    for depth in range(len(scope), -1, -1):
        qualified = ".".join((context.module, *scope[:depth], name))
        if qualified in symbols:
            return symbols[qualified]
    head, _, tail = name.partition(".")
    imported = dict(context.imports).get(head)
    if imported is not None:
        return symbols.get(f"{imported}.{tail}" if tail else imported)
    return None


def _class_ancestors(
    classes: Sequence[PythonDefinition], symbols: dict[str, PythonDefinition]
) -> dict[tuple[str, int, str], set[tuple[str, int, str]]]:
    parents = {
        cls.identity: tuple(
            parent.identity
            for base in cast("ast.ClassDef", cls.node).bases
            if (parent := _resolve_symbol(base, cls, symbols)) is not None
            and parent.kind == "class"
        )
        for cls in classes
    }
    ancestors: dict[tuple[str, int, str], set[tuple[str, int, str]]] = {}
    for cls in classes:
        visited: set[tuple[str, int, str]] = set()
        pending = list(parents[cls.identity])
        while pending:
            parent = pending.pop()
            if parent not in visited:
                visited.add(parent)
                pending.extend(parents[parent])
        ancestors[cls.identity] = visited
    return ancestors


def _related_classes(
    first: PythonDefinition,
    second: PythonDefinition,
    ancestors: dict[tuple[str, int, str], set[tuple[str, int, str]]],
) -> bool:
    return (
        first.identity in ancestors[second.identity] or second.identity in ancestors[first.identity]
    )


def _own_methods(members: Sequence[PythonDefinition]) -> str:
    return "own methods: " + "; ".join(
        f"{member.qualname} [{', '.join(sorted(member.methods)) or 'none'}]" for member in members
    )


def _conversion_witnesses(
    definitions: Sequence[PythonDefinition], symbols: dict[str, PythonDefinition]
) -> tuple[ConversionWitness, ...]:
    """Walk each lexical scope once; annotate captured names from enclosing functions."""
    functions = [definition for definition in definitions if definition.kind == "function"]
    witnesses: dict[tuple[str, int, int, str], ConversionWitness] = {}
    for function in functions:
        node = cast("ast.FunctionDef | ast.AsyncFunctionDef", function.node)
        annotations: dict[str, str] = {}
        enclosing = sorted(
            (
                outer
                for outer in functions
                if outer.path == function.path
                and function.qualname.startswith(f"{outer.qualname}.")
            ),
            key=lambda outer: len(outer.qualname),
        )
        for scope in (*enclosing, function):
            scope_node = cast("ast.FunctionDef | ast.AsyncFunctionDef", scope.node)
            for arg in (
                *scope_node.args.posonlyargs,
                *scope_node.args.args,
                *scope_node.args.kwonlyargs,
            ):
                if arg.annotation is not None:
                    annotations[arg.arg] = ast.unparse(arg.annotation)
            for child in _scope_nodes(scope_node):
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    annotations[child.target.id] = ast.unparse(child.annotation)
        if function.is_method:
            annotations["self"] = function.qualname.rsplit(".", 1)[0]
        for call in _scope_nodes(node):
            if not isinstance(call, ast.Call):
                continue
            target = _resolve_symbol(call.func, function, symbols)
            if target is None or target.kind != "class":
                continue
            by_source: dict[str, set[str]] = {}
            for kw in call.keywords:
                if (
                    kw.arg is not None
                    and isinstance(kw.value, ast.Attribute)
                    and isinstance(kw.value.value, ast.Name)
                    and kw.arg == kw.value.attr
                ):
                    by_source.setdefault(kw.value.value.id, set()).add(kw.arg)
            copied = [(source, fields) for source, fields in by_source.items() if len(fields) >= 3]
            if not copied:
                continue
            # A call can copy several rows; keep its source groups together and
            # deduplicate the count later by the file/call position.
            for source, fields in copied:
                location = Location(
                    function.path,
                    call.lineno,
                    call.end_lineno or call.lineno,
                    f"{function.qualname}: {annotations.get(source, source)} -> {target.qualname} "
                    f"[{', '.join(sorted(fields))}]",
                )
                witnesses[(function.path, call.lineno, call.col_offset, source)] = (
                    ConversionWitness(
                        target,
                        frozenset(fields),
                        location,
                        repr(
                            (
                                function.path,
                                location.label,
                                ast.dump(call, include_attributes=False),
                            )
                        ),
                        (function.path, call.lineno, call.col_offset),
                    )
                )
    return tuple(witnesses[key] for key in sorted(witnesses))


def _field_distinctions(members: Sequence[PythonDefinition]) -> str:
    """Name syntactic differences which the retrieval category does not equate."""
    defaults: dict[str, set[str]] = {}
    tags: list[str] = []
    contexts: list[str] = []
    for member in members:
        for field in member.declared_fields:
            defaults.setdefault(field.name, set()).add(field.default)
            if field.singleton_tag:
                tags.append(f"{member.qualname}.{field.name}: {field.annotation}")
        node = cast("ast.ClassDef", member.node)
        configuration = [
            ast.unparse(statement)
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model_config"
                for target in statement.targets
            )
        ]
        context = [
            *(ast.unparse(base) for base in node.bases),
            *(f"@{ast.unparse(deco)}" for deco in node.decorator_list),
            *configuration,
        ]
        if context:
            contexts.append(f"{member.qualname} [{'; '.join(context)}]")
    parts = [_own_methods(members)]
    different_defaults = sorted(name for name, values in defaults.items() if len(values) > 1)
    if different_defaults:
        parts.append(f"different default/Field expressions: {', '.join(different_defaults)}")
    if tags:
        parts.append(f"literal tags: {'; '.join(tags)}")
    if contexts:
        parts.append(f"declaration context: {'; '.join(contexts)}")
    return "; ".join(parts)


def _record_type_candidate(candidates: list[Candidate], candidate: Candidate | None) -> bool:
    if candidate is not None:
        candidates.append(candidate)
        return True
    return False


def _annotation_constructor(annotation: str) -> str | None:
    """A hole must retain a type constructor above it, rather than erase the whole type."""
    node = _annotation_ast(annotation)
    if isinstance(node, ast.Subscript):
        return f"subscript {ast.unparse(node.value)}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return "union |"
    return None


def _field_candidates(
    classes: Sequence[PythonDefinition],
    definitions: Sequence[PythonDefinition],
    *,
    symbols: dict[str, PythonDefinition],
    ancestors: dict[tuple[str, int, str], set[tuple[str, int, str]]],
    selection: SourceSelection,
) -> tuple[list[Candidate], set[frozenset[tuple[str, int, str]]]]:
    """Stronger field categories claim pairs before the percentage heuristic."""
    candidates: list[Candidate] = []
    represented: set[frozenset[tuple[str, int, str]]] = set()
    exact: dict[tuple[tuple[str, str, str], ...], list[PythonDefinition]] = {}
    rows: dict[frozenset[tuple[str, str]], list[PythonDefinition]] = {}
    payloads: dict[tuple[str, ...], list[PythonDefinition]] = {}
    for cls in classes:
        fields = cls.declared_fields
        if len(fields) >= 2:
            exact.setdefault(tuple((f.name, f.annotation, f.default) for f in fields), []).append(
                cls
            )
        if len(fields) >= 3:
            rows.setdefault(frozenset((f.name, f.annotation) for f in fields), []).append(cls)
        payload = tuple(f.name for f in fields if not f.singleton_tag)
        if payload:
            payloads.setdefault(payload, []).append(cls)
    for fields, group in exact.items():
        for first, second in combinations(group, 2):
            if _related_classes(first, second, ancestors):
                continue
            if _record_type_candidate(
                candidates,
                _type_candidate(
                    "identical fields",
                    (first, second),
                    selection=selection,
                    score=1.0,
                    reason=f"{len(fields)} identical declared field signatures [{', '.join(f[0] for f in fields)}]; {_field_distinctions((first, second))}",
                ),
            ):
                represented.add(frozenset((first.identity, second.identity)))

    witnesses = _conversion_witnesses(definitions, symbols)
    # Small rows claim their inclusion edges first; larger rows can still carry
    # different evidence, but do not repeat an already represented pair.
    for row, owners in sorted(rows.items(), key=lambda item: (len(item[0]), item[1][0].identity)):
        owner = owners[0]
        members = [owner]
        for cls in classes:
            pair = frozenset((owner.identity, cls.identity))
            if (
                cls.identity != owner.identity
                and pair not in represented
                and not _related_classes(owner, cls, ancestors)
                and row <= frozenset((f.name, f.annotation) for f in cls.declared_fields)
            ):
                members.append(cls)
        if len(members) < 2:
            continue
        member_ids = {member.identity for member in members}
        field_names = {name for name, _ in row}
        copies = {
            witness.position: witness
            for witness in witnesses
            if witness.target.identity in member_ids and len(witness.fields & field_names) >= 3
        }
        sites = tuple(witness.location for witness in copies.values())
        additions = "; ".join(
            f"{member.qualname} +[{', '.join(f.name for f in member.declared_fields if f.name not in field_names) or 'none'}]"
            for member in members[1:]
        )
        if _record_type_candidate(
            candidates,
            _type_candidate(
                "repeated row",
                members,
                selection=selection,
                score=0.90 + 0.08 * len(copies) / (len(copies) + 1),
                reason=f"whole {len(row)}-field row [{'; '.join(f'{name}: {annotation}' for name, annotation in sorted(row))}] in {len(members)} declarations; {len(copies)} distinct conversion site(s); extra fields: {additions}; {_field_distinctions(members)}",
                sites=sites,
                site_fingerprints=tuple(witness.fingerprint for witness in copies.values()),
            ),
        ):
            represented.update(
                frozenset((owner.identity, member.identity)) for member in members[1:]
            )

    for names, group in payloads.items():
        annotations = {
            cls.identity: {f.name: f.annotation for f in cls.declared_fields if not f.singleton_tag}
            for cls in group
        }
        constructors = {
            identity: {
                name: _annotation_constructor(annotation) for name, annotation in fields.items()
            }
            for identity, fields in annotations.items()
        }
        for anchor in group:
            family = [anchor]
            holes: set[str] = set()
            for cls in group:
                if cls.identity == anchor.identity:
                    continue
                if any(
                    frozenset((member.identity, cls.identity)) in represented
                    or _related_classes(member, cls, ancestors)
                    or annotations[member.identity] == annotations[cls.identity]
                    for member in family
                ):
                    continue
                differing = {
                    name
                    for name in names
                    if annotations[anchor.identity][name] != annotations[cls.identity][name]
                }
                combined_holes = holes | differing
                shared_fields = set(names) - combined_holes
                shared_roots = all(
                    constructors[anchor.identity][name] is not None
                    and all(
                        constructors[member.identity][name] == constructors[anchor.identity][name]
                        for member in (*family, cls)
                    )
                    for name in combined_holes
                )
                if 1 <= len(combined_holes) <= 2 and (shared_fields or shared_roots):
                    holes = combined_holes
                    family.append(cls)
            if len(family) < 2:
                continue
            details = "; ".join(
                f"{member.qualname} [{'; '.join(f'{name}: {annotations[member.identity][name]}' for name in sorted(holes))}]"
                for member in family
            )
            common = set(names) - holes
            skeleton = (
                f"identical payload fields [{', '.join(sorted(common))}]"
                if common
                else "roots above every hole ["
                + "; ".join(
                    f"{name}: {constructors[anchor.identity][name]}" for name in sorted(holes)
                )
                + "]"
            )
            if _record_type_candidate(
                candidates,
                _type_candidate(
                    "generic family",
                    family,
                    selection=selection,
                    score=0.88,
                    reason=f"{len(family)} declarations share payload names [{', '.join(names)}]; shared skeleton: {skeleton}; {len(holes)} annotation hole(s) [{', '.join(sorted(holes))}]: {details}; {_field_distinctions(family)}",
                ),
            ):
                represented.update(
                    frozenset((a.identity, b.identity)) for a, b in combinations(family, 2)
                )
    return candidates, represented


def _validator_methods(cls: PythonDefinition) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for statement in cast("ast.ClassDef", cls.node).body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            statement.name == "__post_init__"
            or any(
                _terminal_name(deco.func if isinstance(deco, ast.Call) else deco)
                in {"model_validator", "field_validator"}
                for deco in statement.decorator_list
            )
        ):
            yield statement


def _raising_conditions(
    body: Sequence[ast.stmt],
    *,
    context: tuple[ast.expr, ...] = (),
    aliases: dict[str, ast.expr] | None = None,
) -> Iterator[tuple[tuple[ast.expr, ...], dict[str, ast.expr]]]:
    """Carry enclosing branches and the preceding local bindings to each raise."""
    local = dict(aliases or {})
    for statement in body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    local[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                local[statement.target.id] = statement.value
        elif isinstance(statement, ast.Raise):
            yield context, local.copy()
        elif isinstance(statement, ast.If):
            yield from _raising_conditions(
                statement.body, context=(*context, statement.test), aliases=local
            )
            yield from _raising_conditions(
                statement.orelse,
                context=(*context, ast.UnaryOp(op=ast.Not(), operand=statement.test)),
                aliases=local,
            )
        elif isinstance(statement, ast.While):
            yield from _raising_conditions(
                statement.body, context=(*context, statement.test), aliases=local
            )
            yield from _raising_conditions(statement.orelse, context=context, aliases=local)
        elif isinstance(
            statement, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.TryStar)
        ):
            yield from _raising_conditions(statement.body, context=context, aliases=local)
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.Try, ast.TryStar)):
                yield from _raising_conditions(statement.orelse, context=context, aliases=local)
            if isinstance(statement, (ast.Try, ast.TryStar)):
                for handler in statement.handlers:
                    yield from _raising_conditions(handler.body, context=context, aliases=local)
                yield from _raising_conditions(statement.finalbody, context=context, aliases=local)


def _condition_field(node: ast.AST) -> str | None:
    owners = {"self", "cls", "values", "data", "model"}
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in owners
    ):
        return node.attr
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in owners
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in owners
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _presence_atoms(
    conditions: Sequence[ast.expr], aliases: dict[str, ast.expr]
) -> tuple[set[str], set[str], bool]:
    """Separate presence tests from value comparisons, expanding one local alias level."""
    presence: set[str] = set()
    values: set[str] = set()
    positive_guards = True

    def value_fields(node: ast.expr, *, expand: bool) -> None:
        for child in ast.walk(node):
            if (field := _condition_field(child)) is not None:
                values.add(field)
            elif expand and isinstance(child, ast.Name) and child.id in aliases:
                value_fields(aliases[child.id], expand=False)

    def collect(node: ast.expr, *, expand: bool = True) -> None:
        nonlocal positive_guards
        if expand and isinstance(node, ast.Name) and node.id in aliases:
            collect(aliases[node.id], expand=False)
        elif isinstance(node, ast.IfExp):
            positive_guards = False
            collect(node.test, expand=expand)
            collect(node.body, expand=expand)
            collect(node.orelse, expand=expand)
        elif isinstance(node, ast.BoolOp):
            positive_guards &= isinstance(node.op, ast.And)
            for child in node.values:
                collect(child, expand=expand)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            positive_guards = False
            collect(node.operand, expand=expand)
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and any(
                isinstance(side, ast.Constant) and side.value is None for side in sides
            ):
                positive_guards &= all(isinstance(op, ast.IsNot) for op in node.ops)
                for side in sides:
                    collect(side, expand=expand)
            else:
                for side in sides:
                    resolved = (
                        aliases.get(side.id, side)
                        if expand and isinstance(side, ast.Name)
                        else side
                    )
                    if (
                        isinstance(resolved, (ast.Compare, ast.BoolOp))
                        or (
                            isinstance(resolved, ast.Call)
                            and _terminal_name(resolved.func) == "bool"
                        )
                        or (
                            isinstance(resolved, ast.Call)
                            and _terminal_name(resolved.func) == "len"
                            and len(sides) == 2
                            and any(
                                isinstance(other, ast.Constant) and other.value == 0
                                for other in sides
                            )
                        )
                    ):
                        positive_guards = False
                        collect(resolved, expand=False)
                    else:
                        value_fields(side, expand=expand)
        elif (
            isinstance(node, ast.Call)
            and _terminal_name(node.func) in {"bool", "len"}
            and node.args
        ):
            collect(node.args[0], expand=expand)
        elif (field := _condition_field(node)) is not None:
            presence.add(field)

    for condition in conditions:
        collect(condition)
    return presence, values, positive_guards


def _tag_like_field(
    field: FieldSpec, owner: PythonDefinition, symbols: dict[str, PythonDefinition]
) -> bool:
    for node in ast.walk(_annotation_ast(field.annotation)):
        if isinstance(node, ast.Subscript) and _terminal_name(node.value) == "Literal":
            return True
        if isinstance(node, ast.Name) and node.id == "bool":
            return True
        if isinstance(node, (ast.Name, ast.Attribute)):
            resolved = _resolve_symbol(node, owner, symbols)
            if resolved is not None and resolved.vocabulary_kind in {"alias", "enum"}:
                return True
    return False


def _presence_candidates(
    classes: Sequence[PythonDefinition],
    *,
    symbols: dict[str, PythonDefinition],
    ancestors: dict[tuple[str, int, str], set[tuple[str, int, str]]],
    selection: SourceSelection,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for owner in classes:
        inherited = [cls for cls in classes if cls.identity in ancestors[owner.identity]]
        fields: dict[str, FieldSpec] = {}
        field_owners: dict[str, PythonDefinition] = {}
        for cls in sorted(inherited, key=lambda cls: len(ancestors[cls.identity])):
            fields.update((field.name, field) for field in cls.declared_fields)
            field_owners.update((field.name, cls) for field in cls.declared_fields)
        fields.update((field.name, field) for field in owner.declared_fields)
        field_owners.update((field.name, owner) for field in owner.declared_fields)
        optional = {name for name, field in fields.items() if field.optional}
        tags = {
            name
            for name, field in fields.items()
            if _tag_like_field(field, field_owners[name], symbols)
        }
        findings: list[str] = []
        methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for method in _validator_methods(owner):
            aliases: dict[str, ast.expr] = {}
            for decorator in method.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and _terminal_name(decorator.func) == "field_validator"
                ):
                    targets = [
                        arg.value
                        for arg in decorator.args
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    ]
                    arguments = [arg for arg in method.args.args if arg.arg not in {"self", "cls"}]
                    if len(targets) == 1 and arguments:
                        aliases[arguments[0].arg] = ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr=targets[0],
                            ctx=ast.Load(),
                        )
            reasons: set[str] = set()
            for conditions, local in _raising_conditions(method.body, aliases=aliases):
                presence, values, guarded = _presence_atoms(conditions, local)
                tested = presence & optional
                tested_tags = values & tags
                if not tested or (len(tested) < 2 and not tested_tags):
                    continue
                if guarded and presence <= values:
                    continue
                reasons.add(
                    f"optional presence [{', '.join(sorted(tested))}]"
                    + (
                        f" with tag values [{', '.join(sorted(tested_tags))}]"
                        if tested_tags
                        else ""
                    )
                )
            if reasons:
                methods.append(method)
                findings.append(f"{method.name}: {'; '.join(sorted(reasons))}")
        if not methods:
            continue
        descendants: list[PythonDefinition] = []
        inheritance: list[str] = []
        for cls in classes:
            if owner.identity not in ancestors[cls.identity]:
                continue
            inherited_methods = [
                method.name
                for method in methods
                if method.name not in cls.methods
                and not any(
                    method.name in between.methods
                    and between.identity in ancestors[cls.identity]
                    and owner.identity in ancestors[between.identity]
                    for between in classes
                )
            ]
            if inherited_methods:
                descendants.append(cls)
                inheritance.append(f"{cls.qualname} [{', '.join(inherited_methods)}]")
        sites = [
            Location(
                owner.path,
                method.lineno,
                method.end_lineno or method.lineno,
                f"{owner.qualname}.{method.name} (validator owner)",
            )
            for method in methods
        ]
        sites.extend(
            replace(_definition_location(cls), label=f"{cls.qualname} (inherited fields)")
            for cls in inherited
        )
        _record_type_candidate(
            candidates,
            _type_candidate(
                "presence dependency",
                (owner, *descendants),
                selection=selection,
                score=0.91,
                reason=f"{owner.qualname} owns raising presence-dependency evidence: {'; '.join(findings)}; inherited by: {'; '.join(inheritance) or 'none'}; relational/value checks remain review distinctions",
                sites=sites,
                site_fingerprints=tuple(
                    ast.dump(cast("ast.AST", cls.node), include_attributes=False)
                    for cls in inherited
                ),
            ),
        )
    return candidates


def _vocabulary_candidates(
    definitions: Sequence[PythonDefinition], *, selection: SourceSelection
) -> list[Candidate]:
    groups: dict[frozenset[tuple[str, str]], list[PythonDefinition]] = {}
    for definition in definitions:
        if len(definition.vocabulary) >= 2:
            groups.setdefault(definition.vocabulary, []).append(definition)
    candidates: list[Candidate] = []
    for values, members in groups.items():
        if len(members) < 2 or all(member.vocabulary_kind == "enum" for member in members):
            continue
        kinds = Counter(member.vocabulary_kind for member in members)
        alias_copy = bool(kinds["alias"] and kinds["inline"])
        _record_type_candidate(
            candidates,
            _type_candidate(
                "closed vocabulary",
                members,
                selection=selection,
                score=0.99 if alias_copy else 0.89,
                reason=f"exact {'inline copy of named alias' if alias_copy else 'repeated value set'} [{', '.join(value for _, value in sorted(values))}] in {len(members)} places ({', '.join(f'{kind}: {count}' for kind, count in sorted(kinds.items()))}); equal values do not establish a shared semantic owner"
                + ("; enum declarations retain nominal behavior" if kinds["enum"] else ""),
            ),
        )
    named = {
        values: [member for member in members if member.vocabulary_kind in {"alias", "enum"}]
        for values, members in groups.items()
    }
    for smaller, smaller_members in named.items():
        if not smaller_members:
            continue
        for larger, larger_members in named.items():
            if not larger_members or not smaller < larger:
                continue
            _record_type_candidate(
                candidates,
                _type_candidate(
                    "closed vocabulary",
                    (*smaller_members, *larger_members),
                    selection=selection,
                    score=0.93,
                    reason=f"named proper subset: {', '.join(member.qualname for member in smaller_members)} [{', '.join(value for _, value in sorted(smaller))}] < {', '.join(member.qualname for member in larger_members)}; extra values [{', '.join(value for _, value in sorted(larger - smaller))}]; ownership/dependency direction requires review",
                ),
            )
    return candidates


def _class_candidate(
    first: PythonDefinition,
    second: PythonDefinition,
    *,
    deep: bool,
    selection: SourceSelection,
) -> Candidate | None:
    first_fields = {field.name: field for field in first.fields}
    second_fields = {field.name: field for field in second.fields}
    shared_names = set(first_fields) & set(second_fields)
    minimum_shared = 2 if deep else 3
    if len(shared_names) < minimum_shared:
        return None
    if first.name in second.bases or second.name in first.bases:
        return None

    names_union = set(first_fields) | set(second_fields)
    field_jaccard = len(shared_names) / len(names_union)
    containment = len(shared_names) / min(len(first_fields), len(second_fields))
    if containment < (0.65 if deep else 0.75):
        return None
    annotation_matches = sum(
        first_fields[name].annotation == second_fields[name].annotation for name in shared_names
    )
    default_matches = sum(
        first_fields[name].default == second_fields[name].default for name in shared_names
    )
    annotation_score = annotation_matches / len(shared_names)
    default_score = default_matches / len(shared_names)
    method_score = _jaccard(first.methods, second.methods)
    name_score = _jaccard(_name_words(first.name), _name_words(second.name))
    score = (
        0.50 * field_jaccard
        + 0.15 * containment
        + 0.15 * annotation_score
        + 0.10 * method_score
        + 0.05 * default_score
        + 0.05 * name_score
    )
    if first.path == second.path and first.bases & second.bases:
        score -= 0.08
    threshold = DEEP_CLASS_THRESHOLD if deep else DEFAULT_CLASS_THRESHOLD
    if score < threshold:
        return None

    location_first, location_second = _ordered_locations(first, second)
    selected_by_identity = {
        (first.path, first.start, first.end, first.qualname): selection.includes(
            first.path, first.start, first.end
        ),
        (second.path, second.start, second.end, second.qualname): selection.includes(
            second.path, second.start, second.end
        ),
    }
    return Candidate(
        category="class schema",
        score=min(1.0, score),
        first=location_first,
        second=location_second,
        reason=(
            f"{len(shared_names)}/{len(names_union)} shared fields; "
            f"{annotation_matches} matching annotations; {default_matches} matching defaults"
        ),
        first_selected=selected_by_identity[location_first.identity],
        second_selected=selected_by_identity[location_second.identity],
        review_fingerprint=_candidate_review_fingerprint(first, second),
    )


def _function_candidate(
    first: PythonDefinition,
    second: PythonDefinition,
    *,
    deep: bool,
    selection: SourceSelection,
) -> Candidate | None:
    minimum_nodes = DEEP_MIN_FUNCTION_NODES if deep else DEFAULT_MIN_FUNCTION_NODES
    if first.node_count < minimum_nodes or second.node_count < minimum_nodes:
        return None
    if first.path == second.path and (
        (first.start <= second.start and second.end <= first.end)
        or (second.start <= first.start and first.end <= second.end)
    ):
        return None

    exact_structure = first.fingerprint == second.fingerprint
    size_ratio = min(first.node_count, second.node_count) / max(first.node_count, second.node_count)
    if not exact_structure and size_ratio < (0.5 if deep else 0.6):
        return None

    first_calls = first.calls - _UNINFORMATIVE_CALLS
    second_calls = second.calls - _UNINFORMATIVE_CALLS
    shared_calls = first_calls & second_calls

    if exact_structure:
        score = 0.98
        reason = "alpha-normalized AST match"
    else:
        if not shared_calls:
            return None
        name_score = _jaccard(_name_words(first.name), _name_words(second.name))
        if len(shared_calls) < 2 and name_score < 0.5:
            return None
        call_score = _jaccard(first_calls, second_calls)
        shape_score = _cosine(first.node_counts, second.node_counts)
        control_score = _cosine(first.control_counts, second.control_counts)
        signature_score = float(first.signature == second.signature)
        score = (
            0.40 * call_score
            + 0.25 * shape_score
            + 0.15 * control_score
            + 0.10 * name_score
            + 0.10 * signature_score
        )
        reason = (
            f"AST shape {shape_score:.2f}; control flow {control_score:.2f}; "
            f"call overlap {call_score:.2f}"
        )

    sibling_methods = (
        first.is_method
        and second.is_method
        and first.name == second.name
        and first.parent_class != second.parent_class
    )
    if sibling_methods:
        score -= 0.18
    threshold = DEEP_FUNCTION_THRESHOLD if deep else DEFAULT_FUNCTION_THRESHOLD
    if score < threshold:
        return None

    location_first, location_second = _ordered_locations(first, second)
    selected_by_identity = {
        (first.path, first.start, first.end, first.qualname): selection.includes(
            first.path, first.start, first.end
        ),
        (second.path, second.start, second.end, second.qualname): selection.includes(
            second.path, second.start, second.end
        ),
    }
    if shared_calls:
        reason += f"; shared calls: {', '.join(sorted(shared_calls)[:6])}"
    return Candidate(
        category="function behavior",
        score=min(1.0, score),
        first=location_first,
        second=location_second,
        reason=reason,
        first_selected=selected_by_identity[location_first.identity],
        second_selected=selected_by_identity[location_second.identity],
        review_fingerprint=_candidate_review_fingerprint(first, second),
    )


def ast_candidates(
    definitions: Sequence[PythonDefinition],
    *,
    selection: SourceSelection,
    deep: bool,
) -> tuple[Candidate, ...]:
    """Compare selected Python definitions with all repository definitions."""
    selected = [
        definition
        for definition in definitions
        if definition.kind in {"class", "function"}
        and selection.includes(definition.path, definition.start, definition.end)
    ]
    by_kind = {
        "class": [definition for definition in definitions if definition.kind == "class"],
        "function": [definition for definition in definitions if definition.kind == "function"],
    }
    classes = sorted(by_kind["class"], key=lambda cls: cls.identity)
    symbols = _symbol_index(definitions)
    ancestors = _class_ancestors(classes, symbols)
    type_candidates: list[Candidate] = []
    represented: set[frozenset[tuple[str, int, str]]] = set()
    if classes:
        type_candidates, represented = _field_candidates(
            classes, definitions, symbols=symbols, ancestors=ancestors, selection=selection
        )
        type_candidates.extend(
            _presence_candidates(classes, symbols=symbols, ancestors=ancestors, selection=selection)
        )
    type_candidates.extend(_vocabulary_candidates(definitions, selection=selection))
    exact_function_anchors: dict[str, tuple[str, int, str]] = {}
    for definition in by_kind["function"]:
        anchor = exact_function_anchors.get(definition.fingerprint)
        if anchor is None or definition.identity < anchor:
            exact_function_anchors[definition.fingerprint] = definition.identity
    candidates: dict[
        tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]], Candidate
    ] = {candidate.identity: candidate for candidate in type_candidates}
    seen_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for first in selected:
        for second in by_kind[first.kind]:
            if first.identity == second.identity:
                continue
            if (
                first.kind == "function"
                and first.fingerprint == second.fingerprint
                and exact_function_anchors[first.fingerprint]
                not in {first.identity, second.identity}
            ):
                # Report alpha-equivalent groups as a star around one stable
                # representative instead of emitting every O(n²) pair.
                continue
            pair = tuple(sorted((first.identity, second.identity)))
            typed_pair = cast(
                "tuple[tuple[str, int, str], tuple[str, int, str]]",
                pair,
            )
            if typed_pair in seen_pairs:
                continue
            seen_pairs.add(typed_pair)
            if first.kind == "class":
                if frozenset(typed_pair) in represented or _related_classes(
                    first, second, ancestors
                ):
                    continue
                candidate = _class_candidate(
                    first,
                    second,
                    deep=deep,
                    selection=selection,
                )
            else:
                candidate = _function_candidate(
                    first,
                    second,
                    deep=deep,
                    selection=selection,
                )
            if candidate is not None:
                candidates[candidate.identity] = candidate
    return tuple(sorted(candidates.values(), key=_candidate_sort_key))


def _report_path(
    repo_root: Path,
    *,
    name: str,
    format_name: str,
) -> str:
    raw = Path(name)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise DuplicateAuditError(
                f"jscpd reported a path outside the repository: {name}"
            ) from exc
    roots = (
        PYTHON_SOURCE_ROOTS
        if format_name == "python"
        else tuple(root for root in SOURCE_ROOTS if root not in PYTHON_SOURCE_ROOTS)
    )
    matches = [root / raw for root in roots if (repo_root / root / raw).exists()]
    if len(matches) != 1:
        raise DuplicateAuditError(
            f"cannot uniquely resolve jscpd path {name!r} ({format_name}); matches={matches}"
        )
    return matches[0].as_posix()


def _report_location(
    repo_root: Path,
    raw: object,
    *,
    format_name: str,
) -> Location:
    if not isinstance(raw, dict):
        raise DuplicateAuditError("jscpd report contains a non-object file location")
    name = raw.get("name")
    start = raw.get("start")
    end = raw.get("end")
    if not isinstance(name, str) or not isinstance(start, int) or not isinstance(end, int):
        raise DuplicateAuditError("jscpd report contains an invalid file location")
    return Location(
        path=_report_path(repo_root, name=name, format_name=format_name),
        start=start,
        end=end,
    )


def _python_range_is_import_scaffolding(repo_root: Path, location: Location) -> bool:
    """Return whether a clone range contains only module imports or its docstring."""
    try:
        tree = ast.parse((repo_root / location.path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False

    overlapping = [
        statement
        for statement in tree.body
        if statement.lineno <= location.end
        and (statement.end_lineno or statement.lineno) >= location.start
    ]
    if not overlapping:
        return False

    return all(
        isinstance(statement, (ast.Import, ast.ImportFrom))
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        for statement in overlapping
    )


def parse_jscpd_report(
    report_path: Path,
    *,
    repo_root: Path,
    selection: SourceSelection,
) -> tuple[Candidate, ...]:
    """Parse and selection-filter a jscpd JSON report."""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DuplicateAuditError(f"cannot read jscpd report {report_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("duplicates"), list):
        raise DuplicateAuditError("jscpd report has no duplicates array")

    candidates: list[Candidate] = []
    for raw_duplicate in payload["duplicates"]:
        if not isinstance(raw_duplicate, dict):
            raise DuplicateAuditError("jscpd report contains a non-object duplicate")
        format_name = raw_duplicate.get("format")
        lines = raw_duplicate.get("lines")
        tokens = raw_duplicate.get("tokens")
        if (
            not isinstance(format_name, str)
            or not isinstance(lines, int)
            or not isinstance(tokens, int)
        ):
            raise DuplicateAuditError("jscpd report contains invalid duplicate metadata")
        first = _report_location(
            repo_root,
            raw_duplicate.get("firstFile"),
            format_name=format_name,
        )
        second = _report_location(
            repo_root,
            raw_duplicate.get("secondFile"),
            format_name=format_name,
        )
        if (
            format_name == "python"
            and _python_range_is_import_scaffolding(repo_root, first)
            and _python_range_is_import_scaffolding(repo_root, second)
        ):
            continue
        first_selected = selection.includes(first.path, first.start, first.end)
        second_selected = selection.includes(second.path, second.start, second.end)
        if not (first_selected or second_selected):
            continue
        score = min(0.99, 0.76 + min(tokens, 240) / 1200 + min(lines, 30) / 300)
        candidates.append(
            Candidate(
                category="token clone",
                score=score,
                first=first,
                second=second,
                reason=f"{lines} duplicated lines; {tokens} duplicated tokens ({format_name})",
                first_selected=first_selected,
                second_selected=second_selected,
            )
        )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def run_jscpd(
    repo_root: Path,
    *,
    selection: SourceSelection,
    deep: bool,
) -> tuple[Candidate, ...]:
    """Run the repository-local jscpd and return selected clone pairs."""
    minimum_tokens = 45 if deep else 65
    minimum_lines = 5 if deep else 7
    with tempfile.TemporaryDirectory(prefix="duplicate-audit-") as output_dir_text:
        output_dir = Path(output_dir_text)
        command = (
            "bunx",
            "jscpd",
            *(root.as_posix() for root in SOURCE_ROOTS),
            "--min-tokens",
            str(minimum_tokens),
            "--min-lines",
            str(minimum_lines),
            "--mode",
            "weak",
            "--format",
            "python,typescript,tsx",
            "--ignore",
            "**/generated/**,**/*.test.*,**/*.spec.*,**/*.stories.*",
            "--reporters",
            "json",
            "--output",
            output_dir_text,
            "--no-colors",
            "--no-tips",
        )
        _run_command(command, cwd=repo_root)
        return parse_jscpd_report(
            output_dir / "jscpd-report.json",
            repo_root=repo_root,
            selection=selection,
        )


def _ranges_overlap(first: Location, second: Location) -> bool:
    return first.path == second.path and first.start <= second.end and second.start <= first.end


def _covered_by_token_clone(candidate: Candidate, token_clones: Sequence[Candidate]) -> bool:
    for clone in token_clones:
        if (
            _ranges_overlap(candidate.first, clone.first)
            and _ranges_overlap(candidate.second, clone.second)
        ) or (
            _ranges_overlap(candidate.first, clone.second)
            and _ranges_overlap(candidate.second, clone.first)
        ):
            return True
    return False


def _covered_by_type_candidate(
    clone: Candidate,
    type_candidates: Sequence[Candidate],
    definitions: Sequence[PythonDefinition],
) -> bool:
    """A class pair already represented by field evidence needs no weaker clone report."""
    class_locations = {
        _definition_location(definition).identity
        for definition in definitions
        if definition.kind == "class"
    }
    for candidate in type_candidates:
        if candidate.category not in {"identical fields", "repeated row", "generic family"}:
            continue
        members = [
            location
            for location in (candidate.first, candidate.second, *candidate.evidence_locations)
            if location.identity in class_locations
        ]
        pairs = (
            ((members[0], member) for member in members[1:])
            if candidate.category == "repeated row"
            else combinations(members, 2)
        )
        for first, second in pairs:
            if (_ranges_overlap(clone.first, first) and _ranges_overlap(clone.second, second)) or (
                _ranges_overlap(clone.first, second) and _ranges_overlap(clone.second, first)
            ):
                return True
    return False


def _candidate_sort_key(candidate: Candidate) -> tuple[float, str, tuple[str, int, int, str]]:
    return (-candidate.score, candidate.category, candidate.first.identity)


def select_candidates(candidates: Sequence[Candidate], *, limit: int) -> tuple[Candidate, ...]:
    """Cap output while reserving space for every populated candidate category."""
    if limit <= 0:
        return ()
    deduplicated = {candidate.identity: candidate for candidate in candidates}
    by_category: dict[str, list[Candidate]] = {}
    for candidate in sorted(deduplicated.values(), key=_candidate_sort_key):
        by_category.setdefault(candidate.category, []).append(candidate)
    if len(deduplicated) <= limit:
        return tuple(sorted(deduplicated.values(), key=_candidate_sort_key))
    if limit < len(by_category):
        return tuple(sorted(deduplicated.values(), key=_candidate_sort_key)[:limit])

    quota = max(1, min(5, limit // max(1, len(by_category))))
    selected: dict[tuple[str, tuple[str, int, int, str], tuple[str, int, int, str]], Candidate] = {}
    for group in by_category.values():
        for candidate in group[:quota]:
            selected[candidate.identity] = candidate
    remainder = [
        candidate for candidate in deduplicated.values() if candidate.identity not in selected
    ]
    for candidate in sorted(remainder, key=_candidate_sort_key):
        if len(selected) >= limit:
            break
        selected[candidate.identity] = candidate
    return tuple(sorted(selected.values(), key=_candidate_sort_key))


def load_reviewed_pairs(path: Path) -> tuple[ReviewedPair, ...]:
    """Load stable semantic-pair classifications from the checked-in registry."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DuplicateAuditError(
            f"cannot load reviewed duplicate pairs from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        raise DuplicateAuditError(f"{path} must contain a top-level 'reviews' list")

    reviews: list[ReviewedPair] = []
    identities: set[tuple[str, str, str]] = set()
    for index, record in enumerate(payload["reviews"]):
        if not isinstance(record, dict):
            raise DuplicateAuditError(f"{path}: review {index} must be an object")
        values = tuple(
            record.get(field)
            for field in (
                "category",
                "first",
                "second",
                "fingerprint",
                "classification",
                "rationale",
            )
        )
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise DuplicateAuditError(
                f"{path}: review {index} requires non-empty category, first, second, "
                "fingerprint, classification, and rationale strings"
            )
        category, first, second, fingerprint, classification, rationale = cast(
            "tuple[str, str, str, str, str, str]", values
        )
        if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise DuplicateAuditError(
                f"{path}: review {index} fingerprint must be a lowercase SHA-256 digest"
            )
        review = ReviewedPair(category, first, second, fingerprint, classification, rationale)
        if review.identity in identities:
            raise DuplicateAuditError(f"{path}: duplicate reviewed pair at index {index}")
        identities.add(review.identity)
        reviews.append(review)
    return tuple(reviews)


def partition_reviewed_candidates(
    candidates: Sequence[Candidate],
    reviews: Sequence[ReviewedPair],
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...]]:
    """Separate reviewed semantic definitions from candidates needing attention."""
    reviewed_fingerprints = {review.identity: review.fingerprint for review in reviews}
    pending: list[Candidate] = []
    reviewed: list[Candidate] = []
    for candidate in candidates:
        first = f"{candidate.first.path}::{candidate.first.label}"
        second = f"{candidate.second.path}::{candidate.second.label}"
        identity = (candidate.category, *sorted((first, second)))
        target = (
            reviewed
            if reviewed_fingerprints.get(identity) == candidate.review_fingerprint
            else pending
        )
        target.append(candidate)
    return tuple(pending), tuple(reviewed)


def _format_location(location: Location, *, selected: bool, mark_selected: bool) -> str:
    marker = "*" if selected and mark_selected else " "
    line = (
        str(location.start)
        if location.start == location.end
        else f"{location.start}-{location.end}"
    )
    label = f" {location.label}" if location.label else ""
    return f"  {marker} {location.path}:{line}{label}"


def render_report(
    *,
    selection: SourceSelection,
    candidates: Sequence[Candidate],
    shown: Sequence[Candidate],
    definitions: Sequence[PythonDefinition],
    reviewed_count: int = 0,
) -> str:
    """Render concise output intended for a coding agent."""
    classes = sum(definition.kind == "class" for definition in definitions)
    functions = sum(definition.kind == "function" for definition in definitions)
    lines = [
        "Duplicate audit (advisory)",
        f"Scope: {selection.description}",
        f"Indexed: {classes} Python classes; {functions} Python functions/methods",
    ]
    if reviewed_count:
        lines.append(
            f"Suppressed {reviewed_count} reviewed pair(s); pass --include-reviewed to show them."
        )
    if not candidates:
        lines.append("No likely duplicates touched the selected code.")
        return "\n".join(lines)

    lines.append(f"Showing {len(shown)} of {len(candidates)} candidate pair(s).")
    if not selection.is_all:
        lines.append("* marks a definition, evidence site or clone range in the selected code.")
    lines.append("Type categories show syntactic evidence, not contract or semantic equivalence.")
    category_titles = {
        "identical fields": "Identical declared field signatures",
        "repeated row": "Repeated rows with conversion witnesses",
        "generic family": "Generic families (annotation holes)",
        "presence dependency": "Presence-dependency evidence by owning declaration",
        "closed vocabulary": "Repeated closed vocabularies / named subsets",
        "token clone": "Token/near-copy clones",
        "class schema": "Class/schema overlaps",
        "function behavior": "Function behavior overlaps",
    }
    for category in (*TYPE_CATEGORIES, "token clone", "class schema", "function behavior"):
        group = [candidate for candidate in shown if candidate.category == category]
        total = sum(candidate.category == category for candidate in candidates)
        if not total:
            continue
        lines.extend(
            ("", f"{category_titles[category]} ({len(group)} shown / {total} candidate(s))")
        )
        for candidate in group:
            lines.append(
                candidate.reason
                if category in TYPE_CATEGORIES
                else f"{candidate.level} {candidate.score:.0%} — {candidate.reason}"
            )
            lines.append(
                _format_location(
                    candidate.first,
                    selected=candidate.first_selected,
                    mark_selected=not selection.is_all,
                )
            )
            lines.append(
                _format_location(
                    candidate.second,
                    selected=candidate.second_selected,
                    mark_selected=not selection.is_all,
                )
            )
            for location in candidate.evidence_locations:
                lines.append(
                    _format_location(
                        location,
                        selected=selection.includes(location.path, location.start, location.end),
                        mark_selected=not selection.is_all,
                    )
                )
    lines.extend(
        (
            "",
            "Review each pair as: consolidate, intentional mirror/parity check, "
            "related implementation, or false positive.",
        )
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="source files/directories to compare against the repository instead of using git diff",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="audit every pair in the production source tree",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="lower similarity and clone-size thresholds",
    )
    parser.add_argument(
        "--base",
        default="origin/master",
        help="git ref used to find the default comparison merge base",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=f"maximum pairs to show (default {DEFAULT_LIMIT}, deep {DEEP_LIMIT})",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="show pairs classified in scripts/checks/duplicate_reviews.json",
    )
    args = parser.parse_args(argv)
    if args.all and args.targets:
        parser.error("--all cannot be combined with explicit targets")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.all:
            selection = SourceSelection(ranges=None, description="all production source")
        elif args.targets:
            selection = explicit_selection(REPO_ROOT, cast("Sequence[str]", args.targets))
        else:
            selection = changed_selection(REPO_ROOT, base_ref=cast("str", args.base))
        if not selection.is_all and not selection.ranges:
            print(
                "Duplicate audit (advisory)\n"
                f"Scope: {selection.description}\n"
                "No changed production source found; pass paths or use --all."
            )
            return 0

        paths = source_paths(REPO_ROOT)
        definitions = collect_python_definitions(REPO_ROOT, paths)
        token_clones = run_jscpd(REPO_ROOT, selection=selection, deep=bool(args.deep))
        semantic = ast_candidates(definitions, selection=selection, deep=bool(args.deep))
        token_clones = tuple(
            clone
            for clone in token_clones
            if not _covered_by_type_candidate(clone, semantic, definitions)
        )
        semantic = tuple(
            candidate
            for candidate in semantic
            if candidate.category != "function behavior"
            or not _covered_by_token_clone(candidate, token_clones)
        )
        candidates = tuple(sorted((*token_clones, *semantic), key=_candidate_sort_key))
        reviewed: tuple[Candidate, ...] = ()
        if not args.include_reviewed:
            candidates, reviewed = partition_reviewed_candidates(
                candidates,
                load_reviewed_pairs(REVIEWED_PAIRS_PATH),
            )
        limit = cast("int | None", args.limit) or (DEEP_LIMIT if args.deep else DEFAULT_LIMIT)
        shown = select_candidates(candidates, limit=limit)
        print(
            render_report(
                selection=selection,
                candidates=candidates,
                shown=shown,
                definitions=definitions,
                reviewed_count=len(reviewed),
            )
        )
        return 0
    except DuplicateAuditError as exc:
        print(f"duplicate audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
