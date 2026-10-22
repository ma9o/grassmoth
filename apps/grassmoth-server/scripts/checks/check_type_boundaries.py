"""Reject type annotations that erase domain types behind ``dict``.

The checker owns project-specific rules that Ruff, ty and basedpyright cannot
express. Explicit ``Any`` belongs to basedpyright's ``reportExplicitAny``.

``CUSTOM002``
    A named domain type must not be unioned with ``dict``. Raw mappings are
    parsed at I/O boundaries; internal code receives the validated type.

``CUSTOM003``
    A parameter must not include ``None`` in its type only to reject ``None``
    on every normally completing path. Make the parameter required and let the
    type checker push absence handling to the boundary where it originates.

``CORE001``
    Construction bypasses and mutable casts are checked by call name. Revisions
    use validated owner construction; frozen-field initialization is constructor-only.
    Dump-spread rebuilds are forbidden throughout every checked Python tree.

``CORE002``
    Owned Value contracts and published compiler/execution outputs expose read-only
    collections, including composed fields, container members, aliases and private
    fields/properties. An attached cache is owned data, not a private builder.

``VIEW001``
    Declared pure projections cannot raise, assert, or revalidate the core.
    Exhaustive ``assert_never`` arms are permitted.

``IMM001``
    Owned contracts inherit the shared frozen Value configuration; compiled
    alternatives are frozen dataclasses. Private builders are not values.

``IMM002``
    Owned compiled dataclasses detach nested mappings and NumPy arrays through
    freeze_fields(self) at the end of __post_init__. Value owns the same work
    in its field constructor. JAX arrays and callable annotations are excluded.

``PARSE001``
    Across every production role, serialized value parsing belongs to its
    transport/storage edge or to the target type's own module.

``ERR001``
    Built-in error catches, including contextlib.suppress, never classify science.
    Only a single expected
    parser/foreign-constructor error around its operation, or a declared shell
    failure/retry handler, can be translated. Every production role is checked.

``ERR002``
    Calls to declared compiler/execution sum outcomes cannot be discarded as a
    statement or assigned to _. Consume or forward the result so its alternatives
    reach the caller's type checking. This resolves named functions and imports,
    not arbitrary callbacks or data flow after assignment.

"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import override

from scripts.checks.architecture_roles import (
    PACKAGE,
    PACKAGES,
    SOURCE_ROOTS,
    fix_owner,
    projection_checks,
    role_for_path,
    role_inventory,
)
from scripts.checks.check_architecture_boundaries import _scope_bindings

_DOMAIN_DICT_UNION = "CUSTOM002"
_REJECT_ONLY_OPTIONAL_PARAMETER = "CUSTOM003"
_CORE_BYPASS = "CORE001"
_MUTABLE_CORE = "CORE002"
_PARTIAL_VIEW = "VIEW001"
_FROZEN_VALUE = "IMM001"
_OWNED_COLLECTIONS = "IMM002"
_BUILTIN_CATCH = "ERR001"
_DISCARDED_OUTCOME = "ERR002"
_OWNER_PARSE = "PARSE001"
_ALL_RULES = frozenset(
    {
        _DOMAIN_DICT_UNION,
        _REJECT_ONLY_OPTIONAL_PARAMETER,
        _CORE_BYPASS,
        _MUTABLE_CORE,
        _PARTIAL_VIEW,
        _FROZEN_VALUE,
        _OWNED_COLLECTIONS,
        _BUILTIN_CATCH,
        _DISCARDED_OUTCOME,
        _OWNER_PARSE,
    }
)
_DOMAIN_TYPE_SUFFIXES = (
    "Artifact",
    "Contract",
    "Design",
    "Model",
    "Plan",
    "Proposal",
    "Report",
    "Result",
    "Spec",
    "Structure",
)
_MODEL_BASE_NAMES = frozenset({"BaseModel", "NamedTuple", "Protocol", "TypedDict"})


@dataclass(frozen=True)
class _TypingBindings:
    """Names through which one module refers to special typing forms."""

    optional_names: frozenset[str]
    union_names: frozenset[str]
    module_names: frozenset[str]


@dataclass(frozen=True)
class Violation:
    """One source-level type-boundary violation."""

    path: str
    line: int
    column: int
    code: str
    scope: str
    target: str
    annotation: str
    message: str

    def diagnostic(self) -> str:
        """Render in the concise format understood by editors and CI."""
        return (
            f"{self.path}:{self.line}:{self.column}: "
            f"{self.code} {self.message} [{self.scope} {self.target}; "
            f"role={role_for_path(self.path) or 'auxiliary'}; fix at {fix_owner(self.path)}]"
        )


def _discover_typing_bindings(tree: ast.Module) -> _TypingBindings:
    """Resolve common direct and aliased imports from typing modules."""
    names = {
        "Optional": {"Optional"},
        "Union": {"Union"},
    }
    module_names = {"typing", "typing_extensions"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"typing", "typing_extensions"}:
                    module_names.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "typing",
            "typing_extensions",
        }:
            for imported in node.names:
                if imported.name in names:
                    names[imported.name].add(imported.asname or imported.name)
    return _TypingBindings(
        optional_names=frozenset(names["Optional"]),
        union_names=frozenset(names["Union"]),
        module_names=frozenset(module_names),
    )


def _is_typing_form(
    node: ast.expr,
    *,
    direct_names: frozenset[str],
    attribute_name: str,
    bindings: _TypingBindings,
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in direct_names
    return bool(
        isinstance(node, ast.Attribute)
        and node.attr == attribute_name
        and isinstance(node.value, ast.Name)
        and node.value.id in bindings.module_names
    )


def _direct_union_members(
    node: ast.expr,
    bindings: _TypingBindings,
) -> list[ast.expr] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return [node.left, node.right]
    if not isinstance(node, ast.Subscript):
        return None
    if _is_typing_form(
        node.value,
        direct_names=bindings.union_names,
        attribute_name="Union",
        bindings=bindings,
    ):
        if isinstance(node.slice, ast.Tuple):
            return list(node.slice.elts)
        return [node.slice]
    if _is_typing_form(
        node.value,
        direct_names=bindings.optional_names,
        attribute_name="Optional",
        bindings=bindings,
    ):
        return [node.slice, ast.Constant(value=None)]
    return None


def _flatten_union(
    node: ast.expr,
    bindings: _TypingBindings,
) -> list[ast.expr]:
    direct_members = _direct_union_members(node, bindings)
    if direct_members is None:
        return [node]
    return [
        flattened for member in direct_members for flattened in _flatten_union(member, bindings)
    ]


def _is_none_member(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_optional_annotation(
    node: ast.expr,
    bindings: _TypingBindings,
) -> bool:
    direct_members = _direct_union_members(node, bindings)
    return bool(
        direct_members and any(_is_none_member(member) for member in _flatten_union(node, bindings))
    )


def _directly_rejected_none_parameter(statement: ast.stmt) -> str | None:
    """Return the parameter rejected by an exact ``if x is None: raise`` guard."""
    if (
        not isinstance(statement, ast.If)
        or statement.orelse
        or len(statement.body) != 1
        or not isinstance(statement.body[0], ast.Raise)
    ):
        return None

    test = statement.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Is)
        or len(test.comparators) != 1
    ):
        return None

    left, right = test.left, test.comparators[0]
    if isinstance(left, ast.Name) and _is_none_member(right):
        return left.id
    if _is_none_member(left) and isinstance(right, ast.Name):
        return right.id
    return None


def _body_without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _executed_nodes(node: ast.AST):
    """Yield nodes executed by this scope without descending into nested scopes."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _executed_nodes(child)


def _prefix_can_accept_none(
    statements: list[ast.stmt],
    parameter: str,
) -> bool:
    """Whether a prefix may finish normally or replace the incoming parameter."""
    for statement in statements:
        for node in _executed_nodes(statement):
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                return True
            if (
                isinstance(node, ast.Name)
                and node.id == parameter
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return True
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == parameter
            ):
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".", maxsplit=1)[0]) == parameter
                for alias in node.names
            ):
                return True
            if isinstance(node, ast.ExceptHandler) and node.name == parameter:
                return True
    return False


def _dominating_none_rejections(body: list[ast.stmt]) -> frozenset[str]:
    """Find top-level None-rejection guards reached before normal completion."""
    statements = _body_without_docstring(body)
    rejected: set[str] = set()
    for index, statement in enumerate(statements):
        parameter = _directly_rejected_none_parameter(statement)
        if parameter is not None and not _prefix_can_accept_none(statements[:index], parameter):
            rejected.add(parameter)
    return frozenset(rejected)


def _union_nodes(node: ast.AST, bindings: _TypingBindings):
    """Yield maximal PEP 604, Union, and Optional expressions."""
    if isinstance(node, ast.expr) and _direct_union_members(node, bindings) is not None:
        yield node
        for member in _flatten_union(node, bindings):
            for child in ast.iter_child_nodes(member):
                yield from _union_nodes(child, bindings)
        return
    for child in ast.iter_child_nodes(node):
        yield from _union_nodes(child, bindings)


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _generic_base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return _terminal_name(node)


def _is_dict_member(node: ast.expr) -> bool:
    return _generic_base_name(node) in {"dict", "Dict"}


def _is_named_domain_member(
    node: ast.expr,
    domain_type_names: frozenset[str],
) -> bool:
    name = _generic_base_name(node)
    return bool(name and (name in domain_type_names or name.endswith(_DOMAIN_TYPE_SUFFIXES)))


def _discover_domain_type_names(trees: list[ast.Module]) -> frozenset[str]:
    """Find local model-like classes without importing application modules."""
    bases_by_class: dict[str, set[str]] = {}
    domain_names: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases_by_class.setdefault(node.name, set()).update(
                name for base in node.bases if (name := _generic_base_name(base)) is not None
            )
            if any(
                _generic_base_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
                == "dataclass"
                for decorator in node.decorator_list
            ):
                domain_names.add(node.name)

    changed = True
    while changed:
        changed = False
        known_bases = _MODEL_BASE_NAMES | domain_names
        for class_name, base_names in bases_by_class.items():
            if class_name not in domain_names and base_names & known_bases:
                domain_names.add(class_name)
                changed = True
    return frozenset(domain_names)


class _AnnotationVisitor(ast.NodeVisitor):
    """Check value annotations; recursive JSON aliases legitimately mix named aliases and dict."""

    def __init__(
        self,
        path: str,
        domain_type_names: frozenset[str],
        typing_bindings: _TypingBindings,
        rules: frozenset[str],
    ) -> None:
        self.path = path
        self.domain_type_names = domain_type_names
        self.typing_bindings = typing_bindings
        self.rules = rules
        self.scope: list[str] = []
        self.violations: list[Violation] = []

    @override
    def visit(self, node: ast.AST):
        # This Python 3.12 node is dispatched explicitly so static dead-code
        # analysis can see the reference that NodeVisitor otherwise resolves by name.
        if isinstance(node, ast.AnnAssign):
            self.visit_AnnAssign(node)
            return None
        return super().visit(node)

    def _scope_name(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _check_annotation(self, annotation: ast.expr, *, target: str) -> None:
        for union in _union_nodes(annotation, self.typing_bindings):
            members = _flatten_union(union, self.typing_bindings)
            annotation_text = ast.unparse(union)
            if (
                _DOMAIN_DICT_UNION in self.rules
                and any(_is_dict_member(member) for member in members)
                and any(
                    _is_named_domain_member(member, self.domain_type_names) for member in members
                )
            ):
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=union.lineno,
                        column=union.col_offset + 1,
                        code=_DOMAIN_DICT_UNION,
                        scope=self._scope_name(),
                        target=target,
                        annotation=annotation_text,
                        message=(
                            f"`{annotation_text}` mixes a named domain type with `dict`; "
                            "parse the mapping at the I/O boundary"
                        ),
                    )
                )

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        args = node.args
        parameters = (*args.posonlyargs, *args.args, *args.kwonlyargs)
        for arg in parameters:
            if arg.annotation is not None:
                self._check_annotation(arg.annotation, target=f"parameter:{arg.arg}")
        if args.vararg is not None and args.vararg.annotation is not None:
            self._check_annotation(
                args.vararg.annotation,
                target=f"parameter:*{args.vararg.arg}",
            )
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self._check_annotation(
                args.kwarg.annotation,
                target=f"parameter:**{args.kwarg.arg}",
            )
        if node.returns is not None:
            self._check_annotation(node.returns, target="return")

        if _REJECT_ONLY_OPTIONAL_PARAMETER in self.rules:
            rejected_parameters = _dominating_none_rejections(node.body)
            for rejected_arg in parameters:
                annotation = rejected_arg.annotation
                if (
                    rejected_arg.arg not in rejected_parameters
                    or annotation is None
                    or not _is_optional_annotation(annotation, self.typing_bindings)
                ):
                    continue
                annotation_text = ast.unparse(annotation)
                self.violations.append(
                    Violation(
                        path=self.path,
                        line=annotation.lineno,
                        column=annotation.col_offset + 1,
                        code=_REJECT_ONLY_OPTIONAL_PARAMETER,
                        scope=self._scope_name(),
                        target=f"parameter:{rejected_arg.arg}",
                        annotation=annotation_text,
                        message=(
                            f"`{rejected_arg.arg}: {annotation_text}` admits `None` only to "
                            "reject it before normal completion; remove `None` and handle "
                            "absence at the upstream boundary"
                        ),
                    )
                )

        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(
            node.annotation,
            target=f"variable:{ast.unparse(node.target)}",
        )
        if node.value is not None:
            self.visit(node.value)


class TypeIndex:
    """Resolve class ownership and collection aliases without importing src."""

    def __init__(self, sources: list[tuple[str, ast.Module]]) -> None:
        self.classes: dict[str, tuple[str, ast.ClassDef]] = {}
        self.aliases: dict[str, ast.expr] = {}
        self.adapters: dict[str, ast.expr] = {}
        self.imports: dict[str, dict[str, str]] = {}
        self.functions: dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for path, tree in sources:
            module = self.module(path)
            imports: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
                elif isinstance(node, ast.ImportFrom):
                    parent = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
                    target = (
                        importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
                        if node.level
                        else node.module or ""
                    )
                    for alias in node.names:
                        imports[alias.asname or alias.name] = target + "." + alias.name
            self.imports[module] = imports
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
                    value = node.value
                    if (
                        isinstance(target, ast.Name)
                        and isinstance(value, ast.Call)
                        and self.resolve(value.func, module) == "pydantic.TypeAdapter"
                        and value.args
                    ):
                        self.adapters[module + "." + target.id] = value.args[0]
                if isinstance(node, ast.ClassDef):
                    self.classes[module + "." + node.name] = (path, node)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions[module + "." + node.name] = (path, node)
                elif isinstance(node, ast.TypeAlias):
                    self.aliases[module + "." + node.name.id] = node.value
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript, ast.BinOp))
                ):
                    self.aliases[module + "." + node.targets[0].id] = node.value
        self.evidence: set[str] = set()
        self.compiled_outputs: set[str] = set()
        for path, tree in sources:
            if role_for_path(path) not in {"compiler", "execution"}:
                continue
            module = self.module(path)
            for node in tree.body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not node.name.startswith("_")
                    and node.returns is not None
                ):
                    alternatives = {
                        name
                        for name in self.return_variants(node.returns, module)
                        if name in self.classes
                        and role_for_path(self.classes[name][0]) in {"compiler", "execution"}
                    }
                    sum_alternatives = self.return_variants(node.returns, module, containers=False)
                    if len(sum_alternatives) > 1:
                        self.evidence.update(alternatives.intersection(sum_alternatives))
                    self.compiled_outputs.update(alternatives)

        # Published compiler and execution outputs own their composed numerical values too.
        # Follow actual annotations and aliases, rather than a class-name allowlist.
        self.compiled_values = set(self.compiled_outputs)
        pending = list(self.compiled_values)
        while pending:
            name = pending.pop()
            path, node = self.classes[name]
            for field in node.body:
                if not isinstance(field, ast.AnnAssign):
                    continue
                for expression in ast.walk(_annotation_expr(field.annotation)):
                    if not isinstance(expression, ast.expr):
                        continue
                    for contained in self.return_variants(expression, self.module(path)):
                        if (
                            contained in self.classes
                            and contained not in self.compiled_values
                            and role_for_path(self.classes[contained][0])
                            in {"compiler", "execution", "domain"}
                        ):
                            self.compiled_values.add(contained)
                            pending.append(contained)

    def return_variants(
        self,
        node: ast.expr,
        module: str,
        seen: frozenset[str] = frozenset(),
        *,
        containers: bool = True,
    ) -> set[str]:
        """Follow owned outputs, distinguishing a sum's alternatives from product members."""
        node = _annotation_expr(node)
        if isinstance(node, ast.Constant) and node.value is None:
            return {"builtins.None"}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.return_variants(
                node.left, module, seen, containers=containers
            ) | self.return_variants(node.right, module, seen, containers=containers)
        name = self.resolve(node, module)
        if isinstance(node, ast.Subscript) and name.rsplit(".", 1)[-1] == "Optional":
            return {"builtins.None"} | self.return_variants(
                node.slice, module, seen, containers=containers
            )
        if name in self.aliases and name not in seen:
            return self.return_variants(
                self.aliases[name], name.rpartition(".")[0], seen | {name}, containers=containers
            )
        if (
            isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1] in {"Annotated", "Union"}
            and isinstance(node.slice, ast.Tuple)
        ):
            members = node.slice.elts[:1] if name.endswith("Annotated") else node.slice.elts
            return set().union(
                *(
                    self.return_variants(member, module, seen, containers=containers)
                    for member in members
                )
            )
        if (
            containers
            and isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1]
            in {
                "tuple",
                "Tuple",
                "Mapping",
                "Sequence",
                "frozenset",
                "FrozenSet",
            }
        ):
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            return set().union(
                *(
                    self.return_variants(member, module, seen, containers=containers)
                    for member in members
                )
            )
        if isinstance(node, ast.Subscript) and name in self.classes:
            return {name}
        return {name} if isinstance(node, (ast.Name, ast.Attribute)) else set()

    @staticmethod
    def module(path: str) -> str:
        for package in PACKAGES:
            marker = f"src/{package}/"
            if marker in path:
                relative = path.split(marker, 1)[1]
                return (
                    package
                    + "."
                    + relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
                )
        return PACKAGE + "." + path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")

    def resolve(self, node: ast.expr, module: str) -> str:
        if isinstance(node, ast.Name):
            name = self.imports.get(module, {}).get(node.id, module + "." + node.id)
            seen: set[str] = set()
            while name not in seen:
                seen.add(name)
                parent, _, member = name.rpartition(".")
                target = self.imports.get(parent, {}).get(member)
                if target is None or target == name:
                    break
                name = target
            return name
        if isinstance(node, ast.Attribute):
            return self.resolve(node.value, module) + "." + node.attr
        if isinstance(node, ast.Subscript):
            return self.resolve(node.value, module)
        return ""

    def owned(self, name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name in seen:
            return False
        if name in self.aliases:
            return self.owned(
                self.resolve(self.aliases[name], name.rpartition(".")[0]), seen | {name}
            )
        if name not in self.classes:
            return False
        path, node = self.classes[name]
        if name in self.compiled_values:
            return True
        module = self.module(path)
        bases = [self.resolve(base, module) for base in node.bases]
        return any(
            base == PACKAGE + ".artifacts.base.Value" or self.owned(base, seen | {name})
            for base in bases
        ) or (
            not node.name.startswith("_")
            and role_for_path(path) == "domain"
            and "pydantic.BaseModel" in bases
        )

    def value_contract(self, name: str, seen: frozenset[str] = frozenset()) -> bool:
        """An interpreted domain contract shares the configuration at its owner."""
        if name == PACKAGE + ".artifacts.base.Value":
            return True
        if name not in self.classes or name in seen:
            return False
        path, node = self.classes[name]
        return any(
            self.value_contract(self.resolve(base, self.module(path)), seen | {name})
            for base in node.bases
        )

    def mutable(self, node: ast.expr, module: str, seen: frozenset[str] = frozenset()) -> bool:
        node = _annotation_expr(node)
        name = self.resolve(node, module)
        if name in seen:
            return False
        if name in self.aliases:
            return self.mutable(self.aliases[name], name.rpartition(".")[0], seen | {name})
        if name.rsplit(".", 1)[-1] == "Callable":
            return False
        if name.rsplit(".", 1)[-1] in _MUTABLE_NAMES:
            return True
        return any(
            self.mutable(child, module, seen)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        )

    def needs_ownership(
        self, node: ast.expr, module: str, seen: frozenset[str] = frozenset()
    ) -> bool:
        """Read-only interfaces still need construction-time detachment."""
        node = _annotation_expr(node)
        name = self.resolve(node, module)
        if name in seen:
            return False
        if name in self.aliases:
            return self.needs_ownership(self.aliases[name], name.rpartition(".")[0], seen | {name})
        if name.rsplit(".", 1)[-1] == "Callable":
            return False
        if (
            name in {"numpy.ndarray", "numpy.typing.NDArray"}
            or name.rsplit(".", 1)[-1] == "Mapping"
        ):
            return True
        return any(
            self.needs_ownership(child, module, seen)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        )

    def frozen(self, name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name not in self.classes or name in seen:
            return False
        path, node = self.classes[name]
        for keyword in node.keywords:
            if keyword.arg == "frozen":
                return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for part in (
            *node.decorator_list,
            *(
                value.value
                for value in node.body
                if isinstance(value, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "model_config" for t in value.targets)
            ),
        ):
            if isinstance(part, ast.Call):
                for keyword in part.keywords:
                    if keyword.arg == "frozen":
                        return (
                            isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                        )
        return any(
            self.resolve(base, self.module(path))
            in {
                "equinox.Module",
                "enum.StrEnum",
                "enum.Enum",
                "typing.NamedTuple",
                "typing_extensions.NamedTuple",
                "typing.Protocol",
                "typing_extensions.Protocol",
            }
            or (isinstance(base, ast.Name) and base.id in {"str", "int", "float", "tuple"})
            or self.frozen(self.resolve(base, self.module(path)), seen | {name})
            for base in node.bases
        )


@lru_cache(maxsize=1)
def _production_sources() -> tuple[tuple[str, ast.Module], ...]:
    return tuple(
        (
            "src/" + root.name + "/" + path.relative_to(root).as_posix(),
            ast.parse(path.read_text()),
        )
        for root in SOURCE_ROOTS
        for path in sorted(root.rglob("*.py"))
    )


_CORE_REVALIDATION = frozenset(
    {
        "require_priors",
        "require_measurements",
        "require_execution_structure",
        "check_execution",
        "validate_execution",
        "validate_execution_structure",
        "validate_parameter_anchors",
    }
)
_MUTABLE_NAMES = frozenset(
    {
        "list",
        "dict",
        "set",
        "List",
        "Dict",
        "Set",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
    }
)


def _annotation_expr(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError:
            # Native shape and Literal metadata are string values, not forward types.
            return node
    return node


def _constructor_returns(node: ast.AST) -> bool:
    """An early constructor return can skip field ownership; nested helpers cannot."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    return isinstance(node, ast.Return) or any(
        _constructor_returns(child) for child in ast.iter_child_nodes(node)
    )


class _CoreVisitor(ast.NodeVisitor):
    """Syntactic call-name and core collection checks, without receiver inference."""

    def __init__(
        self, tree: ast.Module, path: str, rules: frozenset[str], index: TypeIndex
    ) -> None:
        self.index = index
        self.module = index.module(path)
        self.path = path
        self.rules = rules
        self.scope: list[str] = []
        self.class_name: str | None = None
        self.in_function = False
        self.local_bindings: list[dict[str, str]] = []
        self.failure_handler = False
        self.names: dict[str, str] = {}
        self.violations: list[Violation] = []
        self.rebuild_dicts: set[int] = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and (
                (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr.startswith("model_validate")
                )
                or self.index.owned(self.index.resolve(call.func, self.module))
            ):
                for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
                    self.rebuild_dicts.update(
                        id(value) for value in ast.walk(argument) if isinstance(value, ast.Dict)
                    )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    self.names[name.asname or name.name] = name.name

    def _name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.names.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _owned_here(self, name: str) -> bool:
        return self.index.classes[name][0] == self.path

    def _mutable(self, node: ast.expr) -> bool:
        return self.index.mutable(node, self.module)

    def _class_owned(self) -> bool:
        return self.class_name is not None and self.index.owned(self.module + "." + self.class_name)

    def _add(
        self, node: ast.expr | ast.stmt | ast.ExceptHandler, code: str, target: str, message: str
    ) -> None:
        if code in self.rules:
            self.violations.append(
                Violation(
                    path=self.path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    code=code,
                    scope=".".join(self.scope) or "<module>",
                    target=target,
                    annotation=ast.unparse(node),
                    message=message,
                )
            )

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.class_name
        self.class_name = node.name
        self.scope.append(node.name)
        qualified = self.module + "." + node.name
        if self._class_owned() and (
            not self.index.frozen(qualified)
            or (
                role_for_path(self.path) == "domain"
                and qualified not in self.index.compiled_values
                and not self.index.value_contract(qualified)
            )
        ):
            self._add(
                node,
                _FROZEN_VALUE,
                node.name,
                "Owned values are frozen; use the shared Value configuration or a frozen compiled dataclass",
            )
        owned_fields = [
            field
            for field in node.body
            if isinstance(field, ast.AnnAssign)
            and self.index.needs_ownership(field.annotation, self.module)
        ]
        if (
            self._class_owned()
            and owned_fields
            and not self.index.value_contract(qualified)
            and any(
                isinstance(decorator, ast.Call)
                and self.index.resolve(decorator.func, self.module) == "dataclasses.dataclass"
                for decorator in node.decorator_list
            )
        ):
            constructor = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "__post_init__"
                ),
                None,
            )
            body = constructor.body if constructor is not None else []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if not (
                body
                and isinstance(body[-1], ast.Expr)
                and isinstance(body[-1].value, ast.Call)
                and self.index.resolve(body[-1].value.func, self.module)
                == PACKAGE + ".artifacts.immutability.freeze_fields"
                and len(body[-1].value.args) == 1
                and isinstance(body[-1].value.args[0], ast.Name)
                and body[-1].value.args[0].id == "self"
                and not any(_constructor_returns(statement) for statement in body[:-1])
            ):
                self._add(
                    node,
                    _OWNED_COLLECTIONS,
                    node.name,
                    "Detach owned mappings and NumPy buffers with freeze_fields(self) last in __post_init__",
                )
        self.generic_visit(node)
        self.scope.pop()
        self.class_name = previous

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous_function = self.in_function
        previous_failure_handler = self.failure_handler
        self.failure_handler = any(
            self.index.resolve(decorator, self.module)
            == "grassmoth_server.actions.errors.execution_failure_handler"
            for decorator in node.decorator_list
        )
        self.scope.append(node.name)
        bindings = _scope_bindings(node.body)
        for parameter in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ):
            if parameter is not None:
                bindings[parameter.arg] = ""
        self.local_bindings.append(bindings)
        if (
            self._class_owned()
            and node.returns is not None
            and any(
                self._name(decorator) in {"property", "cached_property"}
                for decorator in node.decorator_list
            )
            and self._mutable(node.returns)
        ):
            self._add(
                node.returns,
                _MUTABLE_CORE,
                "return",
                "Expose a read-only tuple, frozenset or Mapping from the core owner; keep mutable builders private",
            )
        self.in_function = True
        self.generic_visit(node)
        self.scope.pop()
        self.local_bindings.pop()
        self.in_function = previous_function
        self.failure_handler = previous_failure_handler

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and self._class_owned()
            and not self.in_function
            and self._mutable(node.annotation)
        ):
            self._add(
                node.annotation,
                _MUTABLE_CORE,
                "field:" + node.target.id,
                "Core fields expose tuple, frozenset or Mapping; own the data at construction",
            )
        self.generic_visit(node)

    def _view_check(self, node: ast.expr | ast.stmt, target: str) -> None:
        if projection_checks(self.path):
            self._add(
                node,
                _PARTIAL_VIEW,
                target,
                "Projections are total; fix the core type/constructor, or resolve external input at the boundary before projecting",
            )

    def _discarded_outcome(self, node: ast.expr) -> None:
        if not isinstance(node, ast.Call):
            return
        if isinstance(node.func, ast.Name):
            for bindings in reversed(self.local_bindings):
                if node.func.id in bindings:
                    if not bindings[node.func.id]:
                        return
                    break
        name = self.index.resolve(node.func, self.module)
        if name not in self.index.functions:
            return
        path, function = self.index.functions[name]
        if role_for_path(path) not in {"compiler", "execution"} or function.returns is None:
            return
        alternatives = self.index.return_variants(
            function.returns, self.index.module(path), containers=False
        )
        if len(alternatives) > 1 and any(self.index.owned(member) for member in alternatives):
            self._add(
                node,
                _DISCARDED_OUTCOME,
                name,
                "Consume or forward the typed outcome; an expected rejection cannot be discarded",
            )

    @override
    def visit_Expr(self, node: ast.Expr) -> None:
        self._discarded_outcome(node.value)
        self.generic_visit(node)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        if any(isinstance(target, ast.Name) and target.id == "_" for target in node.targets):
            self._discarded_outcome(node.value)
        self.generic_visit(node)

    @override
    def visit_Raise(self, node: ast.Raise) -> None:
        self._view_check(node, "raise")
        self.generic_visit(node)

    @override
    def visit_Assert(self, node: ast.Assert) -> None:
        self._view_check(node, "assert")
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        name = self._name(node.func)
        qualified = self.index.resolve(node.func, self.module)
        if (
            name in _CORE_REVALIDATION
            or name.startswith("model_validate")
            or name in {"validate_python", "validate_json", "validate_strings"}
        ):
            self._view_check(node, name)
        if role_for_path(self.path) not in {None, "edge"} and (
            name.startswith("model_validate")
            or name in {"validate_python", "validate_json", "validate_strings"}
        ):
            receiver = (
                self.index.resolve(node.func.value, self.module)
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"cls", "self"}
                and self.class_name is not None
            ):
                receiver = self.module + "." + self.class_name
            target = self.index.adapters.get(receiver)
            target_module = receiver.rpartition(".")[0]
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and self.index.resolve(node.func.value.func, self.module) == "pydantic.TypeAdapter"
                and node.func.value.args
            ):
                target = node.func.value.args[0]
                target_module = self.module
            variants = (
                self.index.return_variants(target, target_module)
                if target is not None
                else {receiver}
            )
            if not variants or not all(
                variant in self.index.classes and self._owned_here(variant) for variant in variants
            ):
                self._add(
                    node,
                    _OWNER_PARSE,
                    name,
                    "Use the target owner's typed smart constructor; parse stored/transport JSON at its edge before compilation",
                )
        if (
            qualified == "json.loads"
            and role_for_path(self.path) not in {None, "edge"}
            and not self._class_owned()
        ):
            self._add(
                node,
                _OWNER_PARSE,
                "json.loads",
                "Decode external or stored JSON at its edge or in its target value's owner parser",
            )
        bypass = (
            name == "model_construct"
            or (
                name == "model_copy"
                and any(keyword.arg in {"update", None} for keyword in node.keywords)
            )
            or (qualified in self.index.evidence and not self._owned_here(qualified))
        )
        if (
            name == "__setattr__"
            and isinstance(node.func, ast.Attribute)
            and self._name(node.func.value) == "object"
            and node.args
        ):
            bypass = not (
                isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and self.in_function
                and (
                    self.scope[-1] in {"__init__", "__post_init__"}
                    or (
                        self.module == PACKAGE + ".artifacts.immutability"
                        and self.scope == ["freeze_fields"]
                    )
                )
            )
        if qualified == PACKAGE + ".artifacts.immutability.freeze_fields":
            bypass = not (
                self.class_name is not None
                and self.index.frozen(self.module + "." + self.class_name)
                and self.in_function
                and self.scope[-1] == "__post_init__"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            )
        if name == "cast" and len(node.args) == 2:
            target = _annotation_expr(node.args[0])
            bypass = self._mutable(target) or any(
                self.index.resolve(part, self.module) in self.index.evidence
                for part in ast.walk(target)
                if isinstance(part, ast.expr)
            )
        if bypass:
            self._add(
                node,
                _CORE_BYPASS,
                name,
                "Use validated construction/revision or the compiler owner, and preserve read-only interfaces",
            )
        self.generic_visit(node)

    @override
    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                id(node) in self.rebuild_dicts
                and key is None
                and any(
                    isinstance(call, ast.Call)
                    and self._name(call.func) in {"model_dump", "model_dump_json"}
                    for call in ast.walk(value)
                )
            ):
                self._add(
                    node,
                    _CORE_BYPASS,
                    "dump-spread",
                    "Revise owned values at their owner; never dump and spread a value to rebuild it",
                )
        self.generic_visit(node)

    def _check_catches(
        self, node: ast.stmt | ast.ExceptHandler, errors: list[ast.expr], body: list[ast.stmt]
    ) -> None:
        if role_for_path(self.path) is None:
            return
        names = [
            self._name(part)
            for error in errors
            for part in ast.walk(error)
            if isinstance(part, (ast.Name, ast.Attribute))
        ]
        forbidden = {
            name
            for name in names
            if name in vars(builtins)
            and isinstance(vars(builtins)[name], type)
            and issubclass(vars(builtins)[name], BaseException)
        }
        parse_operation = len(body) == 1 and any(
            isinstance(call, ast.Call) and self.index.resolve(call.func, self.module) == "ast.parse"
            for call in ast.walk(body[0])
        )
        declared_failure = (
            role_for_path(self.path) in {"shell", "edge"}
            and self.failure_handler
            and forbidden
            <= {
                "Exception",
                "OSError",
                "FileNotFoundError",
                "FileExistsError",
                "ProcessLookupError",
                "TimeoutError",
            }
        )
        decoding_operation = (
            len(body) == 1
            and forbidden == {"UnicodeDecodeError"}
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "decode"
                for call in ast.walk(body[0])
            )
        )
        if (
            forbidden
            and not declared_failure
            and not decoding_operation
            and not (forbidden == {"SyntaxError"} and parse_operation)
        ):
            self._add(
                node,
                _BUILTIN_CATCH,
                "catch",
                "Catch a specific expected parse/constructor error at its operation; consume typed scientific outcomes and let bugs reach declared shell failure handlers",
            )

    @override
    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            self._check_catches(handler, [handler.type or ast.Name(id="Exception")], node.body)
        self.generic_visit(node)

    @override
    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Call)
                and self.index.resolve(context.func, self.module) == "contextlib.suppress"
            ):
                self._check_catches(node, context.args, node.body)
        self.generic_visit(node)

    @override
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Call)
                and self.index.resolve(context.func, self.module) == "contextlib.suppress"
            ):
                self._check_catches(node, context.args, node.body)
        self.generic_visit(node)


def scan_text(
    source: str,
    *,
    path: str,
    domain_type_names: frozenset[str] | None = None,
    rules: frozenset[str] | None = None,
    type_index: TypeIndex | None = None,
) -> list[Violation]:
    """Return violations from one Python source string."""
    tree = ast.parse(source, filename=path)
    resolved_domain_types = (
        _discover_domain_type_names([tree]) if domain_type_names is None else domain_type_names
    )
    visitor = _AnnotationVisitor(
        path,
        resolved_domain_types,
        _discover_typing_bindings(tree),
        _ALL_RULES if rules is None else rules,
    )
    visitor.visit(tree)
    index = (
        type_index
        if type_index is not None
        else TypeIndex(
            [(name, module) for name, module in _production_sources() if name != path]
            + [(path, tree)]
        )
    )
    core_visitor = _CoreVisitor(tree, path, _ALL_RULES if rules is None else rules, index)
    core_visitor.visit(tree)
    lines = source.splitlines()
    return [
        violation
        for violation in visitor.violations + core_visitor.violations
        if re.search(r"# noqa: " + re.escape(violation.code) + r" -- \S", lines[violation.line - 1])
        is None
    ]


def scan_paths(
    paths: list[Path],
    *,
    repo_root: Path,
    rules: frozenset[str] | None = None,
) -> list[Violation]:
    """Scan Python files under the requested paths."""
    python_files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            python_files.add(path)
        elif path.is_dir():
            python_files.update(path.rglob("*.py"))

    parsed_sources: list[tuple[str, str, ast.Module]] = []
    for path in sorted(python_files):
        relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
        source = path.read_text(encoding="utf-8")
        parsed_sources.append(
            (
                relative_path,
                source,
                ast.parse(
                    source,
                    filename=relative_path,
                ),
            )
        )

    domain_type_names = _discover_domain_type_names(
        [tree for _path, _source, tree in parsed_sources]
    )
    type_index = TypeIndex(
        [
            (path, tree)
            for path, tree in _production_sources()
            if path not in {p for p, _, _ in parsed_sources}
        ]
        + [(path, tree) for path, _, tree in parsed_sources]
    )
    violations: list[Violation] = []
    for relative_path, source, _tree in parsed_sources:
        violations.extend(
            scan_text(
                source,
                path=relative_path,
                domain_type_names=domain_type_names,
                rules=rules,
                type_index=type_index,
            )
        )
    return sorted(
        violations,
        key=lambda item: (item.path, item.line, item.column, item.code),
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            *SOURCE_ROOTS,
            repo_root / "apps/grassmoth-server/scripts",
            repo_root / "apps/grassmoth-server/evaluation",
            repo_root / "apps/grassmoth-server/notebooks",
        ],
        help="Python files or directories to scan (default: production Python trees)",
    )
    parser.add_argument(
        "--select",
        action="append",
        choices=sorted(_ALL_RULES),
        help="Run only this rule (repeatable; default: all rules)",
    )
    args = parser.parse_args(argv)

    paths = [path if path.is_absolute() else Path.cwd() / path for path in args.paths]
    selected_rules = frozenset(args.select or _ALL_RULES)
    role_inventory()
    violations = scan_paths(paths, repo_root=repo_root, rules=selected_rules)
    for violation in violations:
        print(violation.diagnostic(), file=sys.stderr)
    if violations:
        print(f"type-boundary check failed: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
