"""Run pydoclint with one owner for each field's documentation.

A nonblank Pydantic ``Field(description=...)`` satisfies the attribute contract.
Only that declaration is removed from the AST view passed to pydoclint; other
attributes and all function checks retain upstream behavior. Listing the same
field in ``Attributes:`` therefore reports an extra documented attribute.

Recognize assigned Field calls and outer Annotated metadata, including import
aliases. Descriptions may be literal strings or module/class constants holding
literal strings; directly imported constants are read from this app's source.
Never import the modules being checked or evaluate description expressions.
Blank, missing, and statically unresolved descriptions receive no exemption.

Pydoclint has no visitor injection API. The entry point substitutes its Visitor
for the duration of the native CLI, preserving configuration, reporting, and
exit codes. The dependency is pinned and contract tests exercise this seam.
"""

from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, override

import pydoclint.main as pydoclint_main
from pydoclint.visitor import Visitor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SOURCE_ROOTS = (
    Path(__file__).resolve().parents[4] / "packages/grassmoth/src",
    Path(__file__).resolve().parents[2] / "src",
)
_FIELD_NAMES = frozenset({"pydantic.Field", "pydantic.fields.Field"})
_ANNOTATED_NAMES = frozenset({"typing.Annotated", "typing_extensions.Annotated"})


@dataclass(frozen=True)
class _Symbols:
    imports: Mapping[str, str]
    strings: Mapping[str, str]


def _qualified_name(node: ast.expr, symbols: _Symbols) -> str | None:
    if isinstance(node, ast.Name):
        return symbols.imports.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value, symbols)
        return f"{owner}.{node.attr}" if owner else None
    return None


@cache
def _imported_literal(qualified_name: str) -> str | None:
    """Read a directly imported literal constant without executing its module."""
    module, _, name = qualified_name.rpartition(".")
    path = next(
        (
            root.joinpath(*module.split(".")).with_suffix(".py")
            for root in SOURCE_ROOTS
            if root.joinpath(*module.split(".")).with_suffix(".py").is_file()
        ),
        None,
    )
    if path is None:
        return None
    value: str | None = None
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                expression = statement.value
                value = (
                    expression.value
                    if isinstance(expression, ast.Constant) and isinstance(expression.value, str)
                    else None
                )
    return value


def _string_value(node: ast.expr | None, symbols: _Symbols) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in symbols.strings:
        return symbols.strings[node.id]
    if node is not None and (name := _qualified_name(node, symbols)) is not None:
        return _imported_literal(name)
    return None


def _scope_symbols(body: Sequence[ast.stmt], parent: _Symbols) -> _Symbols:
    """Resolve imports and string assignments in a lexical scope."""
    imports, strings = dict(parent.imports), dict(parent.strings)
    for statement in body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                name = alias.asname or alias.name.split(".")[0]
                imports[name] = alias.name if alias.asname else name
                strings.pop(name, None)
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                name = alias.asname or alias.name
                imports.pop(name, None)
                strings.pop(name, None)
                if statement.level == 0:
                    imports[name] = f"{statement.module}.{alias.name}"
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = _string_value(statement.value, _Symbols(imports, strings))
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    imports.pop(target.id, None)
                    strings.pop(target.id, None)
                    if value is not None:
                        strings[target.id] = value
        elif isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            imports.pop(statement.name, None)
            strings.pop(statement.name, None)
    return _Symbols(imports, strings)


def _has_field_description(node: ast.stmt, symbols: _Symbols) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    metadata: list[ast.expr] = []
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.annotation, ast.Subscript)
        and _qualified_name(node.annotation.value, symbols) in _ANNOTATED_NAMES
        and isinstance(node.annotation.slice, ast.Tuple)
    ):
        metadata.extend(node.annotation.slice.elts[1:])
    if node.value is not None:
        metadata.append(node.value)

    # Pydantic merges Annotated metadata from left to right; an assigned Field
    # takes precedence, including an explicit blank/None description.
    description: ast.expr | None = None
    for item in metadata:
        if isinstance(item, ast.Call) and _qualified_name(item.func, symbols) in _FIELD_NAMES:
            for keyword in item.keywords:
                if keyword.arg == "description":
                    description = keyword.value
    value = _string_value(description, symbols)
    return value is not None and bool(value.strip())


class SourceDescriptionVisitor(Visitor):
    """Let source descriptions own their fields without suppressing violations."""

    _symbols: _Symbols

    @override
    def visit_Module(self, node: ast.Module) -> None:
        self._symbols = _scope_symbols(node.body, _Symbols({}, {}))
        self.generic_visit(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self._symbols
        self._symbols = _scope_symbols(node.body, parent)
        view = copy(node)
        view.body = [
            ast.copy_location(ast.Pass(), statement)
            if _has_field_description(statement, self._symbols)
            else statement
            for statement in node.body
        ]
        first_violation = len(self.violations)
        super().visit_ClassDef(view)
        described = [
            ast.unparse(original.target)
            for original, projected in zip(node.body, view.body, strict=True)
            if isinstance(original, ast.AnnAssign) and isinstance(projected, ast.Pass)
        ]
        if described:
            for violation in self.violations[first_violation:]:
                if violation.line == node.lineno and 600 < violation.code < 700:
                    violation.msg += (
                        f" Fields owned by Field(description=...): {', '.join(described)};"
                        " omit them from Attributes."
                    )
        self._symbols = parent

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._symbols
        symbols = _scope_symbols(node.body, parent)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        )
        shadowed = {argument.arg for argument in arguments}
        self._symbols = _Symbols(
            {key: value for key, value in symbols.imports.items() if key not in shadowed},
            {key: value for key, value in symbols.strings.items() if key not in shadowed},
        )
        super().visit_FunctionDef(node)
        self._symbols = parent


def main(argv: list[str] | None = None) -> None:
    """Use the native pydoclint CLI with source-aware attribute checking."""
    original_visitor = pydoclint_main.Visitor
    setattr(pydoclint_main, "Visitor", SourceDescriptionVisitor)  # noqa: B010 -- The native CLI's imported class is intentionally replaced as a runtime factory.
    try:
        pydoclint_main.main(args=argv)
    finally:
        setattr(pydoclint_main, "Visitor", original_visitor)  # noqa: B010 -- Restore the dynamically substituted factory, including when Click exits.


if __name__ == "__main__":
    main()
