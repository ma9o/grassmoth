"""Identify public declarations and members without importing application code."""

from __future__ import annotations

import ast
import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def public_exports(project_root: Path) -> frozenset[str]:
    """Read explicit re-exports from the package's declared public modules."""
    config = tomllib.loads((project_root / "pyproject.toml").read_text())
    return frozenset(
        f"{node.module}.{alias.name}"
        for relative in config["tool"]["grassmoth-checks"]["public-api"]
        for node in ast.parse((project_root / relative).read_text()).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == alias.name
    )


def public_contract_locations(
    source_roots: tuple[Path, ...], exports: frozenset[str]
) -> frozenset[tuple[Path, int]]:
    """Follow public signatures, aliases and bases to their owned member declarations."""
    definitions: dict[str, tuple[Path, str, ast.stmt]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for root in source_roots:
        for path in root.rglob("*.py"):
            module = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(
                ".__init__"
            )
            tree = ast.parse(path.read_text())
            bindings[module] = {
                alias.asname or alias.name: f"{node.module}.{alias.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            } | {
                alias.asname or alias.name.split(".")[0]: alias.name
                if alias.asname
                else alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            for node in tree.body:
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    name = node.name
                elif isinstance(node, ast.TypeAlias):
                    name = node.name.id
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    name = node.target.id
                else:
                    continue
                definitions[f"{module}.{name}"] = path.resolve(), module, node

    def resolve(node: ast.AST, module: str) -> str:
        if isinstance(node, ast.Name):
            return bindings[module].get(node.id, f"{module}.{node.id}")
        if isinstance(node, ast.Attribute):
            return f"{resolve(node.value, module)}.{node.attr}"
        return ""

    locations: set[tuple[Path, int]] = set()
    visited: set[str] = set()
    pending = list(exports)
    while pending:
        identity = pending.pop()
        if identity in visited:
            continue
        visited.add(identity)
        if identity not in definitions:
            module, _, name = identity.rpartition(".")
            target = bindings.get(module, {}).get(name)
            if target is not None:
                pending.append(target)
            continue
        path, module, node = definitions[identity]
        locations.add((path, node.lineno))
        annotations: list[ast.AST] = []
        members = node.body if isinstance(node, ast.ClassDef) else [node]
        if isinstance(node, ast.ClassDef):
            annotations.extend(node.bases)
        for member in members:
            if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                if member.name.startswith("_") and member.name != "__init__":
                    continue
                locations.add(
                    (
                        path,
                        min(
                            [
                                member.lineno,
                                *(decorator.lineno for decorator in member.decorator_list),
                            ]
                        ),
                    )
                )
                args = member.args
                annotations.extend(
                    argument.annotation
                    for argument in (
                        *args.posonlyargs,
                        *args.args,
                        *args.kwonlyargs,
                        args.vararg,
                        args.kwarg,
                    )
                    if argument is not None and argument.annotation is not None
                )
                if member.returns is not None:
                    annotations.append(member.returns)
            elif isinstance(member, ast.AnnAssign):
                locations.add((path, member.lineno))
                annotations.append(member.annotation)
            elif isinstance(member, ast.Assign):
                locations.add((path, member.lineno))
            elif isinstance(member, ast.TypeAlias):
                annotations.append(member.value)
        pending.extend(
            resolve(reference, module)
            for annotation in annotations
            for reference in ast.walk(annotation)
            if isinstance(reference, ast.Name | ast.Attribute)
        )
    return frozenset(locations)
