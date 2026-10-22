"""Enforce the framework's public library boundary for its callers.

ARCH011: callers import nof1_causal_service.public, never implementation modules or
private names. Local helper imports are followed, including imports under
TYPE_CHECKING, so a helper cannot
hide a dependency on execution internals. Dynamic imports must have a literal
module name; runtime code loading and import-path modification are forbidden.

ARCH013: service callers import the scientific library only through
nof1_causal_lab.public. The same traversal checks runtime and type-only imports.

The default selection is evaluation/ plus the service package. Explicit files/directories select other
callers, including notebooks and gitignored scratchpad experiments. Directories
are read from disk, not Git's tracked-file list. The framework package itself,
including its HTTP adapter and isolated patched copies, is implementation.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from scripts.checks.architecture_roles import (
    PACKAGE,
    PACKAGES,
    PROJECT_ROOT,
    SERVICE_PACKAGE,
    SOURCE_ROOT,
)

_PUBLIC = SERVICE_PACKAGE + ".public"
_LIBRARY_PUBLIC = PACKAGE + ".public"
_REPO = PROJECT_ROOT.parents[1]
_DEFAULT_CALLERS = (PROJECT_ROOT / "evaluation",)


@dataclass(frozen=True)
class PublicImportViolation:
    """An import that reaches past the supported library interface."""

    path: Path
    line: int
    target: str
    boundary: str = _PUBLIC

    def diagnostic(self) -> str:
        """Identify the caller responsible for crossing the boundary."""
        return (
            f"{self.path}:{self.line}: {'ARCH013' if self.boundary == _LIBRARY_PUBLIC else 'ARCH011'} {self.target}; "
            f"fix at the caller: use {self.boundary} or patch the framework owner"
        )


def _member(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _module(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        if path.is_relative_to(root):
            parts = path.relative_to(root).with_suffix("").parts
            return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    return path.stem


def _local_module(name: str, roots: tuple[Path, ...]) -> Path | None:
    for root in roots:
        base = root.joinpath(*name.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _local_dependencies(name: str, roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Include package initializers as well as the imported module or symbol owner."""
    parts = name.split(".")
    return tuple(
        path
        for length in range(1, len(parts) + 1)
        if (path := _local_module(".".join(parts[:length]), roots)) is not None
    )


def _implementation(path: Path) -> bool:
    """Package location identifies implementation, also inside isolated checkouts."""
    return any(parent.name in PACKAGES and parent.parent.name == "src" for parent in path.parents)


def _origin(node: ast.expr, bindings: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _origin(node.value, bindings) + "." + node.attr
    return ""


def find_public_import_violations(
    paths: tuple[Path, ...], *, import_roots: tuple[Path, ...] = (), service_callers: bool = False
) -> tuple[PublicImportViolation, ...]:
    """Inspect selected callers and their local dependencies without importing code."""
    public_boundary = _LIBRARY_PUBLIC if service_callers else _PUBLIC
    protected = (PACKAGE,) if service_callers else PACKAGES
    roots = tuple(
        root.resolve() for root in (*import_roots, SOURCE_ROOT.parent, PROJECT_ROOT, _REPO)
    )
    pending = [
        file.resolve()
        for path in paths
        for file in (sorted(path.rglob("*.py")) if path.is_dir() else (path,))
    ]
    visited: set[Path] = set()
    violations: list[PublicImportViolation] = []
    while pending:
        path = pending.pop()
        if path in visited or (
            _implementation(path)
            and not (service_callers and _member(_module(path, roots), SERVICE_PACKAGE))
        ):
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        module = _module(path, roots)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        bindings: dict[str, str] = {}
        targets: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.append((node.lineno, alias.name))
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                name = importlib.util.resolve_name(name, package) if node.level else name
                for alias in node.names:
                    target = name + "." + alias.name
                    targets.append((node.lineno, target))
                    bindings[alias.asname or alias.name] = target

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = _origin(node.func, bindings)
                if target in {"importlib.import_module", "__import__", "builtins.__import__"}:
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and not node.args[0].value.startswith(".")
                    ):
                        targets.append((node.lineno, node.args[0].value))
                    else:
                        violations.append(
                            PublicImportViolation(
                                path, node.lineno, "unresolved dynamic import", public_boundary
                            )
                        )
                elif target in {
                    "exec",
                    "eval",
                    "builtins.exec",
                    "builtins.eval",
                    "importlib.util.spec_from_file_location",
                    "importlib.machinery.SourceFileLoader",
                    "runpy.run_path",
                } or target.startswith("sys.path."):
                    violations.append(
                        PublicImportViolation(
                            path,
                            node.lineno,
                            "runtime code loading or import-path modification",
                            public_boundary,
                        )
                    )
            elif isinstance(node, ast.Attribute) and _member(
                _origin(node, bindings), public_boundary
            ):
                targets.append((node.lineno, _origin(node, bindings)))
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                assigned = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(_origin(target, bindings) == "sys.path" for target in assigned):
                    violations.append(
                        PublicImportViolation(
                            path, node.lineno, "import-path modification", public_boundary
                        )
                    )

        for line, target in targets:
            if any(_member(target, package) for package in protected):
                public = _member(target, public_boundary) and not any(
                    part.startswith("_") or part == "*"
                    for part in target.removeprefix(public_boundary).split(".")
                    if part
                )
                if not public:
                    violations.append(
                        PublicImportViolation(
                            path, line, f"forbidden import {target}", public_boundary
                        )
                    )
                continue
            if service_callers and _member(target, SERVICE_PACKAGE):
                continue
            local_roots = (path.parent, *roots)
            dependencies = _local_dependencies(target, local_roots)
            if any(_implementation(dependency) for dependency in dependencies):
                violations.append(
                    PublicImportViolation(path, line, f"forbidden import {target}", public_boundary)
                )
            else:
                pending.extend(dependencies)
    return tuple(sorted(set(violations), key=lambda item: (str(item.path), item.line, item.target)))


def main() -> int:
    """Check maintained callers or explicitly selected local experiment paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = tuple(args.paths) or _DEFAULT_CALLERS
    for path in paths:
        if not path.exists():
            parser.error(f"Caller path does not exist: {path}")
    violations = (
        *find_public_import_violations(paths),
        *find_public_import_violations(
            (SOURCE_ROOT.with_name(SERVICE_PACKAGE),), service_callers=True
        ),
    )
    for violation in violations:
        print(violation.diagnostic())
    print(f"Public library boundary: {len(violations)} violation(s)")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
