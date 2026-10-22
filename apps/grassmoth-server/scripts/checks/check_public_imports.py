"""Enforce the framework's public library boundary for its callers.

ARCH011: external callers import only grassmoth.public, never service modules,
library implementation modules or private names. Local helper imports are followed,
including imports under TYPE_CHECKING, so a helper cannot hide a dependency on
execution internals. Dynamic imports must have a literal module name; runtime
code loading and import-path modification are forbidden.

ARCH013: service callers import the scientific library only through
grassmoth.public. The same traversal checks runtime and type-only imports.

The default selection is every tracked or new non-ignored Python caller in the
repository, including notebooks and scripts. Explicit paths also select ignored
experiments. Test entry points may use internals; a non-test caller's helpers are
checked even inside tests.
Dependency environments and generated caches are not project callers. Package
implementation, including isolated patched copies, owns its internal imports.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.checks.architecture_roles import (
    PACKAGE,
    PACKAGES,
    PROJECT_ROOT,
    SERVICE_PACKAGE,
    SOURCE_ROOTS,
)

_PUBLIC = PACKAGE + ".public"
_REPO = PROJECT_ROOT.parents[1]
_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        "__marimo__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".vulture_cache",
        ".next",
        ".turbo",
        ".jax_cache",
    }
)


@dataclass(frozen=True)
class PublicImportViolation:
    """An import that reaches past the supported library interface."""

    path: Path
    line: int
    target: str
    boundary: str = _PUBLIC
    rule: Literal["ARCH011", "ARCH013"] = "ARCH011"

    def diagnostic(self) -> str:
        """Identify the caller responsible for crossing the boundary."""
        return (
            f"{self.path}:{self.line}: {self.rule} {self.target}; "
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


def _test(path: Path) -> bool:
    return "tests" in path.parts or path.name == "conftest.py" or path.name.startswith("test_")


def _python_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Discover callers on disk, including ignored files, without entering dependencies."""
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            if not _test(path):
                files.add(path.resolve())
            continue
        for directory, subdirectories, filenames in path.walk():
            if "pyvenv.cfg" in filenames:
                subdirectories.clear()
                continue
            subdirectories[:] = [
                name
                for name in subdirectories
                if name not in _EXCLUDED_DIRECTORIES and name != "tests"
            ]
            files.update(
                (directory / name).resolve()
                for name in filenames
                if name.endswith(".py") and not _test(directory / name)
            )
    return tuple(sorted(files))


def _repository_callers() -> tuple[Path, ...]:
    """Select maintained and new callers across every workspace, honoring Git ignores."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return tuple(_REPO / name for name in result.stdout.split("\0") if name)


def _service_caller(path: Path) -> bool:
    """Service implementation and its maintenance tools own service-private imports."""
    return path.is_relative_to(PROJECT_ROOT / "scripts") or any(
        parent.name == SERVICE_PACKAGE and parent.parent.name == "src" for parent in path.parents
    )


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
    roots = tuple(
        root.resolve()
        for root in (*import_roots, *(root.parent for root in SOURCE_ROOTS), PROJECT_ROOT, _REPO)
    )
    pending = [(file, service_callers or _service_caller(file)) for file in _python_files(paths)]
    visited: set[tuple[Path, bool]] = set()
    violations: list[PublicImportViolation] = []
    while pending:
        path, service = pending.pop()
        if (path, service) in visited or (
            _implementation(path) and not (service and _service_caller(path))
        ):
            continue
        visited.add((path, service))
        public_boundary = _PUBLIC
        protected = (PACKAGE,) if service else PACKAGES
        rule = "ARCH013" if service else "ARCH011"
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
                                path,
                                node.lineno,
                                "unresolved dynamic import",
                                public_boundary,
                                rule,
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
                            rule,
                        )
                    )
            elif isinstance(node, ast.Attribute) and any(
                _member(_origin(node, bindings), package + ".public") for package in protected
            ):
                targets.append((node.lineno, _origin(node, bindings)))
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                assigned = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(_origin(target, bindings) == "sys.path" for target in assigned):
                    violations.append(
                        PublicImportViolation(
                            path, node.lineno, "import-path modification", public_boundary, rule
                        )
                    )

        for line, target in targets:
            package = next((name for name in protected if _member(target, name)), None)
            if package is not None:
                boundary = _PUBLIC
                public = (
                    package == PACKAGE
                    and _member(target, boundary)
                    and not any(
                        part.startswith("_") or part == "*"
                        for part in target.removeprefix(boundary).split(".")
                        if part
                    )
                )
                if not public:
                    violations.append(
                        PublicImportViolation(
                            path, line, f"forbidden import {target}", boundary, rule
                        )
                    )
                continue
            if service and _member(target, SERVICE_PACKAGE):
                continue
            local_roots = (path.parent, *roots)
            dependencies = _local_dependencies(target, local_roots)
            if any(_implementation(dependency) for dependency in dependencies):
                violations.append(
                    PublicImportViolation(
                        path, line, f"forbidden import {target}", public_boundary, rule
                    )
                )
            else:
                pending.extend((dependency, service) for dependency in dependencies)
    return tuple(sorted(set(violations), key=lambda item: (str(item.path), item.line, item.target)))


def main() -> int:
    """Check maintained callers or explicitly selected local experiment paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            parser.error(f"Caller path does not exist: {path}")
    paths = tuple(args.paths) or _repository_callers()
    violations = find_public_import_violations(paths)
    for violation in violations:
        print(violation.diagnostic())
    print(f"Public library boundary: {len(violations)} violation(s)")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
