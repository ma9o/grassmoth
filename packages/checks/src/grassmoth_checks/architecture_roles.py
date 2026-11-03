"""Single owner of production module roles and architectural enforcement.

Directory roles protect future modules without a per-class migration allowlist.
Exact assignments describe mixed directory responsibilities. The two boundary
checkers use this policy, including whole-role annotation coverage.
"""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

type ModuleRole = Literal["edge", "domain", "compiler", "execution", "projection", "shell"]

PURE_ROLES: frozenset[ModuleRole] = frozenset({"domain", "compiler", "execution", "projection"})

PACKAGE = "grassmoth"
SERVICE_PACKAGE = "grassmoth_server"
PACKAGES = (PACKAGE, SERVICE_PACKAGE)
REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "apps/grassmoth-server"
SOURCE_ROOT = REPO_ROOT / "packages/grassmoth/src" / PACKAGE
SOURCE_ROOTS = (SOURCE_ROOT, PROJECT_ROOT / "src" / SERVICE_PACKAGE)


class RolePolicy(TypedDict):
    directories: dict[str, ModuleRole]
    modules: dict[str, ModuleRole]
    annotation_modules: list[str]


@lru_cache(maxsize=2)
def role_policy(package: str) -> RolePolicy:
    """Read assignments from the package that owns the classified modules."""
    source = next(root for root in SOURCE_ROOTS if root.name == package)
    config = tomllib.loads((source.parents[1] / "pyproject.toml").read_text())
    policy = config["tool"]["grassmoth-checks"]["roles"]
    return {
        "directories": policy["directories"],
        "modules": policy["modules"],
        "annotation_modules": policy["annotation-modules"],
    }


# Annotations cover every non-numerical role and the published numerical seams.
ANNOTATION_ROLES: frozenset[ModuleRole] = frozenset(
    {"domain", "projection", "compiler", "edge", "shell"}
)


def module_role(relative_module: str, package: str = PACKAGE) -> ModuleRole:
    """Classify one extensionless package-relative path, including initializers."""
    policy = role_policy(package)
    modules = policy["modules"]
    directories = policy["directories"]
    if relative_module in modules:
        return modules[relative_module]
    choices = [
        prefix
        for prefix in directories
        if relative_module == prefix or relative_module.startswith(prefix + "/")
    ]
    if not choices:
        raise ValueError(
            f"Unclassified production module {relative_module}; assign its role in the owning pyproject.toml"
        )
    return directories[max(choices, key=len)]


def role_for_path(path: str) -> ModuleRole | None:
    """Resolve checker paths; auxiliary Python trees have no production role."""
    for package in PACKAGES:
        marker = f"src/{package}/"
        if marker in path:
            relative = path.split(marker, 1)[1].removesuffix(".py")
            return module_role(relative, package)
    return None


def role_for_module(module: str) -> ModuleRole:
    """Classify an import-graph module name using the same assignments."""
    package, _, relative = module.partition(".")
    return module_role(relative.replace(".", "/") or "__init__", package)


def role_inventory(source_root: Path | None = None) -> dict[Path, ModuleRole]:
    """Fail on missing/stale assignments when scanning the complete package."""
    if source_root is None:
        return {path: role for root in SOURCE_ROOTS for path, role in role_inventory(root).items()}
    package = source_root.name
    modules = {
        path.relative_to(source_root).with_suffix("").as_posix(): path
        for path in source_root.rglob("*.py")
    }
    if source_root.resolve() in SOURCE_ROOTS:
        assignments = role_policy(package)["modules"]
        stale = assignments.keys() - modules.keys()
        if stale:
            raise ValueError(f"Delete stale role assignments: {sorted(stale)}")
    return {path: module_role(name, package) for name, path in modules.items()}


def fix_owner(path: str) -> str:
    """Name the architectural responsibility in every project-check diagnostic."""
    role = role_for_path(path)
    if role is None:
        return "boundary/test owner"
    return {
        "domain": "the value owner or its smart constructor",
        "compiler": "the compiler owner; establish the resolved fact once",
        "execution": "the numerical producer or compiled/bound input owner",
        "projection": "the input value owner; keep projections total",
        "edge": "the transport parser; translate typed outcomes at the edge",
        "shell": "the orchestration owner; declare failure/retry handling",
    }[role]


def projection_checks(path: str) -> bool:
    """Every projection derives a total view from already owned inputs."""
    return role_for_path(path) == "projection"


def run_annotation_checks(source_root: Path) -> int:
    """Check complete roles and named numerical entry-point/solver modules."""
    policy = role_policy(source_root.name)
    paths = sorted(
        str(path)
        for path, role in role_inventory(source_root).items()
        if role in ANNOTATION_ROLES
        or path.relative_to(source_root).with_suffix("").as_posix() in policy["annotation_modules"]
    )
    return subprocess.call(["ruff", "check", "--no-fix", "--select", "ANN", *paths])


def main() -> int:
    """Check annotations for one package supplied by its lint command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    return run_annotation_checks(args.source_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
