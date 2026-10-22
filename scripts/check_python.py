"""Run package-scoped Python checks and shared workspace checks from one owner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "apps/grassmoth-server"
LIBRARY = ROOT / "packages/grassmoth"


def main() -> int:
    """Execute the selected check with the root configuration and shared environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("library", "server", "shared"))
    parser.add_argument("task")
    args, extra = parser.parse_known_args()
    package = LIBRARY if args.scope == "library" else SERVER
    source = str(package / "src")
    paths = [source, str(package / "tests")]
    if args.scope == "server":
        paths.extend(str(SERVER / name) for name in ("evaluation", "notebooks"))
    else:
        paths.extend(
            (str(ROOT / "tests"), str(ROOT / "conftest.py"), str(Path(__file__).resolve()))
        )
    module = [sys.executable, "-m"]
    dependency_rules = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["deptry"]
    dependency_flags = [
        "--per-rule-ignores",
        ",".join(
            f"{rule}={'|'.join(names)}"
            for rule, names in dependency_rules["per_rule_ignores"].items()
        ),
    ]
    commands = {
        "lint:ruff": [
            ["ruff", "check", "--no-fix", *paths],
            [
                *module,
                "scripts.checks.run_pydoclint",
                "--config",
                str(ROOT / "pyproject.toml"),
                "--quiet",
                source,
            ],
        ],
        "format:check": [["ruff", "format", "--check", *paths]],
        "format": [["ruff", "format", *paths]],
        "lint:deptry": [["deptry", source, *dependency_flags]],
        "lint:source-rules": [
            ["ruff", "check", source, "--no-fix", "--select", "FBT001,FBT002,PGH003"]
        ],
        "lint:ty": [["ty", "check"]],
        "lint:basedpyright": [["basedpyright", "--threads", "4"]],
        "lint:config": [[*module, "scripts.checks.validate_config"]],
        "lint:type-boundaries": [[*module, "scripts.checks.check_type_boundaries"]],
        "lint:type-boundaries:tests": [
            [
                *module,
                "scripts.checks.check_type_boundaries",
                "--select",
                "CORE001",
                "--select",
                "CORE002",
                str(LIBRARY / "tests"),
                str(SERVER / "tests"),
                str(ROOT / "tests"),
            ]
        ],
        "lint:architecture": [[*module, "scripts.checks.check_architecture_boundaries"]],
        "lint:public": [[*module, "scripts.checks.check_public_imports"]],
        "lint:imports": [["lint-imports", "--no-cache"]],
        "lint:vulture": [[*module, "scripts.checks.run_vulture"]],
        "lint:fields": [[*module, "scripts.checks.check_fields"]],
        "lint:annotations": [[*module, "scripts.checks.architecture_roles"]],
        "lint:duplicates": [[*module, "scripts.checks.find_duplicates"]],
        "test": [["pytest", str(package / "tests")]],
        "test:all": [["pytest", str(package / "tests"), "-m", ""]],
    }
    scoped = ("lint:ruff", "format:check", "lint:deptry", "lint:source-rules")
    shared = tuple(name for name in commands if name.startswith("lint:") and name not in scoped)
    if args.task == "lint":
        tasks = shared if args.scope == "shared" else scoped
    elif args.task == "lint:fix":
        commands["lint:fix"] = [
            ["ruff", "check", "--fix", *paths],
            ["ruff", "format", *paths],
            [*module, "scripts.checks.validate_config", "--fix"],
        ]
        tasks = ("lint:fix",)
    elif args.task in commands:
        tasks = (args.task,)
    else:
        parser.error(f"unknown Python task: {args.task}")
    if args.task == "lint:public":
        extra = [
            str(Path(value).resolve()) if not value.startswith("-") else value for value in extra
        ]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(SERVER)))}

    def run(task: str) -> int:
        for command in commands[task]:
            result = subprocess.run(
                [*command, *extra],
                cwd=package if task == "lint:deptry" else ROOT,
                env=env,
                check=False,
            )
            if result.returncode:
                return result.returncode
        return 0

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = tuple(pool.map(run, tasks))
    return int(any(results))


if __name__ == "__main__":
    raise SystemExit(main())
