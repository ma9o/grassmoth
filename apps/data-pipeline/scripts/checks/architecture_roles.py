"""Single owner of production module roles and architectural enforcement.

Directory roles protect future modules without a per-class migration allowlist.
Exact assignments describe mixed directory responsibilities. The two boundary
checkers use this policy, including whole-role annotation coverage.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Literal

type ModuleRole = Literal["edge", "domain", "compiler", "execution", "projection", "shell"]

PURE_ROLES: frozenset[ModuleRole] = frozenset({"domain", "compiler", "execution", "projection"})

PACKAGE = "nof1_causal_lab"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / PACKAGE
DIRECTORY_ROLES: dict[str, ModuleRole] = {
    "actions": "shell",
    "actions/contracts": "domain",
    "actions/temporal": "shell",
    "api": "edge",
    "artifacts": "domain",
    "compute": "domain",
    "llm": "domain",
    "llm/harness": "shell",
    "llm/harness/events": "edge",
    "models": "compiler",
    "models/ssm": "execution",
    "models/ssm/compile": "compiler",
    "models/ssm/preparation": "compiler",
    "models/ssm/structure": "compiler",
    "observations": "compiler",
    "storage": "edge",
    "study": "edge",
    "study/projections": "projection",
}
MODULE_ROLES: dict[str, ModuleRole] = {
    "__init__": "domain",
    "actions/call_logs": "edge",
    "actions/errors": "domain",
    "actions/fit/conditioning": "edge",
    "actions/fit/remote": "edge",
    "actions/messages": "projection",
    "actions/prepare_data/context": "domain",
    "actions/prepare_data/materialization": "edge",
    "actions/prepare_data/messages": "domain",
    "actions/prepare_data/planning": "compiler",
    "actions/prepare_data/prompts": "projection",
    "actions/prepare_data/tool_contracts": "domain",
    "actions/prepare_data/window_text": "compiler",
    "actions/prepare_data/worker_output": "domain",
    "actions/progress": "edge",
    "actions/simulate/summaries": "projection",
    "actions/temporal/client": "edge",
    "actions/temporal/errors": "edge",
    "actions/temporal/extraction/activities": "edge",
    "actions/temporal/extraction/cache": "edge",
    "actions/temporal/extraction/messages": "edge",
    "actions/temporal/extraction/settings": "edge",
    "actions/temporal/extraction/source_data": "edge",
    "actions/temporal/llm/activities": "edge",
    "actions/temporal/llm/context": "edge",
    "actions/temporal/llm/messages": "edge",
    "actions/temporal/llm/storage": "edge",
    "actions/temporal/llm/tools": "edge",
    "actions/temporal/study/messages": "edge",
    "artifacts/model/coefficient_redundancy": "compiler",
    "artifacts/model/semantics": "compiler",
    "artifacts/probability/membership": "compiler",
    "artifacts/reports/histograms": "execution",
    "bootstrap": "edge",
    "compilation_errors": "domain",
    "compute/batching": "compiler",
    "compute/traced_memory": "compiler",
    "compute/modal_images": "edge",
    "compute/resources": "edge",
    "config": "edge",
    "json_types": "domain",
    "llm/openrouter": "edge",
    "llm/sessions": "edge",
    "models/diagnostics/history_comparison": "execution",
    "models/ssm/compiled_access": "compiler",
    "models/ssm/constants": "domain",
    "models/ssm/counterfactual/orchestration": "compiler",
    "models/ssm/diagnostics/reachability": "compiler",
    "models/ssm/dynamics/cache_identity": "domain",
    "models/ssm/dynamics/spec": "domain",
    "models/ssm/inference/samplers/sweep_plan": "compiler",
    "models/ssm/inference/fit": "shell",
    "models/ssm/inference/samplers/nuts": "shell",
    "models/ssm/inference/samplers/mpgibbs/fit": "shell",
    "models/ssm/inference/samplers/mpgibbs/runner": "shell",
    "models/ssm/inference/samplers/pgibbs/fit": "shell",
    "models/ssm/inference/samplers/pgibbs/runner": "shell",
    "models/ssm/inference/samplers/progress": "shell",
    "models/ssm/inference/samplers/sweeps": "shell",
    "models/ssm/inference/samplers/particle/context": "compiler",
    "models/ssm/inference/samplers/particle/setup": "compiler",
    "models/ssm/parameterization": "compiler",
    "models/ssm/predictive/types": "domain",
    "models/ssm/priors": "compiler",
    "models/ssm/shapes": "domain",
    "models/ssm/transition_kinds": "domain",
    "storage/cache": "shell",
    "study/artifact_files": "domain",
    "study/errors": "domain",
    "study/records": "domain",
    "study/snapshot": "domain",
    "study/state": "domain",
}

# Annotations cover every non-numerical role and the published numerical seams.
ANNOTATION_ROLES: frozenset[ModuleRole] = frozenset(
    {"domain", "projection", "compiler", "edge", "shell"}
)
ANNOTATION_MODULES = frozenset(
    {
        "models/diagnostics/history_comparison",
        "models/ssm/predictive/observations",
        "models/ssm/execution/contracts",
        "models/ssm/execution/dynamical_model",
        "models/ssm/inference/types",
        "models/ssm/inference/problem",
        "models/ssm/inference/parameter_tracing",
        "models/ssm/inference/parameter_transform",
        "models/ssm/inference/samplers/particle/dsmc",
        "models/ssm/inference/sample_projection",
        "models/ssm/inference/mcmc_state",
        "models/ssm/inference/samplers/mpgibbs/fit",
        "models/ssm/inference/samplers/mpgibbs/runner",
        "models/ssm/inference/samplers/pgibbs/fit",
        "models/ssm/inference/samplers/pgibbs/runner",
    }
)


def module_role(relative_module: str) -> ModuleRole:
    """Classify one extensionless package-relative path, including initializers."""
    if relative_module in MODULE_ROLES:
        return MODULE_ROLES[relative_module]
    choices = [
        prefix
        for prefix in DIRECTORY_ROLES
        if relative_module == prefix or relative_module.startswith(prefix + "/")
    ]
    if not choices:
        raise ValueError(f"Unclassified production module {relative_module}; assign its role here")
    return DIRECTORY_ROLES[max(choices, key=len)]


def role_for_path(path: str) -> ModuleRole | None:
    """Resolve checker paths; auxiliary Python trees have no production role."""
    marker = f"src/{PACKAGE}/"
    if marker not in path:
        return None
    relative = path.split(marker, 1)[1].removesuffix(".py")
    return module_role(relative)


def role_for_module(module: str) -> ModuleRole:
    """Classify an import-graph module name using the same assignments."""
    relative = module.removeprefix(PACKAGE).lstrip(".").replace(".", "/")
    return module_role(relative or "__init__")


def role_inventory(source_root: Path = SOURCE_ROOT) -> dict[Path, ModuleRole]:
    """Fail on missing/stale assignments when scanning the complete package."""
    modules = {
        path.relative_to(source_root).with_suffix("").as_posix(): path
        for path in source_root.rglob("*.py")
    }
    if source_root.resolve() == SOURCE_ROOT.resolve():
        stale = MODULE_ROLES.keys() - modules.keys()
        if stale:
            raise ValueError(f"Delete stale role assignments: {sorted(stale)}")
    return {path: module_role(name) for name, path in modules.items()}


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


def run_annotation_checks() -> int:
    """Check complete roles and named numerical entry-point/solver modules."""
    paths = sorted(
        str(path)
        for path, role in role_inventory().items()
        if role in ANNOTATION_ROLES
        or path.relative_to(SOURCE_ROOT).with_suffix("").as_posix() in ANNOTATION_MODULES
    )
    return subprocess.call(
        ["ruff", "check", "--no-fix", "--select", "ANN", *paths], cwd=PROJECT_ROOT
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(run_annotation_checks())
