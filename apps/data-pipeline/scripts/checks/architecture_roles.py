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

PACKAGE = "nof1_causal_lab"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / PACKAGE
DIRECTORY_ROLES: dict[str, ModuleRole] = {
    "artifacts": "domain",
    "compute": "domain",
    "actions": "shell",
    "models": "compiler",
    "models/ssm": "execution",
    "models/ssm/compile": "compiler",
    "models/ssm/structure": "compiler",
    "study": "edge",
    "utils": "domain",
    "utils/harness": "shell",
    "workers": "shell",
    "workers/prompts": "projection",
    "recipes": "shell",
}
MODULE_ROLES: dict[str, ModuleRole] = {
    "__init__": "domain",
    "actions/contracts": "domain",
    "actions/call_state": "domain",
    "actions/call_logs": "edge",
    "actions/logs": "domain",
    "actions/data_checks": "shell",
    "actions/effects": "domain",
    "actions/errors": "domain",
    "actions/extraction/contracts": "domain",
    "actions/extraction/materialization": "edge",
    "actions/extraction/planning": "compiler",
    "actions/messages": "projection",
    "actions/progress": "edge",
    "actions/progress_contracts": "domain",
    "actions/results": "domain",
    "actions/io": "domain",
    "actions/temporal/activity_errors": "edge",
    "actions/temporal/backend_config": "edge",
    "actions/temporal/client": "edge",
    "actions/temporal/llm_context_adapters": "edge",
    "actions/temporal/llm_subroutine_storage": "edge",
    "actions/temporal/messages": "edge",
    "actions/temporal/preparation_cache": "edge",
    "actions/modal_fit": "edge",
    "actions/modal_runners": "edge",
    "actions/temporal/source_data_activity": "edge",
    "actions/temporal/measurement_activities": "edge",
    "actions/temporal/llm_subroutine_activities": "edge",
    "actions/temporal/llm_tool_adapters": "edge",
    "utils/harness/stream_json": "edge",
    "actions/validation/checks": "compiler",
    "actions/validation/rules": "compiler",
    "compilation_errors": "domain",
    "compute/batching": "compiler",
    "compute/resources": "edge",
    "distributions": "domain",
    "json_types": "domain",
    "llm_specs": "domain",
    "measurement_types": "domain",
    "models/posterior_predictive": "execution",
    "models/predictive_simulation": "execution",
    "models/ssm/constants": "domain",
    "models/ssm/counterfactual/orchestration": "compiler",
    "models/ssm/dynamics/linearisation": "compiler",
    "models/ssm/dynamics/serialization": "domain",
    "models/ssm/dynamics/spec": "domain",
    "models/ssm/inference/methods/marginal_particle_gibbs/_context": "compiler",
    "models/ssm/inference/persistence": "edge",
    "models/ssm/joint_layout": "domain",
    "models/ssm/numerics": "compiler",
    "models/ssm/observation_support": "compiler",
    "models/ssm/parameterization": "compiler",
    "models/ssm/predictive/types": "domain",
    "models/ssm/preflight": "compiler",
    "models/ssm/priors": "compiler",
    "models/ssm/reachability": "compiler",
    "models/ssm/runtime": "compiler",
    "models/ssm/shapes": "domain",
    "models/ssm/transition_kinds": "domain",
    "numpyro_json": "domain",
    "prior_distributions": "domain",
    "publish": "edge",
    "read_facade": "edge",
    "sampler_config": "domain",
    "actions/simulation_summaries": "projection",
    "study/errors": "domain",
    "study/artifact_files": "domain",
    "study/prior_views": "projection",
    "study/records": "domain",
    "study/snapshot_models": "domain",
    "study/state": "domain",
    "study/view_models": "domain",
    "study/visuals": "projection",
    "study_api": "edge",
    "tool_server": "edge",
    "utils/agent_session": "edge",
    "utils/aggregations": "compiler",
    "utils/arrays": "edge",
    "utils/causal_design": "compiler",
    "utils/config": "edge",
    "utils/content_cache": "shell",
    "utils/data": "edge",
    "utils/histograms": "execution",
    "utils/identifiability": "compiler",
    "utils/model_structure": "compiler",
    "utils/observation_rows": "compiler",
    "utils/openrouter_client": "edge",
    "utils/storage": "edge",
    "workers/context": "domain",
    "workers/messages": "domain",
    "workers/schemas": "domain",
    "workers/windows": "compiler",
}

# Annotations cover every non-numerical role and the published numerical seams.
ANNOTATION_ROLES: frozenset[ModuleRole] = frozenset(
    {"domain", "projection", "compiler", "edge", "shell"}
)
ANNOTATION_MODULES = frozenset(
    {
        "models/posterior_predictive",
        "models/predictive_simulation",
        "models/ssm/execution/contracts",
        "models/ssm/execution/dynamical_model",
        "models/ssm/inference/types",
        "models/ssm/inference/problem",
        "models/ssm/inference/utils",
        "models/ssm/inference/parameter_transform",
        "models/ssm/inference/methods/marginal_particle_gibbs/smoothers/dsmc",
        "models/ssm/inference/shared",
        "models/ssm/inference/mcmc_state",
        "models/ssm/inference/methods/marginal_particle_gibbs/fit",
        "models/ssm/inference/methods/marginal_particle_gibbs/runner",
        "models/ssm/inference/targets/laplace/__init__",
        "models/ssm/inference/warmup/parameter_warmup",
        "models/ssm/inference/warmup/map",
        "models/ssm/inference/warmup/scipy_pathfinder",
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
