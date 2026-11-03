"""Check and synchronize the library's pytest runtime shape instrumentation."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

# jaxtyping array/dtype annotation types whose per-call shape checks the
# ``jaxtyping_modules`` import hook installs. ``shapes.py`` re-exports these and
# is the canonical import source; ``FloatScalar`` is its local scalar alias.
_JAXTYPING_TYPES = frozenset(
    {
        "Array",
        "ArrayLike",
        "Bool",
        "Complex",
        "Float",
        "FloatScalar",
        "Inexact",
        "Int",
        "Integer",
        "Key",
        "Num",
        "PRNGKeyArray",
        "Real",
        "Scalar",
        "Shaped",
        "UInt",
    }
)
_SHAPES_MODULE = "grassmoth.models.ssm.shapes"

# (module below inference, qualified function or record, required annotation slots).
# ``return`` denotes the return annotation; the other slots are parameter names.
_REQUIRED_SHAPE_BOUNDARIES = (
    (
        "warmup.initialization",
        "ParticleInitialization",
        ("positions", "paths", "parameter_chol"),
    ),
    (
        "warmup.initialization",
        "prepare_particle_initialization.gaussian_initialization",
        ("draw_key", "return"),
    ),
    (
        "mcmc_state",
        "trajectory_state",
        (
            "position",
            "parameter_gradient",
            "latent_trajectory",
            "trajectory_log_prob",
            "complete_log_posterior",
            "latent_delta",
            "param_step_size",
            "parameter_chol",
        ),
    ),
    (
        "samplers.mpgibbs.runner",
        "_initialize_chain_state",
        (
            "init_position",
            "observations",
            "times",
            "initial_latent_delta",
            "parameter_gradient_fn",
            "parameter_chol",
            "initial_latent_trajectory",
        ),
    ),
    (
        "samplers.mpgibbs.kernel",
        "_bounded_parameter_drift",
        (
            "gradient",
            "chol",
            "scale",
            "return",
        ),
    ),
    (
        "samplers.mpgibbs.kernel",
        "build_marginal_particle_gibbs_kernel._theta_logpost_grad",
        (
            "z",
            "return",
        ),
    ),
    (
        "samplers.particle.context",
        "build_smoother_context",
        (
            "parameter_particles",
            "label_correction",
            "reference",
            "delta",
        ),
    ),
    ("samplers.particle.guided", "guided_step", ("key", "reference", "log_precisions")),
    ("samplers.particle.guided", "trace_guided_step", ("key", "reference", "log_precisions")),
    ("samplers.particle.twist", "future_information", ("matrices", "offsets", "covariances")),
    (
        "samplers.particle.adaptation",
        "_curvature_limits",
        ("observe", "path", "free_mask", "return"),
    ),
    ("samplers.particle.adaptation", "guided_delta_limit", ("path", "free_mask", "return")),
    ("samplers.particle.adaptation", "initialize_adaptation", ("delta",)),
    ("samplers.particle.adaptation", "adapt_scales", ("accepted", "proposed", "maximum")),
    ("warmup.parameter_adaptation", "initialize_parameter_covariance", ("positions", "chol")),
    (
        "warmup.parameter_adaptation",
        "adapt_parameter_covariance",
        (
            "positions",
            "chol",
            "iteration",
            "windows",
            "return",
        ),
    ),
)


def _function_signatures(tree: ast.AST, prefix: str = "") -> dict[str, dict[str, ast.expr | None]]:
    """Index nested call signatures and record fields without conflating names."""
    signatures: dict[str, dict[str, ast.expr | None]] = {}
    for node in ast.iter_child_nodes(tree):
        name = prefix
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = f"{prefix}.{node.name}" if prefix else node.name
        if isinstance(node, ast.ClassDef):
            signatures[name] = {
                field.target.id: field.annotation
                for field in node.body
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
            }
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            signatures[name] = {
                arg.arg: arg.annotation
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                )
                if arg is not None
            } | {"return": node.returns}
        signatures.update(_function_signatures(node, name))
    return signatures


def _missing_shape_slots(
    tree: ast.Module, function: str, slots: tuple[str, ...]
) -> tuple[str, ...]:
    """Require actual dtype/shape annotations, not a bare Array or a quoted hint."""
    shape_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in (_SHAPES_MODULE, "jaxtyping")
        for alias in node.names
        if alias.name in _JAXTYPING_TYPES - {"Array", "ArrayLike"}
    }
    signature = _function_signatures(tree).get(function, {})
    return tuple(
        slot
        for slot in slots
        if (annotation := signature.get(slot)) is None
        or not any(
            isinstance(node, ast.Name) and node.id in shape_names for node in ast.walk(annotation)
        )
    )


def _validate_required_shape_boundaries(package_root: Path) -> list[str]:
    """Keep critical state, gradient, transition and warmup boundaries instrumentable."""
    source = package_root / "src"
    trees: dict[str, ast.Module] = {}
    errors = []
    for relative_module, function, slots in _REQUIRED_SHAPE_BOUNDARIES:
        module = f"grassmoth.models.ssm.inference.{relative_module}"
        if module not in trees:
            path = source.joinpath(*module.split(".")).with_suffix(".py")
            if not path.is_file():
                errors.append(f"{module}: required shape boundary module is missing")
                continue
            trees[module] = ast.parse(path.read_text(encoding="utf-8"))
        missing = _missing_shape_slots(trees[module], function, slots)
        if missing:
            errors.append(
                f"{module}.{function}: required shape annotations missing for {', '.join(missing)}"
            )
    return errors


def _path_to_module(py_path: Path, src_root: Path) -> str:
    parts = list(py_path.relative_to(src_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_shape_names(tree: ast.Module) -> set[str]:
    """Local names bound to a jaxtyping shape type (via shapes or jaxtyping)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == _SHAPES_MODULE:
                names.update(alias.asname or alias.name for alias in node.names)
            elif node.module == "jaxtyping":
                names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in _JAXTYPING_TYPES
                )
    return names


def _has_jaxtyping_annotation(tree: ast.Module, shape_names: set[str]) -> bool:
    """True if any function signature in the module is annotated with a shape type."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        annotations = [
            arg.annotation
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if arg is not None and arg.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        for annotation in annotations:
            if any(
                isinstance(sub, ast.Name) and sub.id in shape_names for sub in ast.walk(annotation)
            ):
                return True
    return False


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _wired_jaxtyping_packages(pyproject: Path) -> set[str]:
    config = tomllib.loads(pyproject.read_text())
    return set(config["tool"]["pytest"]["ini_options"]["jaxtyping_modules"])


def _scan_jaxtyping_modules(src_root: Path) -> tuple[set[str], set[str]]:
    """Scan ``src/`` for modules with shape-annotated function signatures.

    Returns ``(annotated, future_disabled)``: modules whose function signatures use
    jaxtyping shape types, and the subset that also use ``from __future__ import
    annotations`` (which silently turns their runtime checks into no-ops).
    """
    annotated: set[str] = set()
    future_disabled: set[str] = set()
    for py_path in src_root.rglob("*.py"):
        module = _path_to_module(py_path, src_root)
        if module == _SHAPES_MODULE:
            continue  # the type-alias definitions module is not instrumented
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        shape_names = _imported_shape_names(tree)
        if shape_names and _has_jaxtyping_annotation(tree, shape_names):
            annotated.add(module)
            if _has_future_annotations(tree):
                future_disabled.add(module)
    return annotated, future_disabled


def validate_jaxtyping_wiring(package_root: Path) -> tuple[list[str], list[str]]:
    """Check that jaxtyping runtime checks actually cover the modules that need them.

    Returns ``(errors, notes)``. Errors (fatal) are the cases where shape checks
    silently do not run: a module annotated but missing from ``jaxtyping_modules``,
    or a listed module that disabled its checks via ``from __future__ import
    annotations``. Notes (non-fatal) are benign stale list entries. ``--fix``
    resolves the missing-module errors and the stale notes; the ``__future__`` case
    needs a source edit.
    """
    pyproject = package_root / "pyproject.toml"
    wired = _wired_jaxtyping_packages(pyproject)
    annotated, future_disabled = _scan_jaxtyping_modules(package_root / "src")

    errors = [
        f"{module}: uses jaxtyping shape annotations but is missing from "
        "jaxtyping_modules (shape checks silently skipped); run `bun run lint:fix`"
        for module in sorted(annotated - wired)
    ]
    errors += [
        f"{module}: instrumented but uses `from __future__ import annotations`, "
        "which makes its jaxtyping shape checks silently no-op"
        for module in sorted(future_disabled & wired)
    ]
    errors += _validate_required_shape_boundaries(package_root)
    notes = [
        f"{module}: listed in jaxtyping_modules but has no runtime-checkable "
        "jaxtyping annotations; `bun run lint:fix` will drop it"
        for module in sorted(wired - annotated)
    ]
    return errors, notes


def sync_jaxtyping_wiring(package_root: Path) -> str | None:
    """Rewrite ``jaxtyping_modules`` in pyproject.toml to the detected modules.

    Returns a summary of the change, or ``None`` if it was already in sync.
    Disabling annotations through ``from __future__ import annotations`` still
    requires a source edit; syncing cannot repair it.
    """
    pyproject = package_root / "pyproject.toml"
    annotated, _future_disabled = _scan_jaxtyping_modules(package_root / "src")
    desired = sorted(annotated)
    text = pyproject.read_text()
    match = re.search(r"jaxtyping_modules = \[.*?\]", text, re.DOTALL)
    if match is None:
        raise SystemExit("error: jaxtyping_modules missing from the library's pytest configuration")
    current_mods = _wired_jaxtyping_packages(pyproject)
    if current_mods == annotated:
        return None
    replacement = (
        "jaxtyping_modules = [\n" + "".join(f'    "{module}",\n' for module in desired) + "]"
    )
    pyproject.write_text(text[: match.start()] + replacement + text[match.end() :])
    changes = []
    if added := sorted(annotated - current_mods):
        changes.append("added " + ", ".join(added))
    if removed := sorted(current_mods - annotated):
        changes.append("removed " + ", ".join(removed))
    return "; ".join(changes) if changes else "reordered to canonical order"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parents[1]
    if args.fix:
        print(sync_jaxtyping_wiring(package_root) or "Jaxtyping wiring: already in sync")
        return 0
    errors, notes = validate_jaxtyping_wiring(package_root)
    for message in (*notes, *errors):
        print(message)
    print(f"Jaxtyping wiring: {len(errors)} error(s)")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
