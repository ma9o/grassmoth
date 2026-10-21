"""Run Vulture with independent ownership seams.

vulture only scans .py files. This wrapper extracts code cells from every
notebook under ``notebooks/`` into a temporary cache of ``.py`` shadows. The
authoritative source pass scans ``[tool.vulture].paths`` and the maintained
evaluation application that consumes its public library. Evaluation code,
scripts, notebooks, and tests each run in a separate Vulture process that may
analyze source code to understand the protocols it consumes but reports only
definitions owned by that seam. The five passes run concurrently with separate
reference caches; their output retains ownership order. Tests, scripts and
notebooks cannot make source definitions live; the library's maintained callers
can. References cannot leak between non-source seams.

Two kinds of phantom usage are emitted into the cache so vulture stops
flagging legitimate-but-statically-invisible references:

* Identifiers wrapped in backticks inside notebook markdown cells (e.g.
  ``foo_bar``) — covers documented swap-in hooks.
* Identifiers inside string-quoted type expressions — ``cast("X")``,
  ``Annotated["X", ...]``, forward annotations like ``def foo() -> "X"``.
  Vulture treats strings as opaque, so these references are otherwise
  invisible.

Exported field declarations are handed to FIELD003 by their source location;
their names never become phantom references. Internal dataclass, TypedDict,
NamedTuple and Protocol fields retain Vulture's normal detection. String-key
subscript reads count as actual references, including TypedDict readers.

Additionally, vulture's built-in treatment of ``__all__`` entries as "uses" is
disabled (see ``_run_vulture``) so that symbols which are only ever re-exported
from a package ``__init__`` — but never actually referenced — are reported as
dead instead of being kept alive by their ``__all__`` listing.

Usage:
    cd apps/data-pipeline
    uv run python scripts/checks/run_vulture.py                # standard run
"""

from __future__ import annotations

import ast
import json
import keyword
import os
import re
import shutil
import sys
import tempfile
import tomllib
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from multiprocessing import get_context
from pathlib import Path

import vulture.core as vulture_core

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
CACHE_ROOT = REPO_ROOT / ".vulture_cache"
SEAM_PATHS = ("evaluation", "scripts", "notebooks", "tests")
PYPROJECT = REPO_ROOT / "pyproject.toml"

BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

TYPING_CAST_FUNCS = {"cast", "assert_type", "reveal_type"}


@dataclass(frozen=True, slots=True)
class VulturePassResult:
    exit_code: int
    stdout: str
    stderr: str


def _extract_backtick_idents(text: str) -> set[str]:
    refs: set[str] = set()
    for span in BACKTICK_SPAN.findall(text):
        refs.update(IDENT.findall(span))
    return refs


def _marimo_markdown_refs() -> set[str]:
    """Backtick identifiers inside marimo notebook string literals.

    marimo notebooks are plain ``.py`` files whose prose lives in ``mo.md(...)``
    string literals. Track backtick identifiers so documented swap-in hooks —
    e.g. an alternative sampler named only in markdown — are not reported dead.
    The backtick regex only matches `` `delimited` `` spans, so ordinary display
    strings (plot titles, footers) contribute nothing.
    """
    refs: set[str] = set()
    for py_path in NOTEBOOKS_DIR.rglob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "import marimo" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                refs.update(_extract_backtick_idents(node.value))
    return refs


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return None


def _collect_exported_field_locations(paths: list[Path]) -> frozenset[tuple[Path, int]]:
    """Suppress exported declarations by location, without keeping their names live."""
    classes_by_file: dict[Path, list[ast.ClassDef]] = {}
    trees: dict[Path, ast.Module] = {}
    for root in paths:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            classes_by_file[py_path] = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            trees[py_path] = tree

    declarations: dict[str, dict[str, tuple[Path, ast.AnnAssign]]] = {}
    bases: dict[str, tuple[str, ...]] = {}
    imported: dict[str, str] = {}
    for path, classes in classes_by_file.items():
        if not path.is_relative_to(REPO_ROOT / "src"):
            continue
        module = ".".join(path.relative_to(REPO_ROOT / "src").with_suffix("").parts)
        module = module.removesuffix(".__init__")
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        bindings: dict[str, str] = {}
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parts = package.split(".")
                    prefix = ".".join(parts[: len(parts) - node.level + 1])
                    if node.module:
                        prefix += "." + node.module
                for alias in node.names:
                    bindings[alias.asname or alias.name] = f"{prefix}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )

        def resolve(
            node: ast.AST, bindings: dict[str, str] = bindings, module: str = module
        ) -> str:
            if isinstance(node, ast.Subscript):
                return resolve(node.value)
            if isinstance(node, ast.Name):
                return bindings.get(node.id, f"{module}.{node.id}")
            if isinstance(node, ast.Attribute):
                return f"{resolve(node.value)}.{node.attr}"
            return ""

        imported.update((f"{module}.{name}", target) for name, target in bindings.items())
        for node in trees[path].body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name | ast.Attribute | ast.Subscript)
            ):
                imported[f"{module}.{node.targets[0].id}"] = resolve(node.value)
        for cls in classes:
            identity = f"{module}.{cls.name}"
            owned = declarations.setdefault(identity, {})
            bases[identity] = tuple(resolve(base) for base in cls.bases)
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    owned[stmt.target.id] = path, stmt
    if not declarations:
        return frozenset()
    api = json.loads((REPO_ROOT.parents[1] / "packages/api-types/schemas/openapi.json").read_text())
    generics = {
        ref["$ref"].rsplit("/", 1)[-1]: name
        for name, ref in api.get("x-typescript-generics", {}).items()
    }
    changed = True
    while changed:
        changed = False
        for identity, parents in bases.items():
            for parent in parents:
                seen: set[str] = set()
                while parent in imported and parent not in seen:
                    seen.add(parent)
                    parent = imported[parent]
                for name, declaration in tuple(declarations.get(parent, {}).items()):
                    if name not in declarations[identity]:
                        declarations[identity][name] = declaration
                        changed = True
    exported: dict[str, set[str]] = {}
    for component, schema in api["components"]["schemas"].items():
        module = schema.get("x-python-module")
        if module is None:
            continue
        name = generics.get(
            component,
            schema.get("x-typescript-type", schema.get("title", ""))
            .split("<", 1)[0]
            .split("[", 1)[0]
            .removesuffix("-Input")
            .removesuffix("-Output"),
        )
        exported.setdefault(f"{module}.{name}", set()).update(schema.get("properties", {}))
    locations: set[tuple[Path, int]] = set()
    for identity, fields in declarations.items():
        names = exported.get(identity, set())
        for name, (path, stmt) in fields.items():
            aliases = {name}
            metadata = [stmt.value]
            if (
                isinstance(stmt.annotation, ast.Subscript)
                and _base_name(stmt.annotation.value) == "Annotated"
                and isinstance(stmt.annotation.slice, ast.Tuple)
            ):
                metadata.extend(stmt.annotation.slice.elts[1:])
            for node in metadata:
                if isinstance(node, ast.Call):
                    aliases.update(
                        keyword.value.value
                        for keyword in node.keywords
                        if keyword.arg in {"alias", "serialization_alias", "validation_alias"}
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    )
            if aliases & names:
                locations.add((path.resolve(), stmt.lineno))
    return frozenset(locations)


def _collect_subscript_reads(paths: list[str]) -> set[str]:
    """Count real string-key and declared TypedDict iteration reads.

    An annotated record alone contributes nothing. Its fields become used only
    when code actually iterates that record's items/values, or reads a string key.
    """
    refs: set[str] = set()
    trees = [
        ast.parse(path.read_text()) for root in paths for path in (REPO_ROOT / root).rglob("*.py")
    ]
    dictionaries = {
        node.name: {
            field.target.id
            for field in node.body
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
        }
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(_base_name(base) == "TypedDict" for base in node.bases)
    }
    for tree in trees:
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def scope(
            node: ast.AST, parents: dict[ast.AST, ast.AST] = parents, tree: ast.AST = tree
        ) -> ast.AST:
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    return node
            return tree

        def in_annotation(node: ast.AST, parents: dict[ast.AST, ast.AST] = parents) -> bool:
            while node in parents:
                parent = parents[node]
                if (
                    (isinstance(parent, ast.AnnAssign | ast.arg) and parent.annotation is node)
                    or (
                        isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef)
                        and parent.returns is node
                    )
                    or isinstance(parent, ast.TypeAlias)
                ):
                    return True
                node = parent
            return False

        annotations = {
            (scope(node), node.target.id): _base_name(node.annotation)
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        annotations.update(
            ((scope(node), node.arg), _base_name(node.annotation))
            for node in ast.walk(tree)
            if isinstance(node, ast.arg) and node.annotation is not None
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and not in_annotation(node)
            ):
                refs.add(node.slice.value)
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                refs.add(node.args[0].value)
            if node.func.attr in {"items", "values"} and isinstance(node.func.value, ast.Name):
                dictionary = annotations.get((scope(node), node.func.value.id))
                refs.update(dictionaries.get(dictionary, ()))
    return refs


def _scan_string_type_node(node: ast.AST | None, refs: set[str]) -> None:
    if node is None:
        return
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            refs.update(IDENT.findall(sub.value))


def _collect_string_type_refs(paths: list[Path]) -> set[str]:
    """Extract identifiers from string-quoted type positions (cast/Annotated/forward annotations)."""
    refs: set[str] = set()
    for root in paths:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    _scan_string_type_node(node.returns, refs)
                    args = node.args
                    for arg in (*args.args, *args.posonlyargs, *args.kwonlyargs):
                        _scan_string_type_node(arg.annotation, refs)
                elif isinstance(node, ast.AnnAssign):
                    _scan_string_type_node(node.annotation, refs)
                elif isinstance(node, ast.Call):
                    func = node.func
                    func_name = None
                    if isinstance(func, ast.Name):
                        func_name = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name = func.attr
                    if func_name in TYPING_CAST_FUNCS and node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            refs.update(IDENT.findall(first_arg.value))
    return refs


def _write_phantom(cache_dir: Path, filename: str, refs: set[str]) -> None:
    if not refs:
        return
    body = "\n".join(f"_ = {name}" for name in sorted(refs) if not keyword.iskeyword(name))
    (cache_dir / filename).write_text(body + "\n")


def _collect_top_level_defs(roots: list[str]) -> dict[str, list[tuple[str, int, str]]]:
    """Map name -> [(file, lineno, kind)] for module-level bindings under ``roots``.

    Only direct children of a module are recorded (no nested/local defs), since
    those are what vulture's flat name matching confuses across files. ``kind`` is
    "def" for top-level functions/classes and "assign" for module-level name
    bindings (the latter catches factory assignments like ``foo = make_task(...)``).
    """
    found: dict[str, list[tuple[str, int, str]]] = {}
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py_path in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                targets: list[tuple[str, str]] = []
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    targets.append((node.name, "def"))
                elif isinstance(node, ast.Assign):
                    targets += [(t.id, "assign") for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    targets.append((node.target.id, "assign"))
                for name, kind in targets:
                    found.setdefault(name, []).append((str(py_path), node.lineno, kind))
    return found


def _warn_name_collisions(vulture, defs: dict[str, list[tuple[str, int, str]]]) -> None:
    """Warn about referenced names with module-level definitions in >1 file.

    Vulture's unused check is ``item.name not in used_names`` against one flat set,
    so a single use of a name shields *every* same-named definition across all
    modules. We surface names that (a) are bound at module level in more than one
    file with at least one being a function/class, and (b) are referenced somewhere
    — vulture reports none of them as unused, yet one may be dead. Advisory; this is
    an irreducible consequence of vulture's name matching (vulture #366 / #271).
    """
    flagged = 0
    for name in sorted(defs):
        sites = defs[name]
        files = {f for f, _, _ in sites}
        if len(files) < 2 or name not in vulture.used_names:
            continue
        if not any(kind == "def" for _, _, kind in sites):
            continue  # assign-only collisions (constants) are noisy and rarely API
        flagged += 1
        joined = ", ".join(f"{f}:{ln}" for f, ln, _ in sorted(sites))
        print(
            f"name-collision: '{name}' bound at module level in {len(files)} files "
            f"({joined}) and is referenced — vulture cannot tell which are live, so a "
            f"dead one is masked.",
            file=sys.stderr,
        )
    if flagged:
        print(
            f"\n{flagged} top-level name collision(s) may hide dead code that vulture's "
            f"name matching cannot detect (vulture #366/#271) — review manually.",
            file=sys.stderr,
        )


def _is_under_roots(filename: str | Path, roots: list[str]) -> bool:
    path = Path(filename)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    return any(resolved.is_relative_to((REPO_ROOT / root).resolve()) for root in roots)


def _report_vulture(
    vulture: vulture_core.Vulture,
    *,
    min_confidence: int,
    sort_by_size: bool,
    make_whitelist: bool,
    report_roots: list[str],
    exported_fields: frozenset[tuple[Path, int]],
) -> int:
    """Report only definitions owned by ``report_roots`` from a wider analysis."""
    exit_code = int(vulture.exit_code)
    for item in vulture.get_unused_code(
        min_confidence=min_confidence,
        sort_by_size=sort_by_size,
    ):
        if not _is_under_roots(item.filename, report_roots):
            continue
        if (Path(item.filename).resolve(), item.first_lineno) in exported_fields:
            continue
        print(
            item.get_whitelist_string()
            if make_whitelist
            else item.get_report(add_size=sort_by_size)
        )
        exit_code = int(vulture_core.ExitCode.DeadCode)
    return exit_code


def _run_vulture(
    argv: list[str],
    *,
    report_roots: list[str],
    collision_roots: list[str] | None,
) -> int:
    """Run vulture in-process with three local adjustments.

    1. ``__all__`` entries are NOT counted as uses, so re-exported-but-unused
       symbols surface (vulture otherwise treats every ``__all__`` name as a use
       via ``core.visit_Assign`` -> ``_assigns_special_variable__all__``).
    2. After the normal report, warn about referenced names defined in more than
       one file (see ``_warn_name_collisions``).
    3. A seam pass may analyze source code while reporting only definitions owned
       by that seam. Source liveness includes the maintained library callers.

    Replicates ``vulture.core.main`` rather than calling it, so we hold the
    ``Vulture`` instance for the collision pass; config discovery, exit codes, and
    ``--make-whitelist`` output are unchanged. Coupled to vulture internals
    (pinned via ``vulture>=2.14``).
    """
    # setattr (not direct assignment) so vulture doesn't read this monkeypatch as a
    # defined-but-unused module attribute and flag our own override as dead code.
    setattr(vulture_core, "_assigns_special_variable__all__", lambda _node: False)  # noqa: B010 - direct assignment makes Vulture flag its own monkeypatch as an unused module attribute
    saved_argv, saved_cwd = sys.argv, Path.cwd()
    sys.argv = ["vulture", *argv]
    os.chdir(REPO_ROOT)
    try:
        try:
            config = vulture_core.make_config()
        except vulture_core.InputError as exc:
            print(exc, file=sys.stderr)
            return int(vulture_core.ExitCode.InvalidCmdlineArguments)
        vulture = vulture_core.Vulture(
            verbose=config["verbose"],
            ignore_names=config["ignore_names"],
            ignore_decorators=config["ignore_decorators"],
        )
        vulture.scavenge(config["paths"], exclude=config["exclude"])
        vulture.used_names.update(_collect_subscript_reads(config["paths"]))
        exit_code = _report_vulture(
            vulture,
            min_confidence=config["min_confidence"],
            sort_by_size=config["sort_by_size"],
            make_whitelist=config["make_whitelist"],
            report_roots=report_roots,
            exported_fields=_collect_exported_field_locations(
                [Path(path) for path in config["paths"]]
            ),
        )
        if not config["make_whitelist"] and collision_roots:
            _warn_name_collisions(vulture, _collect_top_level_defs(collision_roots))
        return int(exit_code)
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def _cache_path(cache_dir: Path) -> str:
    return str(cache_dir.relative_to(REPO_ROOT))


def _build_phantom_refs(cache_dir: Path, paths: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in paths]
    _write_phantom(
        cache_dir,
        "_string_type_refs.py",
        _collect_string_type_refs(source_paths),
    )


def _aggregate_exit_code(exit_codes: list[int]) -> int:
    for code in exit_codes:
        if code in {
            int(vulture_core.ExitCode.InvalidInput),
            int(vulture_core.ExitCode.InvalidCmdlineArguments),
        }:
            return code
    if int(vulture_core.ExitCode.DeadCode) in exit_codes:
        return int(vulture_core.ExitCode.DeadCode)
    return int(vulture_core.ExitCode.NoDeadCode)


def _run_pass(
    seam_path: str | None,
    *,
    source_paths: list[str],
    cache_dir: Path,
    argv: list[str],
) -> VulturePassResult:
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        analysis_paths = (
            [*source_paths, seam_path] if seam_path is not None else [*source_paths, "evaluation"]
        )
        pass_cache_dir = cache_dir / (seam_path if seam_path is not None else "src")
        _build_phantom_refs(pass_cache_dir, analysis_paths)
        report_roots = [seam_path] if seam_path is not None else source_paths
        if seam_path == "notebooks":
            markdown_refs = _marimo_markdown_refs()
            _write_phantom(pass_cache_dir, "_notebook_markdown_refs.py", markdown_refs)
            report_roots.append(_cache_path(pass_cache_dir))
        exit_code = _run_vulture(
            [*analysis_paths, _cache_path(pass_cache_dir), *argv],
            report_roots=report_roots,
            collision_roots=source_paths if seam_path is None else None,
        )
    return VulturePassResult(exit_code, stdout.getvalue(), stderr.getvalue())


def main() -> int:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="run-", dir=CACHE_ROOT))
    try:
        config = tomllib.loads(PYPROJECT.read_text())
        source_paths = config["tool"]["vulture"]["paths"]
        passes = (None, *SEAM_PATHS)
        # Vulture mutates sys.argv, cwd and module globals, so passes need processes.
        with ProcessPoolExecutor(max_workers=len(passes), mp_context=get_context("spawn")) as pool:
            futures = [
                pool.submit(
                    _run_pass,
                    seam_path,
                    source_paths=source_paths,
                    cache_dir=cache_dir,
                    argv=sys.argv[1:],
                )
                for seam_path in passes
            ]
            results = [future.result() for future in futures]

        for result in results:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
        return _aggregate_exit_code([result.exit_code for result in results])
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
