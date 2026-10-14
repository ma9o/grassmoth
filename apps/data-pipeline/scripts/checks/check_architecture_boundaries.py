"""Enforce the role-derived dependency rules of the architecture.

Static package seams are import-linter contracts in ``pyproject.toml``. The rules
here follow the module roles owned by ``architecture_roles``:

``ARCH007``
    Numerical execution signatures accept compiled models, bound observations
    and the owned resolved SamplerSpec, never authoring or transport inputs,
    partial configurations or forwarding TypedDicts.

``ARCH008``
    Pure roles never reach edge or shell modules, I/O, clock or environment
    acquisition, directly or transitively.

``ARCH009``
    Pure roles cannot call known input/output, filesystem, clock or ambient-randomness
    primitives, including imported and assigned aliases. Date arithmetic and
    explicitly seeded local generators are allowed. This checks primitive calls,
    not arbitrary callback effects or ownership of mutable generator arguments.

``ARCH010``
    Runtime imports between production modules form a directed acyclic graph.
    Shared values and helpers belong below their consumers, never behind a cycle.

Imports guarded by ``TYPE_CHECKING`` are excluded because these rules constrain
runtime ownership and initialization, not type annotation dependencies.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override

import grimp
import networkx as nx

from scripts.checks.architecture_roles import (
    PACKAGE,
    PURE_ROLES,
    fix_owner,
    role_for_module,
    role_inventory,
)

_PACKAGE = PACKAGE
_INPUT_CALLS = frozenset(
    {
        "builtins.open",
        "builtins.input",
        "builtins.print",
        "io.open",
        "codecs.open",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.date.today",
        "uuid.uuid1",
        "uuid.uuid4",
    }
)
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath"})
_PATH_EFFECTS = frozenset(
    {
        "absolute",
        "chmod",
        "cwd",
        "exists",
        "expanduser",
        "glob",
        "home",
        "is_block_device",
        "is_char_device",
        "is_dir",
        "is_fifo",
        "is_file",
        "is_junction",
        "is_mount",
        "is_socket",
        "is_symlink",
        "iterdir",
        "lchmod",
        "lstat",
        "mkdir",
        "open",
        "owner",
        "group",
        "read_bytes",
        "read_text",
        "readlink",
        "rename",
        "replace",
        "resolve",
        "rglob",
        "rmdir",
        "samefile",
        "stat",
        "symlink_to",
        "hardlink_to",
        "touch",
        "unlink",
        "walk",
        "write_bytes",
        "write_text",
    }
)
_PATH_BUILDERS = frozenset({"joinpath", "with_name", "with_stem", "with_suffix"})
_RANDOM_FACTORIES = {
    "random.Random": "x",
    "numpy.random.default_rng": "seed",
    "numpy.random.RandomState": "seed",
    "numpy.random.SeedSequence": "entropy",
    "numpy.random.MT19937": "seed",
    "numpy.random.PCG64": "seed",
    "numpy.random.PCG64DXSM": "seed",
    "numpy.random.Philox": "seed",
    "numpy.random.SFC64": "seed",
}


def _scope_bindings(body: list[ast.stmt]) -> dict[str, str]:
    """Collect lexical bindings without entering nested scopes."""
    names: set[str] = set()
    imports: dict[str, str] = {}
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            continue
        if isinstance(
            node, ast.Lambda | ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
        ):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    imports[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
                else:
                    imports[alias.asname or alias.name] = (node.module or "") + "." + alias.name
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        pending.extend(ast.iter_child_nodes(node))
    return dict.fromkeys(names, "") | imports


class _RuntimeEffects(ast.NodeVisitor):
    """Resolve primitive origins within lexical scopes without importing src."""

    def __init__(self, tree: ast.Module) -> None:
        self.scopes: list[dict[str, str]] = [_scope_bindings(tree.body)]
        self.class_scopes: list[dict[str, str]] = []
        self.calls: list[tuple[ast.Call, str]] = []

    def _origin(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope:
                    return scope[node.id]
            return "builtins." + node.id if node.id in {"open", "input", "print"} else ""
        if isinstance(node, ast.Attribute):
            owner = self._origin(node.value)
            if owner in _PATH_TYPES and node.attr == "parent":
                return owner
            return owner + "." + node.attr if owner else ""
        if isinstance(node, ast.Call):
            origin = self._origin(node.func)
            owner, _, method = origin.rpartition(".")
            if origin in _PATH_TYPES:
                return origin
            if owner in _PATH_TYPES and method in _PATH_BUILDERS:
                return owner
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            owner = self._origin(node.value.value)
            if owner in _PATH_TYPES and node.value.attr == "parents":
                return owner
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            owner = self._origin(node.left)
            return owner if owner in _PATH_TYPES else ""
        return ""

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.scopes[-1][alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.scopes[-1][alias.asname or alias.name] = (node.module or "") + "." + alias.name

    @override
    def visit_If(self, node: ast.If) -> None:
        if self._origin(node.test) == "typing.TYPE_CHECKING":
            for statement in node.orelse:
                self.visit(statement)
            return
        if (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and self._origin(node.test.operand) == "typing.TYPE_CHECKING"
        ):
            for statement in node.body:
                self.visit(statement)
            return
        self.generic_visit(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        if not isinstance(node, ast.Lambda):
            for decorator in node.decorator_list:
                self.visit(decorator)
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        )
        bindings = {} if isinstance(node, ast.Lambda) else _scope_bindings(node.body)
        for argument in arguments:
            if argument is not None:
                origin = (
                    self._origin(argument.annotation) if argument.annotation is not None else ""
                )
                bindings[argument.arg] = origin if origin in _PATH_TYPES else ""
        enclosing = self.scopes
        self.scopes = [
            scope
            for scope in enclosing
            if all(scope is not class_scope for class_scope in self.class_scopes)
        ] + [bindings]
        if isinstance(node, ast.Lambda):
            self.visit(node.body)
        else:
            for statement in node.body:
                self.visit(statement)
        self.scopes = enclosing

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._function(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        bindings = _scope_bindings(node.body)
        self.scopes.append(bindings)
        self.class_scopes.append(bindings)
        for statement in node.body:
            self.visit(statement)
        self.class_scopes.pop()
        self.scopes.pop()

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        self.visit(node.generators[0].iter)
        self.scopes.append({})
        for position, generator in enumerate(node.generators):
            if position:
                self.visit(generator.iter)
            for name in ast.walk(generator.target):
                if isinstance(name, ast.Name):
                    self.scopes[-1][name.id] = ""
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node)

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node)

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scopes[-1][target.id] = self._origin(node.value)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            annotation = self._origin(node.annotation)
            value = self._origin(node.value) if node.value is not None else ""
            self.scopes[-1][node.target.id] = annotation if annotation in _PATH_TYPES else value

    @override
    def visit_Call(self, node: ast.Call) -> None:
        origin = self._origin(node.func)
        owner, _, method = origin.rpartition(".")
        forbidden = origin in _INPUT_CALLS or (owner in _PATH_TYPES and method in _PATH_EFFECTS)
        if origin in _RANDOM_FACTORIES:
            seed = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == _RANDOM_FACTORIES[origin]
                    ),
                    None,
                )
            )
            forbidden = seed is None or (isinstance(seed, ast.Constant) and seed.value is None)
        elif origin.startswith(("random.", "secrets.", "numpy.random.")):
            forbidden = origin != "numpy.random.Generator"
        if forbidden:
            self.calls.append((node, origin))
        self.generic_visit(node)


@dataclass(frozen=True, slots=True)
class ImportRef:
    """One runtime import between project modules."""

    path: Path
    line: int
    importer: str
    imported: str


@dataclass(frozen=True, slots=True)
class Violation:
    """One forbidden dependency."""

    ref: ImportRef
    code: str
    message: str

    def diagnostic(self, source_root: Path) -> str:
        path = self.ref.path.relative_to(source_root.parent).as_posix()
        role = role_for_module(self.ref.importer)
        owner_path = f"src/{self.ref.importer.replace('.', '/')}.py"
        return (
            f"{path}:{self.ref.line}: {self.code} {self.message} "
            f"[role={role}; fix at {fix_owner(owner_path)}]"
        )


def find_violations(source_root: Path) -> tuple[Violation, ...]:
    """Return every forbidden runtime dependency below ``source_root``."""
    violations: list[Violation] = []

    # Resolve signatures by type ownership, including aliases and renamed arguments.
    # Raw data/configuration is bound before execution; native numerical leaves remain valid.
    from scripts.checks.architecture_roles import role_for_path
    from scripts.checks.check_type_boundaries import TypeIndex

    inventory = role_inventory(source_root)
    sources = [
        (
            "src/" + _PACKAGE + "/" + path.relative_to(source_root).as_posix(),
            ast.parse(path.read_text(encoding="utf-8")),
        )
        for path in inventory
    ]
    index = TypeIndex(sources)
    raw = {
        f"{_PACKAGE}.artifacts.model.dynamical_model_spec.DynamicalModelSpec",
        f"{_PACKAGE}.artifacts.model.likelihood.ObservationLawSpec",
        f"{_PACKAGE}.artifacts.model.likelihood.LikelihoodSpec",
        f"{_PACKAGE}.artifacts.sampling.FitSettingsSpec",
        "polars.DataFrame",
        "pandas.DataFrame",
        "pyarrow.Table",
    }

    def unresolved(node: ast.expr, module: str, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                expression = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return False  # Jaxtyping dimension strings are metadata, not Python types.
            return unresolved(expression, module, seen)
        name = index.resolve(node, module)
        if isinstance(node, ast.Subscript) and name.rsplit(".", 1)[-1] == "Literal":
            return False
        if (
            isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1] == "Annotated"
            and isinstance(node.slice, ast.Tuple)
        ):
            return unresolved(node.slice.elts[0], module, seen)
        if name in raw:
            return True
        if name in seen:
            return False
        if name in index.aliases:
            return unresolved(index.aliases[name], name.rpartition(".")[0], seen | {name})
        if name in index.classes:
            owner, declaration = index.classes[name]
            if role_for_path(owner) in {"edge", "shell"}:
                return True
            if any(
                keyword.arg == "total"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in declaration.keywords
            ):
                return True
            return any(
                unresolved(base, name.rpartition(".")[0], seen | {name})
                for base in declaration.bases
            )
        if (
            isinstance(node, ast.Subscript)
            and name.rsplit(".", 1)[-1]
            in {
                "dict",
                "Dict",
                "Mapping",
                "MutableMapping",
            }
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            value = node.slice.elts[1]
            if index.resolve(value, module).rsplit(".", 1)[-1] in {"Any", "object"}:
                return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.BitOr)
            and any(
                isinstance(part, ast.Constant) and part.value is None for part in ast.walk(node)
            )
            and f"{_PACKAGE}.artifacts.sampling.SamplerSpec" in index.return_variants(node, module)
        ):
            return True
        return any(
            unresolved(child, module, seen)
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.expr)
        )

    for (relative, tree), (path, role) in zip(sources, inventory.items(), strict=True):
        importer = index.module(relative)
        if role in PURE_ROLES:
            effects = _RuntimeEffects(tree)
            effects.visit(tree)
            violations.extend(
                Violation(
                    ImportRef(path, call.lineno, importer, primitive),
                    "ARCH009",
                    f"pure roles cannot perform runtime effects through {primitive}; "
                    "acquire it at an edge or shell and pass the owned value explicitly",
                )
                for call, primitive in effects.calls
            )
        if role != "execution":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if argument.annotation is not None and unresolved(argument.annotation, importer):
                    violations.append(
                        Violation(
                            ImportRef(
                                path, argument.lineno, importer, ast.unparse(argument.annotation)
                            ),
                            "ARCH007",
                            "execution requires compiled models, bound observations and resolved sampler values; bind authoring/transport inputs at their compiler or shell owner",
                        )
                    )

    # grimp owns runtime import discovery, local imports and transitive traversal.
    # A caller-supplied source root is also used by isolated checker regressions.
    previous_package = sys.modules.pop(_PACKAGE, None)
    sys.path.insert(0, str(source_root.parent.resolve()))
    importlib.invalidate_caches()
    try:
        graph = grimp.build_graph(
            _PACKAGE,
            include_external_packages=True,
            exclude_type_checking_imports=True,
            cache_dir=None,
        )
    finally:
        sys.path.pop(0)
        if previous_package is not None:
            sys.modules[_PACKAGE] = previous_package

    modules = {
        _PACKAGE
        + (
            "."
            + path.relative_to(source_root)
            .with_suffix("")
            .as_posix()
            .removesuffix("/__init__")
            .replace("/", ".")
            if path != source_root / "__init__.py"
            else ""
        ): path
        for path in inventory
    }
    dependencies = nx.DiGraph()
    dependencies.add_nodes_from(modules)
    dependencies.add_edges_from(
        (importer, imported)
        for importer in modules
        for imported in graph.find_modules_directly_imported_by(importer)
        if imported in modules
    )
    for component in sorted(
        nx.strongly_connected_components(dependencies), key=lambda c: sorted(c)
    ):
        if len(component) == 1 and not any(dependencies.has_edge(m, m) for m in component):
            continue
        cycle = nx.find_cycle(dependencies.subgraph(sorted(component)), source=min(component))
        importer, imported = cycle[0]
        details = graph.get_import_details(importer=importer, imported=imported)
        chain = [edge[0] for edge in cycle] + [cycle[0][0]]
        violations.append(
            Violation(
                ImportRef(modules[importer], details[0]["line_number"], importer, imported),
                "ARCH010",
                "runtime imports must be acyclic; move shared ownership below its consumers: "
                + " -> ".join(chain),
            )
        )

    acquisition = {
        "os",
        "time",
        "socket",
        "subprocess",
        "tempfile",
        "shutil",
        "fsspec",
        "httpx",
        "openai",
        "pygit2",
        "dotenv",
        "uvicorn",
        "fastapi",
        "temporalio",
        "mcp",
        "modal",
        "exa_py",
        "botocore",
        "zipfile",
    }
    for importer, path in sorted(modules.items()):
        if role_for_module(importer) not in PURE_ROLES:
            continue
        for imported in sorted(graph.find_upstream_modules(importer)):
            if imported in acquisition or (
                imported in modules and role_for_module(imported) in {"edge", "shell"}
            ):
                chain = graph.find_shortest_chain(importer, imported)
                assert chain is not None
                details = graph.get_import_details(importer=chain[0], imported=chain[1])
                violations.append(
                    Violation(
                        ImportRef(path, details[0]["line_number"], importer, imported),
                        "ARCH008",
                        "pure roles cannot reach edge, shell, I/O, clock or environment "
                        f"acquisition: {' -> '.join(chain)}",
                    )
                )

    return tuple(violations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        nargs="?",
        type=Path,
        default=Path("src/nof1_causal_lab"),
    )
    args = parser.parse_args()

    role_inventory(args.source_root)
    violations = find_violations(args.source_root)
    for violation in violations:
        print(violation.diagnostic(args.source_root), file=sys.stderr)
    if violations:
        print(f"{len(violations)} architecture boundary violation(s)", file=sys.stderr)
        return 1

    print("Architecture boundaries: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
