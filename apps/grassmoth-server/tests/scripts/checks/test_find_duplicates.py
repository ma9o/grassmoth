"""Tests for the advisory local duplication audit."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract


def _load_checker() -> Any:
    module_name = "find_duplicates_under_test"
    path = Path(__file__).resolve().parents[3] / "scripts" / "checks" / "find_duplicates.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(repo_root: Path, relative_path: str, source: str) -> Path:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return Path(relative_path)


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_zero_context_diff_ranges_include_added_lines_and_deletion_anchor() -> None:
    checker = _load_checker()

    ranges = checker.parse_changed_ranges(
        """
@@ -4,2 +4,0 @@
@@ -10 +8,3 @@
@@ -20 +21 @@
"""
    )

    assert ranges == (
        checker.LineRange(4, 4),
        checker.LineRange(8, 10),
        checker.LineRange(21, 21),
    )


def test_default_selection_includes_tracked_hunks_and_untracked_sources(tmp_path: Path) -> None:
    checker = _load_checker()
    tracked_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/tracked.py",
        "def transform(value: int) -> int:\n    return value + 1\n",
    )
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "duplicate-audit@example.test")
    _git(tmp_path, "config", "user.name", "Duplicate Audit")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    _git(tmp_path, "branch", "audit-base")

    _write_source(
        tmp_path,
        tracked_path.as_posix(),
        "def transform(value: int) -> int:\n    return value + 2\n",
    )
    untracked_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/untracked.py",
        "def load() -> int:\n    return 1\n",
    )

    selection = checker.changed_selection(tmp_path, base_ref="audit-base")

    assert selection.includes(tracked_path.as_posix(), 2, 2)
    assert selection.includes(untracked_path.as_posix(), 1, 1000)
    assert selection.path_count == 2


def test_generic_family_replaces_weaker_schema_overlap_and_keeps_variants(tmp_path: Path) -> None:
    checker = _load_checker()
    runtime_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/runtime.py",
        """
from dataclasses import dataclass

@dataclass
class RuntimeConfig:
    model: str
    timeout: int | None = None
    effort: str | None = None
""",
    )
    transport_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/transport.py",
        """
from typing import Literal
from pydantic import BaseModel

class TransportConfig(BaseModel):
    model: str
    timeout: int | None = None
    effort: Literal["low", "high"] | None = None

class Revised:
    kind: Literal["revised"] = "revised"
    value: int

class Unchanged:
    kind: Literal["unchanged"] = "unchanged"
    value: int

class Box[T]:
    item: T

class Specialized(Box[str]):
    pass

class ThreeHolesA:
    a: int
    b: int
    c: int

class ThreeHolesB:
    a: str
    b: float
    c: bool

class CoupledA:
    action: Literal["a"] = "a"
    request: RequestA | None
    outcome: Envelope[ResultA]

class CoupledB:
    action: Literal["b"] = "b"
    request: RequestB | None
    outcome: Envelope[ResultB]

class ContainerA:
    entries: list[int]

class ContainerB:
    entries: list[str]

class DifferentContainer:
    entries: tuple[str, ...]

class BareA:
    kind: Literal["ref"] = "ref"
    value: FirstId

class BareB:
    kind: Literal["ref"] = "ref"
    value: SecondId

class PartialRootA:
    reasoning: str | None
    details: JsonValue

class PartialRootB:
    reasoning: NotRequired[str]
    details: NotRequired[JsonValue]
""",
    )
    selection = checker.SourceSelection(
        ranges={runtime_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="runtime config",
    )

    definitions = checker.collect_python_definitions(
        tmp_path,
        (runtime_path, transport_path),
    )
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    families = [candidate for candidate in candidates if candidate.category == "generic family"]
    assert len(families) == 1
    assert {families[0].first.label, families[0].second.label} == {
        "RuntimeConfig",
        "TransportConfig",
    }
    assert "1 annotation hole(s) [effort]" in families[0].reason
    assert not any(candidate.category == "class schema" for candidate in candidates)
    all_candidates = checker.ast_candidates(
        definitions, selection=checker.SourceSelection(ranges=None, description="all"), deep=False
    )
    all_families = [c for c in all_candidates if c.category == "generic family"]
    assert {frozenset((c.first.label, c.second.label)) for c in all_families} == {
        frozenset(("RuntimeConfig", "TransportConfig")),
        frozenset(("CoupledA", "CoupledB")),
        frozenset(("ContainerA", "ContainerB")),
    }
    assert "identical payload fields [model, timeout]" in families[0].reason
    coupled = next(c for c in all_families if c.first.label == "CoupledA")
    assert "2 annotation hole(s) [outcome, request]" in coupled.reason
    assert (
        "roots above every hole [outcome: subscript Envelope; request: union |]" in coupled.reason
    )


def test_identical_declared_fields_keep_tags_defaults_methods_and_skip_ancestry(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/signatures.py",
        """
from typing import Literal
from pydantic import Field

class Parent:
    value: list[int] = Field(default_factory=list)
    tag: Literal["open"] = "open"

class Child(Parent):
    value: list[int] = Field(default_factory=list)
    tag: Literal["open"] = "open"

class Grandchild(Child):
    value: list[int] = Field(default_factory=list)
    tag: Literal["open"] = "open"

class Mirror:
    value: list[int] = Field(default_factory=list)
    tag: Literal["open"] = "open"
    def close(self):
        return self.value

class OtherTag:
    value: list[int] = Field(default_factory=list)
    tag: Literal["closed"] = "closed"

class OtherDefault:
    value: list[int] = Field(default_factory=tuple)
    tag: Literal["open"] = "open"

class OtherContainer:
    value: tuple[int, ...] = Field(default_factory=list)
    tag: Literal["open"] = "open"
""",
    )
    ignored_path = _write_source(
        tmp_path,
        "apps/grassmoth-client/src/outside.py",
        "raise AssertionError('outside Python audit scope')\n",
    )
    selection = checker.SourceSelection(ranges=None, description="all")
    definitions = checker.collect_python_definitions(tmp_path, (path, ignored_path))
    candidates = checker.ast_candidates(definitions, selection=selection, deep=True)
    identical = [c for c in candidates if c.category == "identical fields"]

    assert len(identical) == 3
    assert all("Mirror" in {c.first.label, c.second.label} for c in identical)
    assert all(
        "Mirror [close]" in c.reason and "2 identical declared field signatures" in c.reason
        for c in identical
    )
    assert not any({c.first.label, c.second.label} == {"Parent", "Grandchild"} for c in candidates)
    report = checker.render_report(
        selection=selection, candidates=identical, shown=identical, definitions=definitions
    )
    assert "Identical declared field signatures" in report
    assert "%" not in report
    clone = dataclasses.replace(identical[0], category="token clone")
    assert checker._covered_by_type_candidate(clone, identical, definitions)
    outside = dataclasses.replace(clone, second=checker.Location("other.py", 1, 10))
    assert not checker._covered_by_type_candidate(outside, identical, definitions)


def test_repeated_row_counts_nested_copy_once_and_selects_changed_site(tmp_path: Path) -> None:
    checker = _load_checker()
    source = """
raise AssertionError("source collection must not import this module")

class Context:
    a: str
    b: str
    c: list[int]

class Envelope:
    a: str
    b: str
    c: list[int]
    extra: int

class TupleEnvelope:
    a: str
    b: str
    c: tuple[int, ...]
    extra: int

def outer(src: Context):
    def inner():
        return Envelope(a=src.a, b=src.b, c=src.c, extra=0)
    return inner()

def mixed(left, right):
    return Envelope(a=left.a, b=right.b, c=left.c, extra=0)
"""
    path = _write_source(tmp_path, "apps/grassmoth-server/src/domain/rows.py", source)
    line = (
        source.splitlines().index("        return Envelope(a=src.a, b=src.b, c=src.c, extra=0)") + 1
    )
    selection = checker.SourceSelection(
        ranges={path.as_posix(): (checker.LineRange(line, line),)}, description="copy site"
    )
    definitions = checker.collect_python_definitions(tmp_path, (path,))
    rows = [
        c
        for c in checker.ast_candidates(definitions, selection=selection, deep=False)
        if c.category == "repeated row"
    ]

    assert len(rows) == 1
    assert "1 distinct conversion site(s)" in rows[0].reason
    assert {rows[0].first.label, rows[0].second.label} == {"Context", "Envelope"}
    assert rows[0].first_selected is False
    assert rows[0].second_selected is False
    assert len(rows[0].evidence_locations) == 1
    assert "Context -> Envelope" in rows[0].evidence_locations[0].label
    assert "outer.inner" in rows[0].evidence_locations[0].label
    assert (
        checker.ast_candidates(
            definitions,
            selection=checker.SourceSelection(ranges={}, description="none"),
            deep=False,
        )
        == ()
    )
    _write_source(tmp_path, path.as_posix(), source.replace("c=src.c, extra=0", "c=src.c, extra=1"))
    changed = checker.collect_python_definitions(tmp_path, (path,))
    changed_row = next(
        c
        for c in checker.ast_candidates(changed, selection=selection, deep=False)
        if c.category == "repeated row"
    )
    assert changed_row.review_fingerprint != rows[0].review_fingerprint


def test_presence_evidence_groups_owners_and_excludes_guards_and_non_none_defaults(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/presence.py",
        """
from typing import Literal
from pydantic import Field, field_validator, model_validator

type Mode = Literal["a", "b"]

class Levels:
    mode: Mode
    a: tuple[str, ...] | None = None
    b: tuple[str, ...] | None = None
    @model_validator(mode="after")
    def validate_levels(self):
        selected = self.a if self.mode == "a" else self.b
        if not selected:
            raise ValueError("levels required")
        return self

class Child(Levels):
    detail: str | None = Field(default=None)
    enabled: bool
    def __post_init__(self):
        if self.enabled == True and self.detail is None:
            raise ValueError("detail required")

class Pair:
    a: int | None = None
    b: int | None = None
    def __post_init__(self):
        paired = self.a is not None and self.b is not None
        if not paired and self.a is not None:
            raise ValueError("unpaired")

class Guard:
    a: int | None = None
    b: int | None = None
    labels: tuple[str, ...] | None = None
    mode: Mode
    n: int
    def __post_init__(self):
        if self.a is not None and self.b is not None and self.a >= self.b:
            raise ValueError("order")
        if self.labels and len(self.labels) != self.n:
            raise ValueError("length")
        if self.a is not None and self.a not in valid[self.mode]:
            raise ValueError("value allowed by mode")

class NonNoneDefault:
    a: tuple[str, ...] = ()
    b: tuple[str, ...] | None = None
    def __post_init__(self):
        if self.b is not None and self.a and len(self.a) != len(self.b):
            raise ValueError("aligned")

class FieldPair:
    a: int | None = None
    b: int | None = None
    @field_validator("a")
    @classmethod
    def validate_a(cls, value, values):
        if value is not None and values.get("b") is None:
            raise ValueError("unpaired")
        return value
""",
    )
    definitions = checker.collect_python_definitions(tmp_path, (path,))
    candidates = checker.ast_candidates(
        definitions, selection=checker.SourceSelection(ranges=None, description="all"), deep=False
    )
    presence = [c for c in candidates if c.category == "presence dependency"]

    assert {c.first.label for c in presence} == {"Levels", "Child", "Pair", "FieldPair"}
    levels = next(c for c in presence if c.first.label == "Levels")
    assert "optional presence [a, b] with tag values [mode]" in levels.reason
    assert "inherited by: Child [validate_levels]" in levels.reason
    assert "validate_levels" not in next(c for c in presence if c.first.label == "Child").reason


def test_closed_vocabularies_include_alias_copies_enums_and_only_named_subsets(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/vocabularies.py",
        """
from enum import StrEnum
from typing import Literal, TypeAlias

type Mode = Literal["a", "b", "c"]
Legacy = Literal["c", "b", "a"]
AnnotatedAlias: TypeAlias = Literal["a", "b", "c"]
type Narrow = Literal["a", "b"]

class ModeEnum(StrEnum):
    A = "a"
    B = "b"
    C = "c"

class Transport:
    mode: Literal["a", "b", "c"]

def restricted(value: Literal["a", "c"]):
    return value
""",
    )
    definitions = checker.collect_python_definitions(tmp_path, (path,))
    candidates = checker.ast_candidates(
        definitions, selection=checker.SourceSelection(ranges=None, description="all"), deep=False
    )
    vocabularies = [c for c in candidates if c.category == "closed vocabulary"]

    assert len(vocabularies) == 2
    exact = next(c for c in vocabularies if c.reason.startswith("exact"))
    assert "exact inline copy of named alias" in exact.reason
    assert "5 places (alias: 3, enum: 1, inline: 1)" in exact.reason
    assert len(exact.evidence_locations) == 3
    subset = next(c for c in vocabularies if c.reason.startswith("named proper subset"))
    assert subset.first.label == "Narrow"
    assert all(
        "restricted" not in c.reason
        and all("restricted" not in loc.label for loc in (c.first, c.second, *c.evidence_locations))
        for c in vocabularies
    )


def test_function_overlap_normalizes_local_names(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/first.py",
        """
def encode_failure(error: Exception) -> Envelope:
    if isinstance(error, DomainError):
        return Envelope(
            str(error),
            error.diagnostics,
            kind=type(error).__name__,
            terminal=True,
        )
    return Envelope(str(error), kind=type(error).__name__, terminal=True)
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/second.py",
        """
def convert_failure(exc: Exception) -> Envelope:
    if isinstance(exc, DomainError):
        return Envelope(
            str(exc),
            exc.diagnostics,
            kind=type(exc).__name__,
            terminal=True,
        )
    return Envelope(str(exc), kind=type(exc).__name__, terminal=True)
""",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="first helper",
    )

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    function_candidates = [
        candidate for candidate in candidates if candidate.category == "function behavior"
    ]
    assert len(function_candidates) == 1
    assert function_candidates[0].score == 0.98
    assert function_candidates[0].reason.startswith("alpha-normalized AST match")


def test_similar_shape_with_different_calls_is_not_reported(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/first.py",
        """
def calculate_alpha(value: float) -> float:
    prepared = prepare_alpha(value)
    if alpha_is_valid(prepared):
        return finish_alpha(prepared)
    return default_alpha(prepared)
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/second.py",
        """
def calculate_beta(value: float) -> float:
    prepared = prepare_beta(value)
    if beta_is_ready(prepared):
        return finish_beta(prepared)
    return default_beta(prepared)
""",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(1, sys.maxsize),)},
        description="first helper",
    )

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))

    assert checker.ast_candidates(definitions, selection=selection, deep=True) == ()


def test_alpha_equivalent_groups_use_one_stable_representative(tmp_path: Path) -> None:
    checker = _load_checker()
    paths = tuple(
        _write_source(
            tmp_path,
            f"apps/grassmoth-server/src/domain/{name}.py",
            f"""
def {name}(error: Exception) -> Envelope:
    if isinstance(error, DomainError):
        return Envelope(
            str(error),
            error.diagnostics,
            kind=type(error).__name__,
            terminal=True,
        )
    return Envelope(str(error), kind=type(error).__name__, terminal=True)
""",
        )
        for name in ("alpha", "beta", "gamma")
    )
    selection = checker.SourceSelection(ranges=None, description="all")

    definitions = checker.collect_python_definitions(tmp_path, paths)
    candidates = checker.ast_candidates(definitions, selection=selection, deep=False)

    function_candidates = [
        candidate for candidate in candidates if candidate.category == "function behavior"
    ]
    assert len(function_candidates) == 2
    assert all(candidate.first.label == "alpha" for candidate in function_candidates)


def test_function_collection_ignores_interface_and_abstract_stubs(tmp_path: Path) -> None:
    checker = _load_checker()
    source_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/interfaces.py",
        '''
from typing import Protocol

class Reader(Protocol):
    def ellipsis_stub(self, value: int) -> str: ...

    def pass_stub(self, value: int) -> str:
        """Document the interface."""
        pass

    def raise_stub(self, value: int) -> str:
        raise NotImplementedError("implemented by adapters")

    def sentinel_stub(self, value: int) -> str:
        return NotImplemented
''',
    )

    definitions = checker.collect_python_definitions(tmp_path, (source_path,))

    assert [definition.qualname for definition in definitions] == ["Reader"]


def test_function_size_counts_executable_body_not_large_annotations(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/annotated_first.py",
        """
def first(
    value: tuple[dict[str, list[tuple[int, float, str]]], ...],
    context: dict[str, tuple[list[int], list[float], list[str]]],
    options: tuple[str, int, float, bool, bytes, complex],
) -> tuple[dict[str, list[tuple[int, float, str]]], ...]:
    return value
""",
    )
    second_path = _write_source(
        tmp_path,
        "apps/grassmoth-server/src/domain/annotated_second.py",
        """
def second(
    item: tuple[dict[str, list[tuple[int, float, str]]], ...],
    metadata: dict[str, tuple[list[int], list[float], list[str]]],
    settings: tuple[str, int, float, bool, bytes, complex],
) -> tuple[dict[str, list[tuple[int, float, str]]], ...]:
    return item
""",
    )
    selection = checker.SourceSelection(ranges=None, description="all")

    definitions = checker.collect_python_definitions(tmp_path, (first_path, second_path))

    assert {definition.node_count for definition in definitions} == {3}
    assert checker.ast_candidates(definitions, selection=selection, deep=True) == ()


def test_jscpd_report_is_filtered_to_selected_clone_ranges(tmp_path: Path) -> None:
    checker = _load_checker()
    first_path = _write_source(
        tmp_path,
        "packages/grassmoth/src/grassmoth/first.py",
        "\n" * 50,
    )
    second_path = _write_source(
        tmp_path,
        "packages/grassmoth/src/grassmoth/second.py",
        "\n" * 50,
    )
    report_path = tmp_path / "jscpd-report.json"
    report_path.write_text(
        json.dumps(
            {
                "duplicates": [
                    {
                        "format": "python",
                        "lines": 8,
                        "tokens": 70,
                        "firstFile": {
                            "name": "grassmoth/first.py",
                            "start": 10,
                            "end": 17,
                        },
                        "secondFile": {
                            "name": "grassmoth/second.py",
                            "start": 10,
                            "end": 17,
                        },
                    },
                    {
                        "format": "python",
                        "lines": 8,
                        "tokens": 70,
                        "firstFile": {
                            "name": "grassmoth/first.py",
                            "start": 30,
                            "end": 37,
                        },
                        "secondFile": {
                            "name": "grassmoth/second.py",
                            "start": 30,
                            "end": 37,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    selection = checker.SourceSelection(
        ranges={first_path.as_posix(): (checker.LineRange(12, 12),)},
        description="one changed hunk",
    )

    candidates = checker.parse_jscpd_report(
        report_path,
        repo_root=tmp_path,
        selection=selection,
    )

    assert len(candidates) == 1
    assert candidates[0].first.path == first_path.as_posix()
    assert candidates[0].second.path == second_path.as_posix()
    assert candidates[0].first_selected is True
    assert candidates[0].second_selected is False


def test_jscpd_report_ignores_import_only_python_clones(tmp_path: Path) -> None:
    checker = _load_checker()
    source = '''"""Activity module."""

from __future__ import annotations

import json
from package import first, second

VALUE = 1
'''
    _write_source(
        tmp_path,
        "packages/grassmoth/src/grassmoth/first.py",
        source,
    )
    _write_source(
        tmp_path,
        "packages/grassmoth/src/grassmoth/second.py",
        source,
    )
    report_path = tmp_path / "jscpd-report.json"
    report_path.write_text(
        json.dumps(
            {
                "duplicates": [
                    {
                        "format": "python",
                        "lines": 6,
                        "tokens": 70,
                        "firstFile": {
                            "name": "grassmoth/first.py",
                            "start": 1,
                            "end": 6,
                        },
                        "secondFile": {
                            "name": "grassmoth/second.py",
                            "start": 1,
                            "end": 6,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = checker.parse_jscpd_report(
        report_path,
        repo_root=tmp_path,
        selection=checker.SourceSelection(ranges=None, description="all"),
    )

    assert candidates == ()


def test_output_cap_reserves_space_for_each_candidate_category() -> None:
    checker = _load_checker()
    candidates = []
    categories = (*checker.TYPE_CATEGORIES, "token clone", "class schema", "function behavior")
    for category_index, category in enumerate(categories):
        for index in range(4):
            candidates.append(
                checker.Candidate(
                    category=category,
                    score=0.99 - category_index * 0.1 - index * 0.01,
                    first=checker.Location(f"first-{category_index}-{index}.py", 1, 5),
                    second=checker.Location(f"second-{category_index}-{index}.py", 1, 5),
                    reason="fixture",
                    first_selected=True,
                    second_selected=False,
                )
            )

    selected = checker.select_candidates(candidates, limit=len(categories))

    assert len(selected) == len(categories)
    assert {candidate.category for candidate in selected} == set(categories)
    assert len(checker.select_candidates(candidates, limit=2)) == 2


def test_reviewed_pairs_are_suppressed_by_stable_definition_identity(tmp_path: Path) -> None:
    checker = _load_checker()
    registry = tmp_path / "duplicate_reviews.json"
    registry.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "category": "function behavior",
                        "first": "src/first.py::first",
                        "second": "src/second.py::second",
                        "fingerprint": "a" * 64,
                        "classification": "related implementation",
                        "rationale": "The shared shape is intentional.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = checker.Candidate(
        category="function behavior",
        score=0.95,
        first=checker.Location("src/second.py", 20, 30, "second"),
        second=checker.Location("src/first.py", 10, 15, "first"),
        reason="fixture",
        first_selected=True,
        second_selected=False,
        review_fingerprint="a" * 64,
    )

    reviews = checker.load_reviewed_pairs(registry)
    pending, reviewed = checker.partition_reviewed_candidates((candidate,), reviews)

    assert pending == ()
    assert reviewed == (candidate,)

    stale_candidate = dataclasses.replace(candidate, review_fingerprint="b" * 64)
    pending, reviewed = checker.partition_reviewed_candidates((stale_candidate,), reviews)
    assert pending == (stale_candidate,)
    assert reviewed == ()


def test_review_registry_rejects_duplicate_pair_orderings(tmp_path: Path) -> None:
    checker = _load_checker()
    registry = tmp_path / "duplicate_reviews.json"
    common = {
        "category": "class schema",
        "fingerprint": "a" * 64,
        "classification": "intentional mirror/parity check",
        "rationale": "Protocol and model have different responsibilities.",
    }
    registry.write_text(
        json.dumps(
            {
                "reviews": [
                    {**common, "first": "a.py::A", "second": "b.py::B"},
                    {**common, "first": "b.py::B", "second": "a.py::A"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(checker.DuplicateAuditError, match="duplicate reviewed pair"):
        checker.load_reviewed_pairs(registry)
