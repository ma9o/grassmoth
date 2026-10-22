"""Contracts for source-only, binding-aware similarity evidence."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

import pytest

from scripts.checks import find_ast_similarities as checker

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract


def _compare(first: str, second: str) -> checker.AstComparison:
    nodes = (ast.parse(first).body[0], ast.parse(second).body[0])
    assert isinstance(nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef))
    assert isinstance(nodes[1], (ast.FunctionDef, ast.AsyncFunctionDef))
    return checker.compare_terms(
        checker.normalize_function(nodes[0]), checker.normalize_function(nodes[1])
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "def f(a):\n    result = a + 1\n    return result",
            "def g(x):\n    renamed = x + 1\n    return renamed",
        ),
        (
            "def f(a):\n    def inner(b=a):\n        return a + b\n    return inner()",
            "def g(x):\n    def helper(y=x):\n        return x + y\n    return helper()",
        ),
        (
            "def f(a):\n    def inner(a: a=a) -> a:\n        return a\n    return inner()",
            "def g(x):\n    def helper(y: x=x) -> x:\n        return y\n    return helper()",
        ),
        (
            "def f(x):\n    return [x for x in x], x",
            "def g(items):\n    return [item for item in items], items",
        ),
        (
            "def f(xs):\n    values = [(last := x) for x in xs]\n    return values, last",
            "def g(items):\n    result = [(seen := item) for item in items]\n    return result, seen",
        ),
        (
            "def f(xs):\n    return [(x, y) for x in xs for y in x]",
            "def g(items):\n    return [(first, second) for first in items for second in first]",
        ),
        (
            "def f(a):\n    count = a\n    def inc():\n        nonlocal count\n        count += 1\n        return count\n    return inc()",
            "def g(b):\n    total = b\n    def step():\n        nonlocal total\n        total += 1\n        return total\n    return step()",
        ),
        (
            "def f(a):\n    import math as m\n    from math import floor as down\n    return m.sqrt(a), down(a)",
            "def g(x):\n    import math as maths\n    from math import floor as lower\n    return maths.sqrt(x), lower(x)",
        ),
        (
            "def f(a):\n    global shared\n    shared = a\n    return shared",
            "def g(x):\n    global shared\n    shared = x\n    return shared",
        ),
        (
            "def f(a):\n    class C:\n        a = 1\n        def method(self):\n            return a\n    return C",
            "def g(x):\n    class D:\n        a = 1\n        def method(self):\n            return x\n    return D",
        ),
        (
            "def f(a):\n    try:\n        return call(a)\n    except ValueError as err:\n        return str(err)",
            "def g(x):\n    try:\n        return call(x)\n    except ValueError as error:\n        return str(error)",
        ),
        (
            "def f(a):\n    match a:\n        case {'key': value, **rest}:\n            return value, rest\n        case [head, *tail]:\n            return head, tail",
            "def g(x):\n    match x:\n        case {'key': item, **others}:\n            return item, others\n        case [first, *remaining]:\n            return first, remaining",
        ),
    ],
)
def test_consistent_renames_preserve_binding_relationships(first: str, second: str) -> None:
    comparison = _compare(first, second)

    assert comparison.score == 1
    assert comparison.differences == ()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("def f(a, b):\n    return call(a, b)", "def g(x, y):\n    return call(x, x)"),
        (
            "def f(a):\n    return lambda x: call(a, x)",
            "def g(a):\n    return lambda a: call(a, a)",
        ),
        (
            "def f(a):\n    return [x for x in a], a",
            "def g(a):\n    return [a for a in a], a + a",
        ),
    ],
)
def test_repeated_and_shadowed_variables_remain_visible(first: str, second: str) -> None:
    comparison = _compare(first, second)

    assert comparison.score < 1
    assert comparison.differences


def test_captured_variable_changed_to_parameter_is_a_binding_difference() -> None:
    comparison = _compare(
        "def f(a):\n    return lambda x: call(a, x)",
        "def g(a):\n    return lambda a: call(a, a)",
    )

    assert len(comparison.differences) == 1
    assert comparison.differences[0].kind == "binding"
    assert "binding:0:parameter:0" in comparison.differences[0].left
    assert "binding:1:parameter:0" in comparison.differences[0].right


@pytest.mark.parametrize(
    "changed",
    [
        "def g(x):\n    return process(x + 2)",
        "def g(x):\n    return process(x - 1)",
        "def g(x):\n    return another(x + 1)",
        "def g(x):\n    return x.process(1)",
    ],
)
def test_literals_operators_and_external_contract_names_are_retained(changed: str) -> None:
    comparison = _compare("def f(a):\n    return process(a + 1)", changed)

    assert comparison.score < 1
    assert comparison.differences


def test_nested_implementations_are_compared_and_docstrings_are_ignored() -> None:
    first = "def f(a):\n    'first doc'\n    def inner():\n        'inner doc'\n        return a + 1\n    return inner()"
    renamed = "def g(x):\n    'different doc'\n    def helper():\n        'other doc'\n        return x + 1\n    return helper()"

    assert _compare(first, renamed).score == 1
    comparison = _compare(first, renamed.replace("x + 1", "x - 1"))
    assert any(d.left == "Add" and d.right == "Sub" for d in comparison.differences)


def test_statement_alignment_keeps_surrounding_shared_code() -> None:
    comparison = _compare(
        "def f(a):\n    first = transform(a)\n    second = finish(first)\n    return second",
        "def g(x):\n    note(x)\n    first = transform(x)\n    second = finish(first)\n    return second",
    )

    assert comparison.score > 0.8
    assert len(comparison.differences) == 1
    assert comparison.differences[0].kind == "inserted statement"
    assert comparison.differences[0].right_line == 2
    assert "note" in comparison.differences[0].right


def test_repeated_substitution_has_one_consistent_hole() -> None:
    comparison = _compare(
        "def f(a):\n    return before(a) + before(a)",
        "def g(x):\n    return after(x) + after(x)",
    )

    assert len(comparison.differences) == 2
    assert {d.hole for d in comparison.differences} == {0}


def test_extra_parameters_do_not_shift_local_binding_identities() -> None:
    comparison = _compare(
        "def f(a):\n    result = call(a)\n    return result",
        "def g(x, unused):\n    renamed = call(x)\n    return renamed",
    )

    assert comparison.score == 1
    assert comparison.differences == ()


def test_walrus_in_nested_lambda_preserves_store_and_load_identity() -> None:
    comparison = _compare(
        "def f(items):\n    return [lambda x: ((x := 1), x) for item in items]",
        "def g(rows):\n    return [lambda y: ((renamed := 1), renamed) for row in rows]",
    )

    assert len(comparison.differences) == 2
    assert {d.kind for d in comparison.differences} == {"binding"}
    assert len({d.hole for d in comparison.differences}) == 1


def _write_source(repo_root: Path, relative: str, source: str) -> None:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_cli_compares_selected_sources_against_corpus_without_importing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prefix = "apps/grassmoth-server/src/domain"
    normalized: list[str] = []
    normalize = checker.normalize_function

    def _record_normalization(
        node: checker.FunctionNode, outer: tuple[checker.BindingScope, ...] = ()
    ) -> checker.AstTerm:
        normalized.append(node.name)
        return normalize(node, outer)

    _write_source(
        tmp_path,
        f"{prefix}/a.py",
        "raise RuntimeError('must not import')\ndef f(a=1):\n    return call(a)",
    )
    _write_source(tmp_path, f"{prefix}/b.py", "def g(x=2):\n    return call(x)")
    _write_source(tmp_path, f"{prefix}/c.py", "def unrelated(a):\n    return a - 3")
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "normalize_function", _record_normalization)
    argv = [
        f"{prefix}/a.py",
        "--min-nodes",
        "1",
        "--threshold",
        "1",
        "--json",
        "--workers",
        "1",
    ]

    assert checker.main(argv) == 0
    first_output = capsys.readouterr().out
    report = json.loads(first_output)
    assert report["advisory"] is True
    assert report["indexed_functions"] == 3
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert finding["first"]["path"] == f"{prefix}/a.py"
    assert finding["second"]["path"] == f"{prefix}/b.py"
    assert finding["first"]["signature"] == "(a=1)"
    assert finding["second"]["signature"] == "(x=2)"
    assert finding["differences"] == []
    assert normalized == ["f", "g"]
    assert checker.main([*argv[:-1], "2"]) == 0
    assert capsys.readouterr().out == first_output
    assert normalized == ["f", "g", "f", "g"]


@pytest.mark.parametrize("targeted", [False, True])
def test_parallel_search_preserves_serial_evidence_and_order(
    targeted: bool, tmp_path: Path
) -> None:
    prefix = "apps/grassmoth-server/src/domain"
    sources = {
        "a": "def f(a):\n    def inner(b):\n        return before(a) + after(b)\n    return inner(a)",
        "b": "def g(x):\n    def helper(y):\n        return before(x) + after(y)\n    return helper(x)",
        "c": "def h(a):\n    def changed(b):\n        return after(a) + before(b)\n    return changed(a)",
        "d": "def extra(a):\n    first = before(a)\n    note(first)\n    return after(first)",
        "e": "def other(x):\n    renamed = before(x)\n    return after(renamed)",
    }
    for name, source in sources.items():
        _write_source(tmp_path, f"{prefix}/{name}.py", source)
    paths = checker.source_paths(tmp_path)
    selection = (
        checker.SourceSelection(
            {
                f"{prefix}/a.py": (checker.LineRange(3, 3),),
                f"{prefix}/b.py": (checker.LineRange(5, 5),),
            },
            "selected ranges",
        )
        if targeted
        else checker.SourceSelection(None, "all")
    )
    options = {"min_nodes": 1, "threshold": 0.0, "neighbors": 3, "retrieval_threshold": 0.0}

    serial_functions, serial_findings = checker.search_sources(
        tmp_path, paths, selection, workers=1, **options
    )
    parallel_functions, parallel_findings = checker.search_sources(
        tmp_path, paths, selection, workers=2, **options
    )

    assert serial_findings
    assert len(parallel_functions) == len(serial_functions)
    assert checker.render_report(
        parallel_functions, parallel_findings, limit=100, selection=selection
    ) == checker.render_report(serial_functions, serial_findings, limit=100, selection=selection)
    assert checker._json_report(
        parallel_functions, parallel_findings, limit=100, selection=selection
    ) == checker._json_report(serial_functions, serial_findings, limit=100, selection=selection)
    assert all(
        not checker._nested_pair(f.first.definition, f.second.definition) for f in parallel_findings
    )


def test_index_preserves_enclosing_bindings_and_excludes_nested_pairs(tmp_path: Path) -> None:
    relative = "apps/grassmoth-server/src/domain/nested.py"
    _write_source(
        tmp_path,
        relative,
        "def f(a, b):\n    def one():\n        return a\n    def two():\n        return b\n    return a",
    )
    functions = checker.index_functions(tmp_path, checker.source_paths(tmp_path), min_nodes=1)
    indexed = {f.definition.qualname: i for i, f in enumerate(functions)}
    terms = checker._comparison_terms(functions, ((indexed["f.one"], indexed["f.two"]),))
    comparison = checker.compare_terms(terms[indexed["f.one"]], terms[indexed["f.two"]])

    assert comparison.differences[0].kind == "binding"
    pairs = checker.candidate_pairs(
        functions, checker.SourceSelection(None, "all"), neighbors=5, retrieval_threshold=0
    )
    assert pairs
    assert all(
        functions[i].definition.qualname != "f" and functions[j].definition.qualname != "f"
        for i, j in pairs
    )


@pytest.mark.parametrize("source", ["def broken(:", "def ok():\n    return 1"])
@pytest.mark.parametrize("workers", [1, 2])
def test_cli_reports_parse_and_target_errors(
    source: str,
    workers: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    relative = "apps/grassmoth-server/src/domain/a.py"
    _write_source(tmp_path, relative, source)
    _write_source(tmp_path, "apps/grassmoth-server/src/domain/b.py", "def other():\n    return 2")
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    argv = [relative] if source.startswith("def broken") else ["missing.py"]

    assert checker.main([*argv, "--workers", str(workers)]) == 2
    assert "AST similarity review failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ["--threshold", "nan"],
        ["--threshold", "1.1"],
        ["--neighbors", "0"],
        ["--min-nodes", "0"],
        ["--limit", "0"],
        ["--workers", "0"],
        ["--workers", "-1"],
    ],
)
def test_cli_rejects_invalid_search_bounds(option: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        checker.main(option)

    assert raised.value.code == 2
