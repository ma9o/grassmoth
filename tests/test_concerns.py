"""Validate concern ownership and native marker selection at every test scope."""

import shlex
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def concern_suite(pytester: pytest.Pytester) -> pytest.Pytester:
    root = Path(__file__).resolve().parents[1]
    pytester.makeconftest((root / "conftest.py").read_text())
    config = tomllib.loads((root / "pyproject.toml").read_text())
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    options = shlex.split(config["tool"]["pytest"]["ini_options"]["addopts"])
    default_selection = options[options.index("-m") + 1]
    pytester.makeini(
        "[pytest]\naddopts = --strict-markers -m "
        + shlex.quote(default_selection)
        + "\nmarkers =\n    "
        + "\n    ".join(markers)
    )
    return pytester


def test_concerns_inherit_and_allow_multiple_owners(concern_suite: pytest.Pytester) -> None:
    concern_suite.makepyfile(
        test_module="""
        import pytest
        pytestmark = pytest.mark.contract

        def test_module_owner():
            pass
        """,
        test_scoped="""
        import pytest

        @pytest.mark.contract
        class TestOwned:
            @pytest.mark.parametrize("value", [1, 2])
            def test_parameterized(self, value):
                pass

            @pytest.mark.inference(concern="warmup")
            def test_numerical_child(self):
                pass

        @pytest.mark.inference(concern="warmup")
        class TestWarmup:
            @pytest.mark.parametrize("value", [1, 2])
            def test_parameterized(self, value):
                pass

        @pytest.mark.inference(concern="warmup")
        @pytest.mark.inference(concern="recovery")
        def test_multiple_owners():
            pass

        @pytest.mark.parametrize("value", [
            pytest.param(1, marks=pytest.mark.inference(concern="simulation"), id="simulation"),
            pytest.param(2, marks=pytest.mark.inference(concern="predictive"), id="predictive"),
            pytest.param(3, marks=pytest.mark.inference(concern="recovery"), id="recovery"),
        ])
        def test_parameter_owner(value):
            pass
        """,
        test_sampling="""
        import pytest
        pytestmark = pytest.mark.inference(concern="sampling")

        def test_module_owner():
            pass
        """,
        test_workflow="""
        import pytest
        pytestmark = pytest.mark.workflow

        def test_workflow_owner():
            pass
        """,
    )
    contracts = {
        "test_module.py::test_module_owner",
        "test_scoped.py::TestOwned::test_parameterized[1]",
        "test_scoped.py::TestOwned::test_parameterized[2]",
    }
    warmup = {
        "test_scoped.py::TestOwned::test_numerical_child",
        "test_scoped.py::TestWarmup::test_parameterized[1]",
        "test_scoped.py::TestWarmup::test_parameterized[2]",
        "test_scoped.py::test_multiple_owners",
    }
    other_inference = {
        "test_sampling.py::test_module_owner",
        "test_scoped.py::test_parameter_owner[simulation]",
        "test_scoped.py::test_parameter_owner[predictive]",
        "test_scoped.py::test_parameter_owner[recovery]",
    }
    workflows = {"test_workflow.py::test_workflow_owner"}
    for selection, expected in (
        (None, contracts),
        ("", contracts | warmup | other_inference | workflows),
        ("inference", warmup | other_inference),
        ("inference(concern='warmup')", warmup),
        ("inference and not inference(concern='warmup')", other_inference),
        (
            "inference(concern='warmup') and inference(concern='recovery')",
            {"test_scoped.py::test_multiple_owners"},
        ),
        ("workflow", workflows),
    ):
        args = ["--collect-only", "-q", "-n", "0"]
        if selection is not None:
            args.extend(["-m", selection])
        result = concern_suite.runpytest_subprocess(*args)
        assert result.ret == pytest.ExitCode.OK
        collected = {line for line in result.stdout.lines if line.startswith("test_")}
        assert collected == expected, selection


def test_unowned_cases_fail_before_marker_selection(concern_suite: pytest.Pytester) -> None:
    concern_suite.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("value", [
            pytest.param(1, marks=pytest.mark.contract, id="owned"),
            pytest.param(2, marks=pytest.mark.inference, id="parent-only"),
            pytest.param(3, id="unowned"),
            pytest.param(4, marks=pytest.mark.inference(concern="typo"), id="unknown-child"),
            pytest.param(5, marks=pytest.mark.inference("warmup"), id="positional-child"),
            pytest.param(6, marks=pytest.mark.inference(kind="warmup"), id="wrong-keyword"),
            pytest.param(7, marks=[
                pytest.mark.contract,
                pytest.mark.inference(concern="warmup", kind="predictive"),
            ], id="extra-keyword"),
        ])
        def test_parameterized(value):
            pass
        """
    )
    result = concern_suite.runpytest_subprocess("--collect-only", "-q", "-n", "0", "-m", "contract")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Every test must declare at least one concern owner*"])
    for case in (
        "unowned",
        "parent-only",
        "unknown-child",
        "positional-child",
        "wrong-keyword",
        "extra-keyword",
    ):
        assert f"::test_parameterized[{case}]" in result.stderr.str()
    assert "::test_parameterized[owned]" not in result.stderr.str()
