"""Field descriptions satisfy only their own attribute documentation contract."""

import ast
from pathlib import Path

import pytest
from pydoclint.visitor import Visitor

from scripts.checks import run_pydoclint as checker

pytestmark = pytest.mark.contract

_IMPORTS = "from pydantic import BaseModel, Field\nfrom typing import Annotated\n"


def _violations(source: str):
    tree = ast.parse(source)
    original = ast.dump(tree, include_attributes=True)
    visitor = checker.SourceDescriptionVisitor(
        style="google",
        argTypeHintsInDocstring=False,
        skipCheckingShortDocstrings=False,
        checkReturnTypes=False,
        checkYieldTypes=False,
        shouldDeclareAssertErrorIfAssertStatementExists=True,
    )
    visitor.visit(tree)
    assert ast.dump(tree, include_attributes=True) == original
    return visitor.violations


@pytest.mark.parametrize(
    ("imports", "declaration"),
    [
        (_IMPORTS, 'time: float = Field(description="Elapsed model days.")'),
        (_IMPORTS, 'time: Annotated[float, Field(description="Elapsed model days.")]'),
        (
            "import pydantic as p\nimport typing as t\n",
            'time: t.Annotated[float, p.Field(description="Elapsed model days.")]',
        ),
        (
            "from pydantic.fields import Field as F\nfrom typing_extensions import Annotated as A\n",
            'time: A[float, F(description="Elapsed model days.")] = F(ge=0)',
        ),
        (
            _IMPORTS + 'DESCRIPTION = "Elapsed model days."\n',
            "time: float = Field(description=DESCRIPTION)",
        ),
    ],
)
def test_source_descriptions_satisfy_the_attribute_contract(imports, declaration):
    assert not _violations(
        f'{imports}\nclass Observation:\n    """A recorded value."""\n    {declaration}\n'
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "time: float",
        "time: float = Field(ge=0)",
        'time: float = Field(description=" \t ")',
        "time: float = Field(description=None)",
        "time: float = Field(description=unknown)",
        "time: float = Field(description=make_description())",
        'time: list[Annotated[float, Field(description="An element, not this field.")]]',
        'time: Annotated[float, Field(description="Elapsed days.")] = Field(description="")',
        'time: float = unrelated(description="Elapsed model days.")',
    ],
)
def test_absent_blank_or_unresolved_descriptions_still_require_documentation(declaration):
    findings = _violations(
        f'{_IMPORTS}\nclass Observation:\n    """A recorded value."""\n    {declaration}\n'
    )
    assert 601 in {finding.code for finding in findings}


def test_other_attributes_and_duplicate_documentation_remain_checked():
    source = (
        _IMPORTS
        + '''
class Observation:
    """A recorded value.

    Attributes:
        value: Measurement in the indicator's declared units.
    """

    time: float = Field(description="Elapsed model days.")
    value: float
'''
    )
    assert not _violations(source)
    missing = _violations(source + "    indicator_id: str\n")
    assert 601 in {finding.code for finding in missing}
    duplicate = _violations(
        source.replace(
            "    Attributes:\n", "    Attributes:\n        time: Days since the origin.\n"
        )
    )
    assert 602 in {finding.code for finding in duplicate}
    assert any(
        "Fields owned by Field(description=...): time" in finding.msg for finding in duplicate
    )


def test_imported_literal_constants_are_read_without_importing_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "SOURCE_ROOTS", (tmp_path,))
    checker._imported_literal.cache_clear()
    (tmp_path / "descriptions.py").write_text(
        'raise RuntimeError("This source must never execute")\nTIME = "Elapsed model days."\n'
    )
    source = (
        _IMPORTS
        + 'from descriptions import TIME\nclass Observation:\n    """A recorded value."""\n    time: float = Field(description=TIME)\n'
    )
    assert not _violations(source)
    checker._imported_literal.cache_clear()


@pytest.mark.parametrize(
    "prefix",
    [
        "from unrelated import Field\n",
        "from pydantic import Field\nField = unrelated\n",
        "from pydantic import Field\ndef Field(**kwargs):\n    pass\n",
    ],
)
def test_matching_names_do_not_make_unrelated_factories_pydantic_fields(prefix):
    source = (
        prefix
        + 'class Observation:\n    """A recorded value."""\n    time: float = Field(description="Elapsed model days.")\n'
    )
    assert 601 in {finding.code for finding in _violations(source)}


def test_function_checks_remain_active_inside_classes_with_source_descriptions():
    source = (
        _IMPORTS
        + '''
class Observation:
    """A recorded value."""

    time: float = Field(description="Elapsed model days.")

    def scale(self, value: float, factor: float) -> float:
        """Scale a measurement.

        Args:
            value: Measurement in the original units.
        """
        return value * factor

    def values(self, count: int):
        """Read successive values.

        Args:
            count: Number of values to read.
        """
        if count < 0:
            raise ValueError("Negative count")
        yield self.time
'''
    )
    assert {finding.code for finding in _violations(source)} == {101, 103, 201, 402, 501, 503}


def test_native_cli_preserves_configuration_locations_and_failure_status(tmp_path, capsys):
    source = tmp_path / "observation.py"
    source.write_text(
        _IMPORTS
        + '''
class Observation:
    """A recorded value.

    Attributes:
        value: Measurement in the indicator's declared units.
    """
    time: float = Field(description="Elapsed model days.")
    value: float
'''
    )
    config = Path(__file__).resolve().parents[5] / "pyproject.toml"
    args = ["--quiet", "--config", str(config), str(source)]
    with pytest.raises(SystemExit) as success:
        checker.main(args)
    assert success.value.code == 0
    assert checker.pydoclint_main.Visitor is Visitor
    source.write_text(source.read_text() + "    indicator_id: str\n")
    with pytest.raises(SystemExit) as failure:
        checker.main(args)
    assert failure.value.code == 1
    output = capsys.readouterr().err
    assert str(source) in output
    assert "4: DOC601" in output
    assert checker.pydoclint_main.Visitor is Visitor
