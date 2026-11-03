"""Public sinks follow signatures and preserve ownership of identically named members."""

import ast

import pytest

from grassmoth_checks.public_contracts import public_contract_locations, public_exports

pytestmark = pytest.mark.contract


def test_public_sinks_follow_aliases_bases_and_returns_without_marking_internal_members(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.grassmoth-checks]\npublic-api = ["src/example/public.py"]\n'
    )
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    (source / "public.py").write_text("from example.api import make as make\n")
    owner = source / "api.py"
    owner.write_text(
        "from example.records import Result as Payload\n"
        "type Output = Payload\n"
        "def make() -> Output: pass\n"
        "def hidden(): pass\n"
    )
    records = source / "records.py"
    records.write_text(
        "class Base:\n"
        "    label: str\n"
        "class Result(Base):\n"
        "    value: int\n"
        "    @classmethod\n"
        "    def create(cls) -> Result: pass\n"
        "class Internal:\n"
        "    value: int\n"
        "    def create(self): pass\n"
    )
    exports = public_exports(tmp_path)
    assert exports == frozenset({"example.api.make"})
    locations = public_contract_locations((tmp_path / "src",), exports)
    assert (owner, 3) in locations
    assert (owner, 4) not in locations
    assert (records, 2) in locations
    assert (records, 4) in locations
    assert (records, 5) in locations  # Vulture identifies a method by its first decorator.
    internal = ast.parse(records.read_text()).body[-1]
    assert all(
        (records, node.lineno) not in locations
        for node in ast.walk(internal)
        if hasattr(node, "lineno")
    )
