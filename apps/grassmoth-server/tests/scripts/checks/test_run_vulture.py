"""Vulture reference generation and concurrent ownership passes."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import vulture.core as vulture_core

from scripts.checks import run_vulture
from scripts.checks.run_vulture import _scan_string_type_node, _write_phantom

pytestmark = pytest.mark.contract


def test_keyword_field_alias_does_not_break_string_type_references(tmp_path):
    annotation = ast.parse('Annotated["ArtifactId", Field(alias="from")]', mode="eval").body
    refs: set[str] = set()
    _scan_string_type_node(annotation, refs)
    _write_phantom(tmp_path, "refs.py", refs)
    tree = ast.parse((tmp_path / "refs.py").read_text())
    assert {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} == {"_", "ArtifactId"}


def test_exported_inherited_fields_are_suppressed_by_location_only(tmp_path, monkeypatch, capsys):
    pipeline = tmp_path / "apps/grassmoth-server"
    source = pipeline / "src/domain.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from pydantic import BaseModel, Field\n"
        "class Parent(BaseModel):\n"
        "    shared: int = Field(alias='wire_shared')\n"
        "class Exported(Parent):\n"
        "    pass\n"
        "class Internal(BaseModel):\n"
        "    shared: int\n"
        "def shared():\n"
        "    return 1\n"
    )
    api = tmp_path / "packages/api-types/schemas/openapi.json"
    api.parent.mkdir(parents=True)
    api.write_text(
        json.dumps(
            {
                "components": {
                    "schemas": {
                        "Exported-Output": {
                            "title": "Exported-Output",
                            "x-python-module": "domain",
                            "properties": {"wire_shared": {}},
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(run_vulture, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_vulture, "SOURCE_ROOTS", (pipeline / "src",))
    exported = run_vulture._collect_exported_field_locations([Path("apps/grassmoth-server/src")])
    assert exported == frozenset({(source.resolve(), 3)})
    analyzer = vulture_core.Vulture()
    analyzer.scan(source.read_text(), filename=str(source))
    run_vulture._report_vulture(
        analyzer,
        min_confidence=60,
        sort_by_size=False,
        make_whitelist=False,
        report_roots=["apps/grassmoth-server/src"],
        exported_fields=exported,
    )
    findings = capsys.readouterr().out
    assert f"{source}:3: unused variable 'shared'" not in findings
    assert ":7: unused variable 'shared'" in findings
    assert "unused function 'shared'" in findings


def test_key_reads_and_typed_iteration_do_not_make_unread_fields_live(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/records.py").write_text(
        "from typing import Literal, TypedDict\n"
        "class Metrics(TypedDict):\n"
        "    mean: float\n"
        "    spread: float\n"
        "class Unread(TypedDict):\n"
        "    never: int\n"
        "    tag: Literal['literal_only']\n"
        "def total(metrics: Metrics):\n"
        "    return sum(metrics.values())\n"
        "data['write_only'] = 3\n"
        "read = data['read_key']\n"
        "optional = data.get('optional_key')\n"
    )
    monkeypatch.setattr(run_vulture, "REPO_ROOT", tmp_path)
    assert run_vulture._collect_subscript_reads(["src"]) == {
        "mean",
        "spread",
        "read_key",
        "optional_key",
    }


@pytest.mark.parametrize("used_locally", [False, True])
def test_concurrent_passes_preserve_ownership_and_clean_caches(tmp_path, used_locally):
    script = tmp_path / "apps/grassmoth-server/scripts/checks/run_vulture.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(run_vulture.__file__), script)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.vulture]\npaths = ["packages/grassmoth/src", "apps/grassmoth-server/src"]\nmin_confidence = 60\n'
    )
    for directory in (
        "apps/grassmoth-server/src",
        "apps/grassmoth-server/tests",
        "packages/grassmoth/tests",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    files = {
        "packages/grassmoth/src/api.py": ("production_helper", ""),
        "apps/grassmoth-server/scripts/usage.py": ("scripts_helper", ""),
        "apps/grassmoth-server/notebooks/demo.py": (
            "notebook_helper",
            "from scripts.usage import scripts_helper\nscripts_helper()\n",
        ),
        "packages/grassmoth/tests/test_usage.py": (
            "fixture_helper",
            "from src.api import production_helper\n"
            "from notebooks.demo import notebook_helper\n"
            "production_helper()\nnotebook_helper()\n",
        ),
    }
    for relative, (helper, imports) in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        local_use = f"{helper}()\n" if used_locally else ""
        path.write_text(f"{imports}\ndef {helper}():\n    return 1\n\n{local_use}")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == (0 if used_locally else 3), completed.stdout + completed.stderr
    assert not completed.stderr
    findings = [line for line in completed.stdout.splitlines() if "unused function" in line]
    expected_helpers = [] if used_locally else [helper for helper, _ in files.values()]
    assert len(findings) == len(expected_helpers)
    for finding, helper in zip(findings, expected_helpers, strict=True):
        assert f"unused function '{helper}'" in finding
    assert not list((tmp_path / ".vulture_cache").iterdir())
