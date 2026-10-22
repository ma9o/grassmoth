"""Library callers cannot cross into implementation, including through local helpers."""

import subprocess
from pathlib import Path

import pytest

from scripts.checks.check_public_imports import find_public_import_violations

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "source",
    [
        "from grassmoth.models.ssm.inference.fit import fit as run",
        "from grassmoth.public import _fit",
        "from grassmoth.public import *",
        "import grassmoth.public as library\nlibrary._fit",
        "from importlib import import_module\nimport_module('grassmoth.models.ssm.inference.fit')",
        "import grassmoth_server.actions.fit.remote as compute",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from grassmoth.models.ssm.inference.types import MCMCPosterior",
        "from grassmoth_server.public.client import _ErrorBody",
        "from grassmoth_server.public.client import *",
        "from grassmoth_server.public.contracts import FitRequest",
        "from grassmoth_server.actions.contracts.polling import ActionResult",
        "from importlib import import_module\nimport_module('grassmoth_server.public.contracts')",
        "from importlib import import_module as load\nload('grassmoth_server.actions.runners')",
        "import importlib\nimportlib.import_module(name)",
        "import sys as system\nsystem.path.insert(0, 'source')",
        "from grassmoth_server.public import client\nclient._ErrorBody",
    ],
)
def test_rejects_private_imports_and_dynamic_bypasses(tmp_path: Path, source: str) -> None:
    caller = tmp_path / "caller.py"
    caller.write_text(source)
    assert find_public_import_violations((caller,))


def test_scans_ignored_files_and_follows_helpers_outside_selected_directory(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("scratchpad/\n")
    campaign = tmp_path / "scratchpad"
    campaign.mkdir()
    caller = campaign / "run.py"
    caller.write_text("from helper import execute\n")
    helper = tmp_path / "helper.py"
    helper.write_text(
        "from grassmoth_server.actions.fit.remote import execute_fit_compute as execute\n"
    )
    violations = find_public_import_violations((campaign,), import_roots=(tmp_path,))
    assert len(violations) == 1
    assert violations[0].path == helper


def test_public_imports_and_relative_local_helpers_are_allowed(tmp_path: Path) -> None:
    package = tmp_path / "experiment"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "run.py").write_text("from .helper import run\n")
    helper = package / "helper.py"
    helper.write_text(
        "from grassmoth.public import fit, FitOutput\n"
        "from grassmoth import public\n"
        "import grassmoth.public as library\n"
        "from importlib import import_module\n"
        "import_module('grassmoth.public')\n"
        "def run():\n return fit, FitOutput, public.fit, library.fit\n"
    )
    assert find_public_import_violations((package / "run.py",), import_roots=(tmp_path,)) == ()
    helper.write_text("from grassmoth_server.compute.modal_images import gpu_image\n")
    assert find_public_import_violations((package / "run.py",), import_roots=(tmp_path,))


def test_helper_package_initializers_cannot_hide_internal_imports(tmp_path: Path) -> None:
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "__init__.py").write_text("import grassmoth_server.actions.fit.remote\n")
    (helper / "run.py").write_text("def run(): pass\n")
    caller = tmp_path / "experiment.py"
    caller.write_text("from helper.run import run\n")
    violations = find_public_import_violations((caller,), import_roots=(tmp_path,))
    assert len(violations) == 1
    assert violations[0].path == helper / "__init__.py"


def test_namespace_alias_cannot_turn_framework_implementation_into_a_local_helper(
    tmp_path: Path,
) -> None:
    module = tmp_path / "src/grassmoth/engine.py"
    module.parent.mkdir(parents=True)
    module.write_text("def fit(): pass\n")
    caller = tmp_path / "experiment.py"
    caller.write_text("from src.grassmoth.engine import fit\n")
    violations = find_public_import_violations((caller,), import_roots=(tmp_path,))
    assert len(violations) == 1
    assert violations[0].path == caller
    assert find_public_import_violations((module,)) == ()


def test_repository_discovery_covers_callers_and_exempts_only_test_entry_points(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("scratchpad/\n")
    callers = (
        "new_package/caller.py",
        "scripts/build.py",
        "notebooks/exploration.py",
        "scratchpad/experiment.py",
        ".agents/check.py",
    )
    exempt = (
        "tests/helper.py",
        "tests/test_internal.py",
        "conftest.py",
        ".venv/lib/dependency.py",
        "node_modules/dependency.py",
        "__marimo__/session.py",
    )
    for name in (*callers, *exempt):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from grassmoth.models.ssm.inference.fit import fit\n")
    violations = find_public_import_violations((tmp_path,))
    assert {violation.path.relative_to(tmp_path).as_posix() for violation in violations} == set(
        callers
    )
    (tmp_path / "new_package/caller.py").write_text("from tests.helper import fit\n")
    violations = find_public_import_violations((tmp_path,), import_roots=(tmp_path,))
    assert {violation.path.relative_to(tmp_path).as_posix() for violation in violations} == {
        *callers[1:],
        "tests/helper.py",
    }


def test_default_selection_covers_tracked_and_new_files_with_explicit_ignored_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.checks import check_public_imports

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("scratchpad/\n")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("from grassmoth.public import fit\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    caller = tmp_path / "new_workspace/caller.py"
    caller.parent.mkdir()
    caller.write_text("from grassmoth.actions.fit.action import fit\n")
    ignored = tmp_path / "scratchpad/experiment.py"
    ignored.parent.mkdir()
    ignored.write_text("from grassmoth.actions.fit.action import fit\n")
    monkeypatch.setattr(check_public_imports, "_REPO", tmp_path)
    monkeypatch.setattr("sys.argv", ["check_public_imports"])
    assert check_public_imports.main() == 1
    output = capsys.readouterr().out
    assert str(caller) in output
    assert str(ignored) not in output
    caller.write_text("from grassmoth.public import fit\n")
    assert check_public_imports.main() == 0
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["check_public_imports", str(ignored)])
    assert check_public_imports.main() == 1
    assert str(ignored) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "rejected"),
    [
        ("from grassmoth.public import fit, FitOutput", False),
        ("from grassmoth import public\npublic.fit", False),
        ("from grassmoth_server.study.records import Applied", False),
        ("from grassmoth.actions.fit.action import fit", True),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from grassmoth.artifacts.base import Value",
            True,
        ),
        (
            "from importlib import import_module\nimport_module('grassmoth.actions.fit.action')",
            True,
        ),
        ("import grassmoth.public as api\napi.__dict__", True),
    ],
)
def test_service_uses_only_the_scientific_public_module(
    tmp_path: Path, source: str, rejected: bool
) -> None:
    service = tmp_path / "src/grassmoth_server"
    service.mkdir(parents=True)
    caller = service / "caller.py"
    caller.write_text(source)
    violations = find_public_import_violations(
        (service,), import_roots=(tmp_path / "src",), service_callers=True
    )
    assert bool(violations) == rejected
    assert all("ARCH013" in violation.diagnostic() for violation in violations)
