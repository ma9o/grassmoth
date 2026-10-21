"""Library callers cannot cross into implementation, including through local helpers."""

from pathlib import Path

import pytest
from scripts.checks.check_public_imports import find_public_import_violations

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "source",
    [
        "from nof1_causal_lab.models.ssm.inference.fit import fit as run",
        "import nof1_causal_service.actions.fit.remote as compute",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from nof1_causal_lab.models.ssm.inference.types import MCMCPosterior",
        "from nof1_causal_service.public.studies import _study_handle",
        "from nof1_causal_service.public.studies import *",
        "from importlib import import_module as load\nload('nof1_causal_service.actions.runners')",
        "import importlib\nimportlib.import_module(name)",
        "import sys as system\nsystem.path.insert(0, 'source')",
        "from nof1_causal_service.public import studies\nstudies._study_handle()",
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
        "from nof1_causal_service.actions.fit.remote import execute_fit_compute as execute\n"
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
        "from nof1_causal_service.public.client import StudyClient\n"
        "from nof1_causal_service.public import studies\n"
        "import nof1_causal_service.public.contracts as contracts\n"
        "def run():\n return StudyClient, studies.submit_action, contracts.FitRequest\n"
    )
    assert find_public_import_violations((package / "run.py",), import_roots=(tmp_path,)) == ()
    helper.write_text("from nof1_causal_service.compute.modal_images import gpu_image\n")
    assert find_public_import_violations((package / "run.py",), import_roots=(tmp_path,))


def test_helper_package_initializers_cannot_hide_internal_imports(tmp_path: Path) -> None:
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "__init__.py").write_text("import nof1_causal_service.actions.fit.remote\n")
    (helper / "run.py").write_text("def run(): pass\n")
    caller = tmp_path / "experiment.py"
    caller.write_text("from helper.run import run\n")
    violations = find_public_import_violations((caller,), import_roots=(tmp_path,))
    assert len(violations) == 1
    assert violations[0].path == helper / "__init__.py"


def test_namespace_alias_cannot_turn_framework_implementation_into_a_local_helper(
    tmp_path: Path,
) -> None:
    module = tmp_path / "src/nof1_causal_lab/engine.py"
    module.parent.mkdir(parents=True)
    module.write_text("def fit(): pass\n")
    caller = tmp_path / "experiment.py"
    caller.write_text("from src.nof1_causal_lab.engine import fit\n")
    violations = find_public_import_violations((caller,), import_roots=(tmp_path,))
    assert len(violations) == 1
    assert violations[0].path == caller
    assert find_public_import_violations((module,)) == ()
