"""Exercise the configured package contract against runtime and type-only imports."""

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_library_exposes_only_the_seven_action_functions() -> None:
    from inspect import getmembers, isfunction

    from grassmoth import public

    assert {name for name, _ in getmembers(public, isfunction)} == {
        "edit_question",
        "edit_model",
        "prepare_data",
        "fit",
        "simulate",
        "model_diff",
        "data_diff",
    }


@pytest.mark.parametrize("direction", ["runtime", "type_only", "service_to_library"])
def test_library_service_import_contract(tmp_path: Path, direction: str) -> None:
    settings = tomllib.loads((Path(__file__).resolve().parents[5] / "pyproject.toml").read_text())[
        "tool"
    ]["importlinter"]
    contract = next(item for item in settings["contracts"] if item["name"].startswith("ARCH012 "))
    configuration = {
        "root_packages": settings["root_packages"],
        "exclude_type_checking_imports": settings["exclude_type_checking_imports"],
    }
    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.importlinter]\n"
        + "\n".join(f"{key} = {json.dumps(value)}" for key, value in configuration.items())
        + "\n[[tool.importlinter.contracts]]\n"
        + "\n".join(f"{key} = {json.dumps(value)}" for key, value in contract.items())
    )
    library = tmp_path / "grassmoth"
    service = tmp_path / "grassmoth_server"
    for package in (library, service):
        package.mkdir()
        (package / "__init__.py").touch()
    if direction == "service_to_library":
        (service / "actions.py").write_text("import grassmoth\n")
    else:
        prefix = (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    "
            if direction == "type_only"
            else ""
        )
        (library / "actions.py").write_text(prefix + "import grassmoth_server\n")

    result = subprocess.run(
        ["lint-imports", "--config", str(config), "--no-cache"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if direction == "service_to_library" else 1), (
        result.stdout + result.stderr
    )
    assert ("KEPT" if direction == "service_to_library" else "BROKEN") in result.stdout
