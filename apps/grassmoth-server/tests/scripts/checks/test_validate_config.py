"""Required numerical boundaries cannot lose coverage when hook wiring is synced."""

import ast
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract


@lru_cache(maxsize=1)
def _checker() -> Any:
    name = "validate_config_under_test"
    path = Path(__file__).resolve().parents[3] / "scripts/checks/validate_config.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("annotation", ["", ": Array", ": jax.Array", ": \"Float[Array, ' U']\""])
def test_annotation_removal_cannot_silently_drop_required_module(tmp_path, monkeypatch, annotation):
    checker = _checker()
    monkeypatch.setattr(
        checker, "_REQUIRED_SHAPE_BOUNDARIES", (("mcmc_state", "make", ("position",)),)
    )
    path = tmp_path / "packages/grassmoth/src/grassmoth/models/ssm/inference/mcmc_state.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "import jax\nfrom grassmoth.models.ssm.shapes import Array, Float\n"
        f"def make(position{annotation}): pass\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--jaxtyping-packages=beartype.beartype"\n'
    )
    checker.sync_jaxtyping_wiring(tmp_path)
    errors, _ = checker.validate_jaxtyping_wiring(tmp_path)
    assert len(errors) == 1
    assert "make: required shape annotations missing for position" in errors[0]


def test_nested_boundaries_and_import_aliases_keep_separate_signatures():
    checker = _checker()
    tree = ast.parse("""
from grassmoth.models.ssm.shapes import Array, Float as F, PRNGKeyArray
def build():
    def step(key: PRNGKeyArray, position: F[Array, " U"]) -> F[Array, " U"]: pass
def step(key, position): pass
class Starts:
    positions: F[Array, "B U"]
    paths: Array
""")
    slots = ("key", "position", "return")
    assert checker._missing_shape_slots(tree, "build.step", slots) == ()
    assert checker._missing_shape_slots(tree, "step", slots) == slots
    assert checker._missing_shape_slots(tree, "removed", slots) == slots
    assert checker._missing_shape_slots(tree, "Starts", ("positions", "paths")) == ("paths",)
