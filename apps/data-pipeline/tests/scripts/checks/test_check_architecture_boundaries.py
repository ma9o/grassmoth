"""Tests for the executable architecture-boundary checker."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract


@lru_cache(maxsize=1)
def _load_checker() -> Any:
    module_name = "check_architecture_boundaries_under_test"
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "checks"
        / "check_architecture_boundaries.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_module(source_root: Path, relative_path: str, source: str) -> None:
    path = source_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent
    while directory.is_relative_to(source_root):
        (directory / "__init__.py").touch()
        directory = directory.parent
    path.write_text(source, encoding="utf-8")


def test_new_modules_inherit_roles_and_unclassified_roots_fail(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(source_root, "artifacts/new_contract.py", "")
    inventory = checker.role_inventory(source_root)
    assert inventory[source_root / "artifacts/new_contract.py"] == "domain"
    assert set(inventory.values()) == {"domain"}
    _write_module(source_root, "unclassified.py", "")
    with pytest.raises(ValueError, match="Unclassified production module"):
        checker.role_inventory(source_root)


def test_type_only_dependencies_do_not_initialize_forbidden_layers(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "artifacts/new_value.py",
        """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_service.storage.backend import read_text
""",
    )
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
import typing

if typing.TYPE_CHECKING:
    import time
""",
    )
    service_root = tmp_path / "nof1_causal_service"
    _write_module(service_root, "storage/backend.py", "")

    assert checker.find_violations((source_root, service_root)) == ()


@pytest.mark.parametrize(
    "annotation",
    [
        "FitSettingsSpec",
        "dict[str, object]",
        "Options",
        "SamplerSpec | None",
        "DynamicalModelSpec",
        "ObservationLawSpec",
        "LikelihoodSpec",
        "pl.DataFrame",
        "SettingsAlias",
    ],
)
def test_execution_rejects_partial_sampler_inputs(tmp_path: Path, annotation: str) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/inference/new_sampler.py",
        f"""
from typing import TypedDict
import polars as pl
from nof1_causal_service.actions.contracts.settings import FitSettingsSpec
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.likelihood import ObservationLawSpec, LikelihoodSpec
from nof1_causal_lab.artifacts.sampling import SamplerSpec
type SettingsAlias = FitSettingsSpec
class Options(TypedDict, total=False):
    num_samples: int

def run(*, request_values: {annotation}) -> None:
    pass
""",
    )
    assert [violation.code for violation in checker.find_violations((source_root,))] == ["ARCH007"]


def test_execution_accepts_resolved_sampler_alias(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/inference/new_sampler.py",
        """
from nof1_causal_lab.artifacts.sampling import SamplerSpec as Resolved

def run(*, sampler: Resolved) -> None:
    pass
""",
    )
    assert checker.find_violations((source_root,)) == ()


def test_pure_roles_reject_transitive_function_local_acquisition(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    producers = (
        "artifacts/new_value.py",
        "models/new_compiler.py",
        "models/ssm/new_engine.py",
        "study/projections/new_projection.py",
    )
    for relative in producers:
        _write_module(source_root, relative, "from nof1_causal_lab.models.shared import read\n")
    _write_module(
        source_root,
        "models/shared.py",
        "def read():\n    from nof1_causal_service.storage.backend import read_text\n",
    )
    service_root = tmp_path / "nof1_causal_service"
    _write_module(service_root, "storage/backend.py", "import time\n")

    violations = checker.find_violations((source_root, service_root))

    for relative in producers:
        findings = [item for item in violations if item.ref.path == source_root / relative]
        assert {item.ref.imported for item in findings} == {
            "nof1_causal_service.storage.backend",
            "time",
        }
        assert all(item.code == "ARCH008" for item in findings)
        assert all("nof1_causal_lab.models.shared ->" in item.message for item in findings)
    assert all(
        "role=" in item.diagnostic(source_root) and "fix at" in item.diagnostic(source_root)
        for item in violations
    )


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/new_value.py",
        "models/new_compiler.py",
        "models/ssm/new_engine.py",
        "study/projections/new_projection.py",
    ],
)
def test_pure_roles_reject_primitive_acquisition_and_aliases(tmp_path: Path, relative: str) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        relative,
        """
import builtins as py
from datetime import datetime as Clock, date as Day
from pathlib import Path as File
from uuid import uuid4 as random_id

def derive(path: File):
    read = py.open
    read("model.json")
    input("question")
    output = py.print
    output("derived")
    now = Clock.now
    now()
    Day.today()
    random_id()
    path.read_bytes()
    (File("data") / "model.json").read_text()
    File.cwd()
""",
    )

    violations = checker.find_violations((source_root,))

    assert {item.ref.imported for item in violations} == {
        "builtins.open",
        "builtins.input",
        "builtins.print",
        "datetime.datetime.now",
        "datetime.date.today",
        "uuid.uuid4",
        "pathlib.Path.read_bytes",
        "pathlib.Path.read_text",
        "pathlib.Path.cwd",
    }
    assert len(violations) == 9
    assert all(item.code == "ARCH009" for item in violations)
    assert all("role=" in item.diagnostic(source_root) for item in violations)


def test_ambient_randomness_requires_explicit_local_seeds(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
import random as ambient
from numpy.random import default_rng as generator, SeedSequence
import numpy as np
from secrets import token_hex

def derive():
    ambient.random()
    ambient.Random()
    ambient.SystemRandom(42)
    np.random.normal()
    generator()
    generator(seed=None)
    SeedSequence()
    token_hex()
""",
    )

    violations = checker.find_violations((source_root,))

    assert [item.ref.imported for item in violations] == [
        "random.random",
        "random.Random",
        "random.SystemRandom",
        "numpy.random.normal",
        "numpy.random.default_rng",
        "numpy.random.default_rng",
        "numpy.random.SeedSequence",
        "secrets.token_hex",
    ]
    assert all(item.code == "ARCH009" for item in violations)


def test_pure_operations_shadowed_names_and_seeded_generators_are_allowed(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
from datetime import datetime as Clock, timedelta
from pathlib import Path
from random import Random
from numpy.random import default_rng, SeedSequence, Generator, PCG64

def derive(open, input, Clock, seed):
    open()
    input()
    Clock.now()
    local = Random(seed)
    local.random()
    generator = default_rng(seed=seed)
    generator.normal()
    SeedSequence(entropy=seed)
    Generator(PCG64(seed))
    return (Path("data") / "model.json").with_suffix(".csv")

def date_arithmetic():
    return Clock(2020, 1, 1) + timedelta(days=1)

def shadowed_clock():
    Clock = PureClock
    return Clock.now()
""",
    )

    assert checker.find_violations((source_root,)) == ()


def test_local_imports_assigned_paths_and_type_only_effects(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
from typing import TYPE_CHECKING as TYPES
import typing as t

if TYPES:
    from datetime import datetime
    datetime.now()
    open("types-only")

if t.TYPE_CHECKING:
    input("types-only")

def derive():
    from datetime import datetime as Clock
    import pathlib as files
    now = Clock.now
    now()
    path = files.Path("data")
    path.parent.joinpath("model.json").read_text()
    path.parents[0].read_bytes()
    other: files.Path = supplied_path()
    other.exists()

if not TYPES:
    input("runtime")
else:
    open("types-only")
""",
    )

    violations = checker.find_violations((source_root,))

    assert [item.ref.imported for item in violations] == [
        "datetime.datetime.now",
        "pathlib.Path.read_text",
        "pathlib.Path.read_bytes",
        "pathlib.Path.exists",
        "builtins.input",
    ]


def test_classes_lambdas_and_comprehensions_preserve_lexical_origins(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/new_compiler.py",
        """
from datetime import datetime as Clock

class PureClock:
    def now(self):
        return 0

class Reader:
    Clock = PureClock
    Clock.now()

    def read(self):
        return Clock.now()

def callbacks(clocks, readers):
    (lambda Clock: Clock.now())(PureClock)
    [input() for input in readers]
    {Clock.now() for Clock in clocks}
    {Clock: Clock.now() for Clock in clocks}
    tuple(Clock.now() for Clock in clocks)
    return Clock.now()

def local_import():
    from builtins import input as ask
    ask("runtime")

def separate_scope(ask):
    ask("pure callback")
""",
    )

    violations = checker.find_violations((source_root,))

    assert [item.ref.imported for item in violations] == [
        "datetime.datetime.now",
        "datetime.datetime.now",
        "builtins.input",
    ]


@pytest.mark.parametrize("relative", ["study/new_reader.py", "actions/new_action.py"])
def test_edges_and_shells_own_primitive_effects(tmp_path: Path, relative: str) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_service"
    _write_module(
        source_root,
        relative,
        """
from datetime import datetime
from pathlib import Path
from numpy.random import default_rng

def acquire():
    open("model.json")
    input("question")
    datetime.now()
    Path("model.json").read_text()
    default_rng()
""",
    )

    assert checker.find_violations((source_root,)) == ()


@pytest.mark.parametrize("type_only", [False, True])
def test_runtime_cycles_are_rejected_but_type_only_edges_are_excluded(
    tmp_path: Path, type_only: bool
) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root, "artifacts/first.py", "from nof1_causal_lab.artifacts import second\n"
    )
    prefix = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        if type_only
        else "def local_import():\n"
    )
    _write_module(
        source_root,
        "artifacts/second.py",
        prefix + "    from nof1_causal_lab.artifacts import first\n",
    )
    violations = checker.find_violations((source_root,))
    cycles = [violation for violation in violations if violation.code == "ARCH010"]
    assert len(cycles) == (0 if type_only else 1)
