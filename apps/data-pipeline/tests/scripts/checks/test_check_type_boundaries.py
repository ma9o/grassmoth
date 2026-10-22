"""Tests for the project-specific type-boundary checker."""

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
    module_name = "check_type_boundaries_under_test"
    path = Path(__file__).resolve().parents[3] / "scripts" / "checks" / "check_type_boundaries.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_typing_unions_optional_and_import_aliases_are_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
import typing as t
from typing import Optional as Maybe
from typing import Union as Either

def parse(
    first: Either[DynamicalModelSpec, dict],
    second: t.Optional[t.Union[DynamicalModelSpec, dict]],
    third: Maybe[DynamicalModelSpec],
    fourth: list[DynamicalModelSpec | dict] | None,
) -> None:
    ...
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM002"}),
    )

    assert [violation.annotation for violation in violations] == [
        "Either[DynamicalModelSpec, dict]",
        "t.Optional[t.Union[DynamicalModelSpec, dict]]",
        "DynamicalModelSpec | dict",
    ]


def test_rules_can_be_selected_independently() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
def compile_plan(plan: DynamicalModelSpec | dict[str, int] | None) -> None:
    if plan is None:
        raise ValueError("plan is required")
""",
        path="tests/example.py",
        rules=frozenset({"CUSTOM003"}),
    )

    assert [violation.code for violation in violations] == ["CUSTOM003"]
    assert violations[0].target == "parameter:plan"


def test_domain_model_must_not_be_union_member_with_dict() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
from typing import Any, Union
from pydantic import BaseModel

class Indicator(BaseModel):
    name: str

def compile_plan(
    plan: DynamicalModelSpec | dict[str, Any] | None,
    indicator: Indicator | dict[str, Any],
    legacy: Union[Indicator, dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> None:
    ...
        """,
        path="src/example.py",
        rules=frozenset({"CUSTOM002"}),
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM002",
        "CUSTOM002",
        "CUSTOM002",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:indicator",
        "parameter:legacy",
        "parameter:plan",
    }


def test_recursive_json_type_alias_is_not_treated_as_domain_union() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

def register(value: dict | Callable) -> None:
    ...
""",
        path="src/example.py",
    )

    assert violations == []


def test_reject_only_optional_parameter_is_checked() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        '''
from typing import Optional as Maybe

def compile_plan(plan: Maybe[DynamicalModelSpec]) -> None:
    """Compile an already validated plan."""
    if plan is None:
        raise ValueError("plan is required")

def compile_spec(spec: StatisticalModelSpec | None = None) -> None:
    if None is spec:
        raise ValueError("spec is required")

def compile_after_audit(plan: DynamicalModelSpec | None) -> None:
    audit_request()
    audit_complete = True
    if plan is None:
        raise ValueError("plan is required")
''',
        path="src/example.py",
    )

    assert [violation.code for violation in violations] == [
        "CUSTOM003",
        "CUSTOM003",
        "CUSTOM003",
    ]
    assert {violation.target for violation in violations} == {
        "parameter:plan",
        "parameter:spec",
    }


def test_conditional_or_non_rejecting_optional_parameter_is_allowed() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """
def conditionally_require_plan(
    plan: DynamicalModelSpec | None,
    *,
    enabled: bool,
) -> None:
    if enabled:
        if plan is None:
            raise ValueError("enabled compilation requires a plan")

def default_plan(plan: DynamicalModelSpec | None = None) -> None:
    if plan is None:
        return

def return_before_rejection(plan: DynamicalModelSpec | None, enabled: bool) -> None:
    if not enabled:
        return
    if plan is None:
        raise ValueError("plan is required")

def replace_before_rejection(plan: DynamicalModelSpec | None) -> None:
    if plan is None:
        plan = default_plan()
    if plan is None:
        raise ValueError("plan is required")
""",
        path="src/example.py",
        rules=frozenset({"CUSTOM003"}),
    )

    assert violations == []


@pytest.mark.parametrize(
    "operation",
    [
        "value.model_construct()",
        "value.model_copy(update={})",
        "value.model_copy(**changes)",
        "object.__setattr__(value, 'question', 'changed')",
        "Evidence(spec=value)",
        "IncompleteModel('missing')",
        "UnsupportedFit(('unsupported',))",
        "cast('Evidence', value)",
        "cast('IncompleteModel', value)",
        "cast('UnsupportedFit', value)",
        "cast('list[int]', value)",
        "cast('dict[str, str]', value)",
        "cast('set[int]', value)",
        "cast('MutableSequence[int]', value)",
    ],
)
def test_core_bypasses_match_calls_without_receiver_types(operation: str) -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        f"""from typing import cast
from collections.abc import MutableSequence
from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs as Evidence, IncompleteModel, UnsupportedFit

def consumer(value):
    return {operation}
""",
        path="tests/consumer.py",
        rules=frozenset({"CORE001"}),
    )
    assert len(violations) == 1
    assert violations[0].code == "CORE001"
    assert "owner" in violations[0].message


def test_core_bypasses_reject_self_updates_and_allow_validated_revisions() -> None:
    checker = _load_checker()
    source = """class Request:
    def revise(self, value, changes):
        self.model_copy(update={})
        self.model_copy(**changes)
        object.__setattr__(self, 'question', 'changed')
        self.revised(question='changed')
        value.model_copy()
        cast('Mapping[str, int]', value)
"""
    violations = checker.scan_text(
        source,
        path="src/nof1_causal_lab/artifacts/model/dynamical_model_spec.py",
        rules=frozenset({"CORE001"}),
    )
    assert [item.target for item in violations] == ["model_copy", "model_copy", "__setattr__"]


def test_core_initialization_is_local_to_constructor_scope() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """class Owned:
    def __post_init__(self):
        object.__setattr__(self, 'items', immutable_items)
        def revise_later():
            object.__setattr__(self, 'items', mutable_items)
""",
        path="src/nof1_causal_lab/models/new_owner.py",
        rules=frozenset({"CORE001"}),
    )
    assert [item.target for item in violations] == ["__setattr__"]


def test_core_owner_file_only_exempts_evidence_construction() -> None:
    checker = _load_checker()
    owner = Path(__file__).resolve().parents[3] / "src/nof1_causal_lab/models/ssm/compile/inputs.py"
    violations = checker.scan_text(
        owner.read_text()
        + """\nCompiledFitInputs()
IncompleteModel('missing')
UnsupportedFit(('unsupported',))
value.model_copy(update={})
value.model_construct()
cast('CompiledFitInputs', value)
""",
        path="src/nof1_causal_lab/models/ssm/compile/inputs.py",
        rules=frozenset({"CORE001"}),
    )
    assert [v.target for v in violations] == ["model_copy", "model_construct", "cast"]


def test_owned_value_inheritance_and_collection_aliases_bind_new_modules() -> None:
    violations = _load_checker().scan_text(
        """from nof1_causal_lab.artifacts.base import Value as Frozen
from collections.abc import Mapping
type Builder = dict[str, list[int]]
class Intermediate(Frozen):
    pass
class NewContract(Intermediate):
    items: tuple[Mapping[str, Builder], ...]
""",
        path="src/nof1_causal_lab/artifacts/new_contract.py",
        rules=frozenset({"CORE002", "IMM001"}),
    )
    assert [(item.code, item.target) for item in violations] == [("CORE002", "field:items")]
    assert "role=domain" in violations[0].diagnostic()
    assert "value owner" in violations[0].diagnostic()


def test_owned_values_cannot_override_freezing() -> None:
    violations = _load_checker().scan_text(
        """from pydantic import ConfigDict
from nof1_causal_lab.artifacts.base import Value
class NewContract(Value):
    model_config = ConfigDict(frozen=False)
    value: int
""",
        path="src/nof1_causal_lab/artifacts/new_contract.py",
        rules=frozenset({"IMM001"}),
    )
    assert [item.code for item in violations] == ["IMM001"]


def test_interpreted_contracts_share_the_value_configuration() -> None:
    violations = _load_checker().scan_text(
        """from pydantic import BaseModel, ConfigDict
class NewContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int
""",
        path="src/nof1_causal_lab/artifacts/new_contract.py",
        rules=frozenset({"IMM001"}),
    )
    assert [item.code for item in violations] == ["IMM001"]


@pytest.mark.parametrize(
    "caught", ["ValueError", "KeyError", "Exception", "(TypeError, ValueError)"]
)
def test_domain_builtin_catches_are_not_scientific_decisions(caught: str) -> None:
    violations = _load_checker().scan_text(
        f"try:\n    execute_science()\nexcept {caught}:\n    return_unavailable()\n",
        path="src/nof1_causal_lab/artifacts/new_contract.py",
        rules=frozenset({"ERR001"}),
    )
    assert [item.code for item in violations] == ["ERR001"]


def test_expected_syntax_error_is_local_to_one_parse_operation() -> None:
    checker = _load_checker()
    permitted = "import ast\ntry:\n    expression = ast.parse(text)\nexcept SyntaxError:\n    reject_input()\n"
    assert (
        checker.scan_text(
            permitted,
            path="src/nof1_causal_lab/artifacts/new_contract.py",
            rules=frozenset({"ERR001"}),
        )
        == []
    )
    extended = permitted.replace(
        "    expression = ast.parse(text)",
        "    expression = ast.parse(text)\n    execute_science()",
    )
    assert [
        item.code
        for item in checker.scan_text(
            extended,
            path="src/nof1_causal_lab/artifacts/new_contract.py",
            rules=frozenset({"ERR001"}),
        )
    ] == ["ERR001"]


def test_core_collections_cover_private_owned_data_and_properties() -> None:
    checker = _load_checker()
    violations = checker.scan_text(
        """from typing import Dict as MutableDict
from nof1_causal_lab.artifacts.base import Value
class DynamicalModelSpec(Value):
    entries: tuple[MutableDict[str, int], ...]
    readonly: Mapping[str, int]
    _builder: dict[str, int]
    @cached_property
    def derived(self) -> dict[str, int]: ...
    @property
    def safe(self) -> tuple[int, ...]: ...
    def _build(self) -> list[int]: ...
    def edit(self) -> None:
        local: list[int] = []
class Request:
    entries: dict[str, int]
""",
        path="src/nof1_causal_lab/artifacts/model/dynamical_model_spec.py",
        rules=frozenset({"CORE002"}),
    )
    assert [(v.code, v.target) for v in violations] == [
        ("CORE002", "field:entries"),
        ("CORE002", "field:_builder"),
        ("CORE002", "return"),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "raise ValueError('invalid')",
        "assert model.indicators",
        "model.require_priors()",
        "validate_execution(model)",
        "validate_parameter_anchors(model)",
        "Result.model_validate(payload)",
        "Result.model_validate_json(payload)",
        "Result.model_validate_strings(payload)",
    ],
)
def test_pure_projection_scope_rejects_revalidation(operation: str) -> None:
    checker = _load_checker()
    source = f"def newly_added_projection(model: DynamicalModelSpec, payload):\n    {operation}\n"
    assert [
        v.code
        for v in checker.scan_text(
            source,
            path="src/nof1_causal_lab/study/projections/visuals.py",
            rules=frozenset({"VIEW001"}),
        )
    ] == ["VIEW001"]
    assert checker.scan_text(source, path="src/boundary.py", rules=frozenset({"VIEW001"})) == []


def test_projection_allows_exhaustiveness() -> None:
    checker = _load_checker()
    assert (
        checker.scan_text(
            "def project(value):\n    assert_never(value)\n",
            path="src/nof1_causal_lab/study/projections/visuals.py",
            rules=frozenset({"VIEW001"}),
        )
        == []
    )


@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        ("import contextlib", "contextlib.suppress"),
        ("from contextlib import suppress as ignore", "ignore"),
    ],
)
def test_suppress_obeys_the_same_builtin_error_rule(prefix: str, call: str) -> None:
    source = f"{prefix}\nwith {call}(ValueError, KeyError):\n    classify_science()\n"
    violations = _load_checker().scan_text(
        source, path="src/nof1_causal_lab/artifacts/reports/checks.py", rules=frozenset({"ERR001"})
    )
    assert [v.code for v in violations] == ["ERR001"]


@pytest.mark.parametrize("role", ["models/ssm/compile", "models/ssm/execution", "study"])
def test_constructor_parsing_stays_with_the_target_owner_or_edge(role: str) -> None:
    package = "nof1_causal_service" if role == "study" else "nof1_causal_lab"
    violations = _load_checker().scan_text(
        """from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec as Authored
from pydantic import TypeAdapter

def read(payload):
    value = Authored.model_validate(payload)
    return TypeAdapter(Authored).validate_python(payload)
""",
        path=f"src/{package}/{role}/new_parser.py",
        rules=frozenset({"PARSE001"}),
    )
    assert [item.code for item in violations] == (
        ["PARSE001", "PARSE001"] if role != "study" else []
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
def test_type_adapter_parses_only_at_its_value_owner(cached: bool, foreign: bool) -> None:
    target = "Foreign" if foreign else "Outcome"
    receiver = "adapter" if cached else f"TypeAdapter({target})"
    source = f"""from pydantic import TypeAdapter
from nof1_causal_lab.artifacts.model.question import QuestionSpec as Foreign
from nof1_causal_lab.artifacts.base import Value

class Ready(Value):
    value: int
class Rejected(Value):
    detail: str

type Outcome = Ready | Rejected
adapter: TypeAdapter[{target}] = TypeAdapter({target})

def parse(payload):
    return {receiver}.validate_python(payload)
"""
    violations = _load_checker().scan_text(
        source,
        path="src/nof1_causal_service/actions/contracts/new_owner.py",
        rules=frozenset({"PARSE001"}),
    )
    assert [item.code for item in violations] == (["PARSE001"] if foreign else [])


@pytest.mark.parametrize("directory", ["models/ssm/compile", "models/ssm/execution"])
@pytest.mark.parametrize(
    "result",
    [
        "Compiled",
        "Compiled | Unavailable",
        "tuple[Compiled, str]",
        "tuple[Compiled, Unavailable]",
        "Mapping[str, Compiled]",
    ],
)
def test_published_results_protect_composed_aliases(directory: str, result: str) -> None:
    violations = _load_checker().scan_text(
        f"""from dataclasses import dataclass

type Contents = dict[str, int]
@dataclass(frozen=True)
class Block:
    items: Contents
@dataclass(frozen=True)
class Compiled:
    block: Block
@dataclass(frozen=True)
class Unavailable:
    reason: str

def compile_model() -> {result}:
    return Unavailable("missing")
""",
        path=f"src/nof1_causal_lab/{directory}/new_owner.py",
        rules=frozenset({"IMM001", "CORE002"}),
    )
    assert [(item.code, item.target) for item in violations] == [("CORE002", "field:items")]


@pytest.mark.parametrize(
    "directory",
    [
        "observations",
        "actions",
        "study/projections/visuals",
        "models/ssm/compile",
        "models/ssm/execution",
    ],
)
def test_json_decoding_obeys_whole_role_parse_rules(directory: str) -> None:
    checker = _load_checker()
    # study/projections/visuals is explicitly projection-owned; actions and utils use their directory role.
    path = (
        "src/nof1_causal_lab/study/projections/visuals.py"
        if directory == "study/projections/visuals"
        else f"src/nof1_causal_lab/{directory}/new_consumer.py"
    )
    violations = checker.scan_text(
        "import json as wire\ndef consume(text):\n    return wire.loads(text)\n",
        path=path,
        rules=frozenset({"PARSE001"}),
    )
    assert [violation.code for violation in violations] == ["PARSE001"]


@pytest.mark.parametrize(
    "path",
    [
        "src/nof1_causal_service/study/new_reader.py",
        "scripts/fixtures/example.py",
        "evaluation/fixtures/example.py",
        "notebooks/example.py",
        "tests/schemas/example.py",
    ],
)
def test_dump_spread_reconstruction_is_rejected_across_checked_python_trees(path: str) -> None:
    violations = _load_checker().scan_text(
        "def read(value):\n    return Result.model_validate({**value.model_dump(), 'changed': 1})\n",
        path=path,
        rules=frozenset({"CORE001"}),
    )
    assert [violation.target for violation in violations] == ["dump-spread"]


def test_declared_failure_handler_does_not_authorize_nested_scientific_catches() -> None:
    source = """from nof1_causal_service.actions.errors import execution_failure_handler as shell_failure
@shell_failure
def execute():
    try:
        foreign_execution()
    except Exception:
        record_failure()
    def scientific_decision():
        try:
            evaluate()
        except Exception:
            publish_scientific_rejection()
"""
    checker = _load_checker()
    violations = checker.scan_text(
        source, path="src/nof1_causal_service/actions/new_runner.py", rules=frozenset({"ERR001"})
    )
    assert len(violations) == 1
    assert "scientific_decision" in violations[0].scope
    # The marker cannot exempt a compiler or domain module.
    assert (
        len(
            checker.scan_text(
                source,
                path="src/nof1_causal_lab/models/new_compiler.py",
                rules=frozenset({"ERR001"}),
            )
        )
        == 2
    )


@pytest.mark.parametrize("annotation", ["Mapping[str, tuple[np.ndarray, ...]]", "np.ndarray"])
def test_compiled_records_must_own_nested_mapping_and_numpy_storage(annotation: str) -> None:
    source = f"""from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from nof1_causal_lab.artifacts.immutability import freeze_fields as own_fields

@dataclass(frozen=True)
class Compiled:
    values: {annotation}

def compile_values() -> Compiled:
    return Compiled(values)
"""
    checker = _load_checker()
    path = "src/nof1_causal_lab/models/new_owner.py"
    violations = checker.scan_text(source, path=path, rules=frozenset({"IMM002"}))
    assert [item.target for item in violations] == ["Compiled"]
    initialized = source.replace(
        "\ndef compile_values",
        "\n    def __post_init__(self):\n        own_fields(self)\n\ndef compile_values",
    )
    assert checker.scan_text(initialized, path=path, rules=frozenset({"IMM002"})) == []
    skipped = initialized.replace(
        "        own_fields(self)", "        if skip:\n            return\n        own_fields(self)"
    )
    assert [
        item.code for item in checker.scan_text(skipped, path=path, rules=frozenset({"IMM002"}))
    ] == ["IMM002"]


def test_jax_buffers_and_callable_arguments_need_no_numpy_ownership() -> None:
    source = """from dataclasses import dataclass
from collections.abc import Callable, Mapping
import jax.numpy as jnp
import numpy as np
@dataclass(frozen=True)
class Compiled:
    values: jnp.ndarray
    compute: Callable[[Mapping[str, np.ndarray]], jnp.ndarray]
def compile_values() -> Compiled:
    return Compiled(values, compute)
"""
    assert (
        _load_checker().scan_text(
            source, path="src/nof1_causal_lab/models/new_owner.py", rules=frozenset({"IMM002"})
        )
        == []
    )


@pytest.mark.parametrize("outcome", ["Ready | Rejected", "Optional[Rejected]", "Outcome"])
def test_declared_sum_outcomes_cannot_be_discarded(outcome: str) -> None:
    source = f"""from dataclasses import dataclass
from typing import Optional
@dataclass(frozen=True)
class Ready:
    value: int
@dataclass(frozen=True)
class Rejected:
    message: str
type Outcome = Ready | Rejected
def decide() -> {outcome}:
    return Rejected('missing')
def caller():
    decide()
    _ = decide()
    result = decide()
    return result
def callback(decide):
    decide()
"""
    violations = _load_checker().scan_text(
        source, path="src/nof1_causal_lab/models/new_owner.py", rules=frozenset({"ERR002"})
    )
    assert len(violations) == 2
    assert all(item.scope == "caller" for item in violations)


@pytest.mark.parametrize("directory", ["models", "actions"])
def test_field_ownership_cannot_be_reused_for_later_mutation(directory: str) -> None:
    violations = _load_checker().scan_text(
        """from dataclasses import dataclass
from collections.abc import Mapping
from nof1_causal_lab.artifacts.immutability import freeze_fields
@dataclass(frozen=True)
class Compiled:
    values: Mapping[str, int]
    def __post_init__(self):
        freeze_fields(self)
    def revise(self):
        freeze_fields(self)
def compile_values() -> Compiled:
    return Compiled(values)
""",
        path=f"src/nof1_causal_lab/{directory}/new_owner.py",
        rules=frozenset({"CORE001"}),
    )
    assert [item.target for item in violations] == ["freeze_fields"]
