"""Exhaustive theoretical tests for identifiability logic.

This test suite verifies the theoretical soundness of our identification logic
by testing against classic causal inference scenarios, complex graph topologies,
and edge cases. All tests are derived from Pearl's do-calculus literature and
the y0 package's identification algorithms.

Most cases share the same shape: build a latent + measurement structure, call
``check_identifiability``, then assert facts about the result. They are
expressed as a data table driving a single parametrized test
(``test_identification``). Tests using other entry points
(``StructuralSelection.marginalized_construct_ids``, ``unroll_temporal_dag``, ``dag_to_admg``,
``find_blocking_confounders``) live below as small parametrized or standalone
tests.

References:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Shpitser & Pearl (2006). Identification of Joint Interventional Distributions
- arXiv:2504.20172 - Jahn, Karnik & Schulman on bounded latent reach
"""

from typing import Any

import pytest

from grassmoth.artifacts.identity import ConstructId
from grassmoth.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    replace_constructs,
)
from grassmoth.artifacts.reports.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
)
from grassmoth.models.causal.identification import (
    check_identifiability,
    dag_to_admg,
    find_blocking_confounders,
    unroll_temporal_dag,
)
from grassmoth.models.structure import StructuralSelection, unsupported_construct_ids
from grassmoth_test_support.causal_graphs import make_graph
from grassmoth_test_support.helpers import fixture_entity_id, make_model

pytestmark = pytest.mark.contract

# =============================================================================
# HELPERS
# =============================================================================


def _construct(name: str) -> ConstructId:
    return fixture_entity_id("construct", name)


def _get_estimand(result: IdentificationReport, treatment: str) -> str:
    finding = result.treatments[_construct(treatment)]
    assert isinstance(finding, IdentifiedTreatmentStatus)
    return finding.estimand


def _get_blockers(result: IdentificationReport, treatment: str) -> tuple[ConstructId, ...]:
    return result.non_identifiable[_construct(treatment)].confounders


def _run_checks(
    result: IdentificationReport,
    checks: list[tuple[Any, ...]],
) -> None:
    """Dispatch a list of check tuples against a check_identifiability result.

    Supported check kinds:
        ("identifiable", treatment)
        ("not_identifiable", treatment)
        ("blocked_by", treatment, blocker)
        ("estimand_contains", treatment, substring)        -- case-sensitive
        ("estimand_contains_ci", treatment, substring)     -- case-insensitive
        ("treatment_absent", treatment)
        ("blocker_in", treatment, [candidates])            -- any of these
        ("no_treatments_at_all",)
        ("no_identifiable_treatments",)
    """
    for check in checks:
        kind = check[0]
        if kind == "identifiable":
            (_, t) = check
            assert _construct(t) in result.estimable_treatments, (
                f"{t} should be identifiable. Result: {result}"
            )
        elif kind == "not_identifiable":
            (_, t) = check
            assert _construct(t) in result.non_identifiable, (
                f"{t} should NOT be identifiable. Result: {result}"
            )
        elif kind == "blocked_by":
            (_, t, blocker) = check
            details = result.non_identifiable.get(_construct(t))
            assert details, f"{t} should have blocking confounders. Result: {result}"
            blockers = details.confounders
            assert _construct(blocker) in blockers, (
                f"{t} should be blocked by {blocker}. Blockers: {blockers}"
            )
        elif kind == "estimand_contains":
            (_, t, sub) = check
            est = _get_estimand(result, t)
            assert sub in est, f"Estimand for {t} should contain {sub!r}. Got: {est!r}"
        elif kind == "estimand_contains_ci":
            (_, t, sub) = check
            est = _get_estimand(result, t).lower()
            assert sub.lower() in est, (
                f"Estimand for {t} should contain {sub!r} (case-insensitive). Got: {est!r}"
            )
        elif kind == "treatment_absent":
            (_, t) = check
            assert _construct(t) not in result.treatments
        elif kind == "blocker_in":
            (_, t, candidates) = check
            blockers = _get_blockers(result, t)
            assert any(_construct(b) in blockers for b in candidates), (
                f"{t} blockers should include one of {candidates}. Got: {blockers}"
            )
        elif kind == "no_treatments_at_all":
            assert len(result.treatments) == 0
        elif kind == "no_identifiable_treatments":
            assert len(result.estimable_treatments) == 0
        else:
            raise AssertionError(f"Unknown check kind: {kind}")


# =============================================================================
# IDENTIFICATION CASES
#
# Each case is a dict:
#   id:          pytest test id
#   constructs:  list of {name, [role], [temporal_status]}
#   outcome:     persistent reference to the question's outcome
#   edges:       list of {cause, effect}
#   observed:    list of construct names with measurement indicators
#   checks:      list of check tuples (see _run_checks for kinds)
# =============================================================================

IDENTIFICATION_CASES: list[dict[str, Any]] = [
    # ---- 1. Classic Pearl Graphs ------------------------------------------
    # Bow graph: X->Y, U->X, U->Y. Simplest non-identifiable structure.
    {
        "outcome": "construct:Y",
        "id": "pearl_bow_non_identifiable",
        "constructs": [{"name": "X"}, {"name": "Y"}, {"name": "U"}],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "U")],
    },
    # Chain with confounding at every step. Front-door fails (M confounded with Y).
    {
        "outcome": "construct:Y",
        "id": "pearl_confounded_chain_non_identifiable",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "M"},
            {"cause": "U2", "effect": "M"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # Verma-constraint graph: W->X->Y->Z with U1 confounding X-Z. W identifies X.
    {
        "outcome": "construct:Y",
        "id": "pearl_verma_constraint",
        "constructs": [
            {"name": "W", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "Z"},
            {"name": "U1"},
        ],
        "edges": [
            {"cause": "W", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "Z"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Z"},
        ],
        "observed": ["W", "X", "Y", "Z"],
        "checks": [("identifiable", "X"), ("identifiable", "W")],
    },
    # ---- 2. Backdoor Criterion --------------------------------------------
    {
        "outcome": "construct:Y",
        "id": "backdoor_observed_confounder",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "Z", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [("identifiable", "X"), ("identifiable", "Z")],
    },
    {
        "outcome": "construct:Y",
        "id": "backdoor_multiple_confounders_all_observed",
        "constructs": [
            {"name": "Z1", "role": "endogenous"},
            {"name": "Z2", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "Z1", "effect": "X"},
            {"cause": "Z2", "effect": "X"},
            {"cause": "Z2", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Z1", "Z2", "X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Z1 is an IV candidate; it cannot identify X without extra assumptions.
    {
        "outcome": "construct:Y",
        "id": "backdoor_unobserved_but_iv_available",
        "constructs": [
            {"name": "Z1", "role": "endogenous"},
            {"name": "U", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "Z1", "effect": "X"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Z1", "X", "Y"],
        "checks": [("not_identifiable", "X"), ("identifiable", "Z1")],
    },
    # Z -> W, W -> {X, Y}; adjusting for W blocks the backdoor.
    {
        "outcome": "construct:Y",
        "id": "backdoor_chain_of_confounders",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "W"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "Z", "effect": "W"},
            {"cause": "W", "effect": "X"},
            {"cause": "W", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Z", "W", "X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # ---- 3. Front-Door Criterion ------------------------------------------
    # Classic front-door: X->M->Y with U->X, U->Y.
    {
        "outcome": "construct:Y",
        "id": "frontdoor_classic",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("identifiable", "X"), ("estimand_contains", "X", "M")],
    },
    # U -> M breaks front-door condition.
    {
        "outcome": "construct:Y",
        "id": "frontdoor_fails_if_mediator_confounded",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
            {"cause": "U", "effect": "M"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "frontdoor_with_multiple_mediators",
        "constructs": [
            {"name": "X"},
            {"name": "M1"},
            {"name": "M2"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "X", "effect": "M1"},
            {"cause": "X", "effect": "M2"},
            {"cause": "M1", "effect": "Y"},
            {"cause": "M2", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M1", "M2", "Y"],
        "checks": [("identifiable", "X")],
    },
    # ---- 4. Instrumental Variables ----------------------------------------
    # Classic IV: Z -> X -> Y, U -> X, U -> Y. IV identification under linearity.
    {
        "outcome": "construct:Y",
        "id": "iv_classic",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [
            ("not_identifiable", "X"),
            ("identifiable", "Z"),
        ],
    },
    # U -> Z breaks IV exogeneity.
    {
        "outcome": "construct:Y",
        "id": "iv_fails_if_instrument_confounded",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
            {"cause": "U", "effect": "Z"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # Z -> Y violates IV exclusion. Z's own effect on Y is still identifiable.
    {
        "outcome": "construct:Y",
        "id": "iv_fails_if_direct_path_to_outcome",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "Z", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [("not_identifiable", "X"), ("identifiable", "Z")],
    },
    # ---- 5. Temporal Dynamics (AR(1) under A3a) ---------------------------
    {
        "outcome": "construct:Y",
        "id": "temporal_static_confounding_blocks_id",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_invariant"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "U")],
    },
    {
        "outcome": "construct:Y",
        "id": "temporal_dynamic_common_cause",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # AR(1) enables identification: conditioning on X_{t-1} blocks U_{t-1}->X_{t-1}->X_t.
    {
        "outcome": "construct:Y",
        "id": "temporal_ar1_enables_id_via_lagged_adjustment",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Staggered: U_{t-1}->X_t, U_t->Y_t. Lagged adjustment still works.
    {
        "outcome": "construct:Y",
        "id": "temporal_staggered_id_via_lagged_adjustment",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "temporal_lagged_treatment_observed",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "temporal_declared_direct_effect",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # X_t->Y_t with Y_{t-1}->X_t (acyclic when unrolled).
    {
        "outcome": "construct:Y",
        "id": "temporal_feedback_loop",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "X"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # ---- 6. Time-Invariant Constructs -------------------------------------
    {
        "outcome": "construct:Y",
        "id": "tinv_confounder_observed",
        "constructs": [
            {
                "id": "construct:ca7081092c417bf11cf3",
                "name": "Trait",
                "role": "endogenous",
                "temporal_status": "time_invariant",
            },
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "Trait", "effect": "X"},
            {"cause": "Trait", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Trait", "X", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "tinv_confounder_unobserved",
        "constructs": [
            {
                "id": "construct:ca7081092c417bf11cf3",
                "name": "Trait",
                "role": "endogenous",
                "temporal_status": "time_invariant",
            },
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "Trait", "effect": "X"},
            {"cause": "Trait", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "Trait")],
    },
    {
        "outcome": "construct:Y",
        "id": "tinv_treatment",
        "constructs": [
            {"name": "Treatment", "temporal_status": "time_invariant"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [{"cause": "Treatment", "effect": "Y"}],
        "observed": ["Treatment", "Y"],
        "checks": [("identifiable", "Treatment")],
    },
    {
        "outcome": "construct:Y",
        "id": "tinv_mixed_status_chain",
        "constructs": [
            {
                "id": "construct:ca7081092c417bf11cf3",
                "name": "Trait",
                "role": "endogenous",
                "temporal_status": "time_invariant",
            },
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "Trait", "effect": "X"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Trait", "X", "Y"],
        "checks": [("identifiable", "X"), ("identifiable", "Trait")],
    },
    # ---- 7. Complex Confounding Patterns ----------------------------------
    # Diamond X -> {A, B} -> Y, all observed.
    {
        "outcome": "construct:Y",
        "id": "complex_diamond_all_observed",
        "constructs": [
            {"name": "X"},
            {"name": "A"},
            {"name": "B"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X", "effect": "A"},
            {"cause": "X", "effect": "B"},
            {"cause": "A", "effect": "Y"},
            {"cause": "B", "effect": "Y"},
        ],
        "observed": ["X", "A", "B", "Y"],
        "checks": [("identifiable", "X")],
    },
    # U1 -> U2; U2 -> {X, Y}. Confounder is U2 (not U1).
    {
        "outcome": "construct:Y",
        "id": "complex_chain_of_unobserved",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "U2"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "U2")],
    },
    # M-bias structure: pre-treatment A and B, no direct X-Y backdoor.
    {
        "outcome": "construct:Y",
        "id": "complex_m_bias_structure",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "A"},
            {"name": "B"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "A"},
            {"cause": "A", "effect": "X"},
            {"cause": "U2", "effect": "B"},
            {"cause": "B", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["A", "B", "X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # M-bias with U3 confounding X-Y; an IV candidate does not remove that obstruction.
    {
        "outcome": "construct:Y",
        "id": "complex_m_bias_with_iv_available",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "U3"},
            {"name": "A"},
            {"name": "B"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "A"},
            {"cause": "A", "effect": "X"},
            {"cause": "U2", "effect": "B"},
            {"cause": "B", "effect": "Y"},
            {"cause": "U3", "effect": "X"},
            {"cause": "U3", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["A", "B", "X", "Y"],
        "checks": [("not_identifiable", "X"), ("identifiable", "A")],
    },
    # Two independent unobserved confounders.
    {
        "outcome": "construct:Y",
        "id": "complex_multiple_disjoint_confounders",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Y"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [
            ("not_identifiable", "X"),
            ("blocker_in", "X", ["U1", "U2"]),
        ],
    },
    # ---- 8. Collider Structures -------------------------------------------
    # X -> C <- Y collider; X is independent of Y through C.
    {
        "outcome": "construct:Y",
        "id": "collider_simple",
        "constructs": [
            {"name": "X"},
            {"name": "C"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X", "effect": "C"},
            {"cause": "Y", "effect": "C"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "C", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Collider with descendant: X -> C <- U -> Y, C -> D. Path blocked at collider.
    {
        "outcome": "construct:Y",
        "id": "collider_with_descendant",
        "constructs": [
            {"name": "X"},
            {"name": "C"},
            {"name": "D"},
            {"name": "U"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X", "effect": "C"},
            {"cause": "U", "effect": "C"},
            {"cause": "U", "effect": "Y"},
            {"cause": "C", "effect": "D"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "C", "D", "Y"],
        "checks": [("identifiable", "X")],
    },
    # ---- 9. Multiple Treatments -------------------------------------------
    {
        "outcome": "construct:Y",
        "id": "multi_treatments_all_identifiable",
        "constructs": [
            {"name": "X1"},
            {"name": "X2"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X1", "effect": "Y"},
            {"cause": "X2", "effect": "Y"},
        ],
        "observed": ["X1", "X2", "Y"],
        "checks": [("identifiable", "X1"), ("identifiable", "X2")],
    },
    {
        "outcome": "construct:Y",
        "id": "multi_treatments_some_identifiable",
        "constructs": [
            {"name": "X1"},
            {"name": "X2"},
            {"name": "U"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X1", "effect": "Y"},
            {"cause": "X2", "effect": "Y"},
            {"cause": "U", "effect": "X2"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X1", "X2", "Y"],
        "checks": [("identifiable", "X1"), ("not_identifiable", "X2")],
    },
    {
        "outcome": "construct:Y",
        "id": "multi_treatment_chain",
        "constructs": [
            {"name": "X1"},
            {"name": "X2"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X1", "effect": "X2"},
            {"cause": "X2", "effect": "Y"},
        ],
        "observed": ["X1", "X2", "Y"],
        "checks": [("identifiable", "X1"), ("identifiable", "X2")],
    },
    # ---- 10. Edge Cases ---------------------------------------------------
    {
        "outcome": "construct:Y",
        "id": "edge_outcome_only_no_treatments",
        "constructs": [{"name": "Y"}],
        "edges": [],
        "observed": ["Y"],
        "checks": [("no_treatments_at_all",)],
    },
    {
        "outcome": "construct:Y",
        "id": "edge_unobserved_outcome",
        "constructs": [{"name": "X"}, {"name": "Y"}],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["X"],
        "checks": [("no_identifiable_treatments",)],
    },
    {
        "outcome": "construct:Y",
        "id": "edge_all_unobserved_except_outcome",
        "constructs": [{"name": "X"}, {"name": "Y"}],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["Y"],
        "checks": [("treatment_absent", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "edge_long_causal_chain",
        "constructs": [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
            {"name": "E"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "A", "effect": "B"},
            {"cause": "B", "effect": "C"},
            {"cause": "C", "effect": "D"},
            {"cause": "D", "effect": "E"},
            {"cause": "E", "effect": "Y"},
        ],
        "observed": ["A", "B", "C", "D", "E", "Y"],
        "checks": [
            ("identifiable", "A"),
            ("identifiable", "B"),
            ("identifiable", "C"),
            ("identifiable", "D"),
            ("identifiable", "E"),
        ],
    },
    {
        "outcome": "construct:Y",
        "id": "edge_isolated_construct",
        "constructs": [
            {"name": "X"},
            {"name": "Isolated"},
            {"name": "Y"},
        ],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["X", "Isolated", "Y"],
        "checks": [("treatment_absent", "Isolated"), ("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "edge_no_path_to_outcome",
        "constructs": [
            {"name": "X"},
            {"name": "Z"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "Z", "effect": "X"},
        ],
        "observed": ["X", "Z", "Y"],
        "checks": [("identifiable", "Z"), ("identifiable", "X")],
    },
    # ---- 14. Complex Hedge Structures -------------------------------------
    # Napkin-like: collider W2 blocks the backdoor.
    {
        "outcome": "construct:Y",
        "id": "hedge_napkin_like_identifiable",
        "constructs": [
            {"name": "X"},
            {"name": "W1"},
            {"name": "W2"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "W1", "effect": "W2"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "W1"},
            {"cause": "U2", "effect": "W2"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "W1", "W2", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Kite: confounding triangle on chain breaks front-door.
    {
        "outcome": "construct:Y",
        "id": "hedge_kite_graph",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "M"},
            {"cause": "U2", "effect": "M"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # Two stacked bow graphs.
    {
        "outcome": "construct:Y",
        "id": "hedge_double_bow",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "M"},
            {"cause": "U2", "effect": "M"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # W-structure: Z is a collider between U1 and U2 paths.
    {
        "outcome": "construct:Y",
        "id": "hedge_w_structure",
        "constructs": [
            {"name": "X"},
            {"name": "Z"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Z"},
            {"cause": "U2", "effect": "Z"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "Z", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Verma extended; W can serve as IV/adjustment for X.
    {
        "outcome": "construct:Y",
        "id": "hedge_verma_extended",
        "constructs": [
            {"name": "V", "role": "endogenous"},
            {"name": "W", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "Z"},
            {"name": "U1"},
        ],
        "edges": [
            {"cause": "V", "effect": "U1"},
            {"cause": "W", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "Z"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Z"},
        ],
        "observed": ["V", "W", "X", "Y", "Z"],
        "checks": [("identifiable", "X")],
    },
    # ---- 15. Conditional / Complex IV Scenarios ---------------------------
    # Conditional IV (C is needed) — y0 doesn't find it. Documents the limit.
    {
        "outcome": "construct:Y",
        "id": "iv_conditional_not_supported",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "C", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U1"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "C", "effect": "Z"},
            {"cause": "C", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Y"},
        ],
        "observed": ["Z", "C", "X", "Y"],
        "checks": [
            ("not_identifiable", "X"),
            ("identifiable", "Z"),
            ("identifiable", "C"),
        ],
    },
    {
        "outcome": "construct:Y",
        "id": "iv_multiple_instruments",
        "constructs": [
            {"name": "Z1", "role": "endogenous"},
            {"name": "Z2", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z1", "effect": "X"},
            {"cause": "Z2", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z1", "Z2", "X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # W is an IV candidate, but U2 still prevents nonparametric identification of X.
    {
        "outcome": "construct:Y",
        "id": "iv_weak_instrument_chain",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "W"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "Z", "effect": "W"},
            {"cause": "W", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U1", "effect": "W"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["Z", "W", "X", "Y"],
        "checks": [("not_identifiable", "X"), ("not_identifiable", "W")],
    },
    # ---- 16. Temporal Complexity (panel data) -----------------------------
    {
        "outcome": "construct:Y",
        "id": "tcomplex_cross_lagged_panel_no_confounding",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "X"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    # CLPM with unobserved trait confounding (RI-CLPM motivation).
    {
        "outcome": "construct:Y",
        "id": "tcomplex_cross_lagged_with_trait_confounding",
        "constructs": [
            {
                "id": "construct:ca7081092c417bf11cf3",
                "name": "Trait",
                "role": "endogenous",
                "temporal_status": "time_invariant",
            },
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "Trait", "effect": "X"},
            {"cause": "Trait", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "X"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "Trait")],
    },
    # Front-door with temporal carry-over from M.
    {
        "outcome": "construct:Y",
        "id": "tcomplex_temporal_front_door",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "M", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "tcomplex_feedback_confounded",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_invariant"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "X"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "tcomplex_lagged_instrument_temporal",
        "constructs": [
            {"name": "Z", "temporal_status": "time_varying"},
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_invariant"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # ---- 17. Overlapping Confounders --------------------------------------
    # U1 -> {X, M}, U2 -> {M, Y}, X -> M -> Y.
    {
        "outcome": "construct:Y",
        "id": "overlap_partial_confounding_coverage",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "M"},
            {"cause": "U2", "effect": "M"},
            {"cause": "U2", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    # Triangle with three confounders, every pair confounded.
    {
        "outcome": "construct:Y",
        "id": "overlap_triangle_confounding",
        "constructs": [
            {"name": "X"},
            {"name": "Y"},
            {"name": "Z"},
            {"name": "U1"},
            {"name": "U2"},
            {"name": "U3"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "Y", "effect": "Z"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "Y"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Z"},
            {"cause": "U3", "effect": "Y"},
            {"cause": "U3", "effect": "Z"},
        ],
        "observed": ["X", "Y", "Z"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "U1")],
    },
    # Dense pairwise confounding A->B->C->D.
    {
        "outcome": "construct:D",
        "id": "overlap_four_node_complete_confounding",
        "constructs": [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
            {"name": "U_AB"},
            {"name": "U_BC"},
            {"name": "U_CD"},
            {"name": "U_AD"},
        ],
        "edges": [
            {"cause": "A", "effect": "B"},
            {"cause": "B", "effect": "C"},
            {"cause": "C", "effect": "D"},
            {"cause": "U_AB", "effect": "A"},
            {"cause": "U_AB", "effect": "B"},
            {"cause": "U_BC", "effect": "B"},
            {"cause": "U_BC", "effect": "C"},
            {"cause": "U_CD", "effect": "C"},
            {"cause": "U_CD", "effect": "D"},
            {"cause": "U_AD", "effect": "A"},
            {"cause": "U_AD", "effect": "D"},
        ],
        "observed": ["A", "B", "C", "D"],
        "checks": [("not_identifiable", "A")],
    },
    # A -> B -> C -> D with U -> {B, C}; A and C are identifiable.
    {
        "outcome": "construct:D",
        "id": "overlap_selective_some_identifiable",
        "constructs": [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "A", "effect": "B"},
            {"cause": "B", "effect": "C"},
            {"cause": "C", "effect": "D"},
            {"cause": "U", "effect": "B"},
            {"cause": "U", "effect": "C"},
        ],
        "observed": ["A", "B", "C", "D"],
        "checks": [("identifiable", "A"), ("identifiable", "C")],
    },
    # ---- 18. Mediator-Collider Duality ------------------------------------
    {
        "outcome": "construct:Y",
        "id": "mediator_collider_simple",
        "constructs": [
            {"name": "A"},
            {"name": "B"},
            {"name": "M"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "A", "effect": "M"},
            {"cause": "B", "effect": "M"},
            {"cause": "M", "effect": "Y"},
        ],
        "observed": ["A", "B", "M", "Y"],
        "checks": [("identifiable", "A"), ("identifiable", "B")],
    },
    {
        "outcome": "construct:Y",
        "id": "mediator_collider_with_confounding",
        "constructs": [
            {"name": "A"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "A", "effect": "M"},
            {"cause": "U", "effect": "M"},
            {"cause": "U", "effect": "Y"},
            {"cause": "M", "effect": "Y"},
        ],
        "observed": ["A", "M", "Y"],
        "checks": [("identifiable", "A"), ("not_identifiable", "M")],
    },
    # ---- 19. Nested / Hierarchical ----------------------------------------
    # Nested overlapping confounders along X -> M1 -> M2 -> Y.
    {
        "outcome": "construct:Y",
        "id": "nested_front_door_blocked",
        "constructs": [
            {"name": "X"},
            {"name": "M1"},
            {"name": "M2"},
            {"name": "Y"},
            {"name": "U1"},
            {"name": "U2"},
            {"name": "U3"},
        ],
        "edges": [
            {"cause": "X", "effect": "M1"},
            {"cause": "M1", "effect": "M2"},
            {"cause": "M2", "effect": "Y"},
            {"cause": "U1", "effect": "X"},
            {"cause": "U1", "effect": "M1"},
            {"cause": "U2", "effect": "M1"},
            {"cause": "U2", "effect": "M2"},
            {"cause": "U3", "effect": "M2"},
            {"cause": "U3", "effect": "Y"},
        ],
        "observed": ["X", "M1", "M2", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "nested_hierarchical_treatment",
        "constructs": [
            {"name": "X"},
            {"name": "A"},
            {"name": "B"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "X", "effect": "A"},
            {"cause": "X", "effect": "B"},
            {"cause": "A", "effect": "B"},
            {"cause": "A", "effect": "Y"},
            {"cause": "B", "effect": "Y"},
        ],
        "observed": ["X", "A", "B", "Y"],
        "checks": [
            ("identifiable", "X"),
            ("identifiable", "A"),
            ("identifiable", "B"),
        ],
    },
    # ---- 20. Measurement Coverage -----------------------------------------
    # Z->X->M->Y, U->X, U->Y. Coverage variations affect ID strategy.
    {
        "outcome": "construct:Y",
        "id": "coverage_with_iv_observed",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z", "X", "M", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "coverage_without_iv_uses_front_door",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("identifiable", "X")],
    },
    # Same Z->X->M1->M2->Y, U->X, U->Y latent structure with three coverage choices.
    {
        "outcome": "construct:Y",
        "id": "coverage_minimal_xy_hidden_mediators",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "M1"},
            {"name": "M2"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "M1"},
            {"cause": "M1", "effect": "M2"},
            {"cause": "M2", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "coverage_minimal_with_z",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "M1"},
            {"name": "M2"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "M1"},
            {"cause": "M1", "effect": "M2"},
            {"cause": "M2", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["Z", "X", "Y"],
        "checks": [("not_identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "coverage_minimal_with_m1",
        "constructs": [
            {"name": "Z", "role": "endogenous"},
            {"name": "X"},
            {"name": "M1"},
            {"name": "M2"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "M1"},
            {"cause": "M1", "effect": "M2"},
            {"cause": "M2", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M1", "Y"],
        "checks": [("identifiable", "X")],
    },
    # ---- 21. Special IV Structures ----------------------------------------
    # These DAGs alone do not declare the extra assumptions needed by IV designs.
    {
        "outcome": "construct:Y",
        "id": "special_iv_regression_discontinuity_like",
        "constructs": [
            {"name": "R", "role": "endogenous"},
            {"name": "D"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "R", "effect": "D"},
            {"cause": "D", "effect": "Y"},
            {"cause": "U", "effect": "D"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["R", "D", "Y"],
        "checks": [("not_identifiable", "D"), ("identifiable", "R")],
    },
    {
        "outcome": "construct:Y",
        "id": "special_iv_mendelian_randomization",
        "constructs": [
            {"name": "G", "role": "endogenous"},
            {"name": "X"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "G", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["G", "X", "Y"],
        "checks": [("not_identifiable", "X"), ("identifiable", "G")],
    },
    # ---- 22. Temporal Unrolling Edge Cases --------------------------------
    {
        "outcome": "construct:Y",
        "id": "tedge_only_lagged_no_contemporaneous",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "tedge_time_invariant_only",
        "constructs": [
            {"name": "X", "temporal_status": "time_invariant"},
            {"name": "Y", "temporal_status": "time_invariant"},
        ],
        "edges": [{"cause": "X", "effect": "Y"}],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
    {
        "outcome": "construct:Y",
        "id": "tedge_mixed_with_iv_via_trait",
        "constructs": [
            {
                "id": "construct:ca7081092c417bf11cf3",
                "name": "Trait",
                "temporal_status": "time_invariant",
                "role": "endogenous",
            },
            {"name": "State", "temporal_status": "time_invariant"},
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "Trait", "effect": "X"},
            {"cause": "State", "effect": "X"},
            {"cause": "State", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["Trait", "X", "Y"],
        "checks": [
            ("not_identifiable", "X"),
            ("identifiable", "Trait"),
        ],
    },
    {
        "outcome": "construct:Y",
        "id": "tedge_mixed_no_instrument",
        "constructs": [
            {"name": "State", "temporal_status": "time_invariant"},
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "State", "effect": "X"},
            {"cause": "State", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("not_identifiable", "X"), ("blocked_by", "X", "State")],
    },
    {
        "outcome": "construct:Y",
        "id": "tedge_lagged_confounding_with_lagged_treatment",
        "constructs": [
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [("identifiable", "X")],
    },
]


@pytest.mark.parametrize("case", IDENTIFICATION_CASES, ids=lambda c: c["id"])
def test_identification(case):
    """Run check_identifiability against a graph and assert the listed checks."""
    constructs, edges = make_graph(case["constructs"], case["edges"])
    outcome = case.get("outcome")
    result = check_identifiability(
        constructs,
        edges,
        outcome_id=fixture_entity_id("construct", outcome.removeprefix("construct:"))
        if outcome is not None
        else None,
        observed_constructs=set(case["observed"]),
    )
    _run_checks(result, case["checks"])


# =============================================================================
# MARGINALIZATION ANALYSIS CASES
#
# Each case derives the canonical model's execution selection and
# asserts which unobserved constructs can be marginalized vs which need modeling.
# Check kinds:
#   ("can_marginalize", name)
#   ("needs_modeling", name, [treatment_names])  -- in blocking_details with these
#   ("not_blocking", name)
#   ("not_marginalizable", name)
# =============================================================================

MARGINALIZATION_CASES: list[dict[str, Any]] = [
    # U has only one observed child (X) — can be marginalized.
    {
        "outcome": "construct:Y",
        "id": "marg_single_child_confounder",
        "constructs": [
            {"name": "U"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U", "effect": "X"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [
            ("can_marginalize", "U"),
            ("not_blocking", "U"),
        ],
    },
    # Bow graph U: blocks X, needs modeling.
    {
        "outcome": "construct:Y",
        "id": "marg_needs_modeling_blocking_confounder",
        "constructs": [
            {"name": "U"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [
            ("needs_modeling", "U", ["X"]),
            ("not_marginalizable", "U"),
        ],
    },
    # Front-door handles U; U can be marginalized.
    {
        "outcome": "construct:Y",
        "id": "marg_front_door_handled",
        "constructs": [
            {"name": "X"},
            {"name": "M"},
            {"name": "Y"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
        "observed": ["X", "M", "Y"],
        "checks": [("can_marginalize", "U")],
    },
    # Mixed: U1 marginalize, U2 model.
    {
        "outcome": "construct:Y",
        "id": "marg_mixed",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "X"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [
            ("can_marginalize", "U1"),
            ("needs_modeling", "U2", None),
        ],
    },
    # Chain U1 -> U2 -> {X, Y}: U1 marginalize, U2 model.
    {
        "outcome": "construct:Y",
        "id": "marg_chain_of_unobserved",
        "constructs": [
            {"name": "U1"},
            {"name": "U2"},
            {"name": "X"},
            {"name": "Y"},
        ],
        "edges": [
            {"cause": "U1", "effect": "U2"},
            {"cause": "U2", "effect": "X"},
            {"cause": "U2", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
        "observed": ["X", "Y"],
        "checks": [
            ("can_marginalize", "U1"),
            ("needs_modeling", "U2", None),
        ],
    },
]


@pytest.mark.parametrize("case", MARGINALIZATION_CASES, ids=lambda c: c["id"])
def test_marginalization(case):
    """Check projection and unsupported dependencies on the scientific model."""
    dynamical_model_spec = make_model(
        [construct["name"] for construct in case["constructs"]],
        [(edge["cause"], edge["effect"]) for edge in case["edges"]],
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(
                    role="endogenous",
                    temporal_status="time_invariant",
                    indicators=construct.indicators if construct.name in case["observed"] else (),
                )
                for construct in dynamical_model_spec.constructs
            ),
        ),
    )
    selection = StructuralSelection(
        dynamical_model_spec,
        fixture_entity_id("construct", case["outcome"].removeprefix("construct:")),
    )
    report = selection.identification
    marginalized = selection.marginalized_construct_ids
    unsupported = unsupported_construct_ids(selection)

    for check in case["checks"]:
        kind = check[0]
        identity = fixture_entity_id("construct", check[1])
        if kind == "can_marginalize":
            assert identity in marginalized
        elif kind == "needs_modeling":
            (_, _name, expected_targets) = check
            assert identity in unsupported
            if expected_targets is not None:
                assert (
                    sorted(
                        dynamical_model_spec.get_construct(treatment).name
                        for treatment, finding in report.non_identifiable.items()
                        if identity in finding.confounders
                        and dynamical_model_spec.get_construct(treatment).indicators
                    )
                    == expected_targets
                )
        elif kind == "not_blocking":
            assert identity not in unsupported
        elif kind == "not_marginalizable":
            assert identity not in marginalized
        else:
            raise AssertionError(f"Unknown marginalization check kind: {kind}")


# =============================================================================
# UNROLLING VERIFICATION (different API: unroll_temporal_dag)
# =============================================================================


def _ar1_obs_xy() -> tuple[tuple[tuple[ConstructSpec, ...], tuple[CausalEdgeSpec, ...]], set[str]]:
    latent = make_graph(
        constructs=[
            {"name": "X", "role": "endogenous", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[{"cause": "X", "effect": "Y"}],
    )
    return latent, {"X", "Y"}


def test_unroll_creates_two_timesteps():
    latent, observed = _ar1_obs_xy()
    dag = unroll_temporal_dag(*latent, observed)
    nodes = set(dag.nodes())
    for n in ["X_t", "X_{t-1}", "Y_t", "Y_{t-1}"]:
        assert n in nodes


def test_unroll_ar1_edges():
    latent, observed = _ar1_obs_xy()
    dag = unroll_temporal_dag(*latent, observed)
    edges = list(dag.edges())
    assert ("X_{t-1}", "X_t") in edges
    assert ("Y_{t-1}", "Y_t") in edges


def test_unroll_carryover_includes_hidden_states():
    """Hidden states carry forward and stay hidden after latent projection."""
    latent = make_graph(
        constructs=[
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    )
    dag = unroll_temporal_dag(*latent, {"X", "Y"})
    edges = list(dag.edges())
    assert ("X_{t-1}", "X_t") in edges
    assert ("Y_{t-1}", "Y_t") in edges
    assert ("U_{t-1}", "U_t") in edges
    admg, _ = dag_to_admg(*latent, {"X", "Y"})
    assert {str(node) for node in admg.nodes()} == {"X_t", "X_{t-1}", "Y_t", "Y_{t-1}"}
    assert "U_t" in dag.nodes()
    assert "U_{t-1}" in dag.nodes()


def test_unroll_dynamic_causes_use_previous_slice():
    latent, observed = _ar1_obs_xy()
    dag = unroll_temporal_dag(*latent, observed)
    edges = list(dag.edges())
    assert ("X_t", "Y_t") not in edges
    assert ("X_{t-1}", "Y_{t-1}") not in edges
    assert ("X_{t-1}", "Y_t") in edges


def test_unroll_lagged_edges():
    latent = make_graph(
        constructs=[
            {"name": "X", "role": "endogenous", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[{"cause": "X", "effect": "Y"}],
    )
    dag = unroll_temporal_dag(*latent, {"X", "Y"})
    assert ("X_{t-1}", "Y_t") in list(dag.edges())


def test_unroll_time_invariant_single_node():
    latent = make_graph(
        constructs=[
            {"name": "Trait", "temporal_status": "time_invariant"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[{"cause": "Trait", "effect": "Y"}],
    )
    dag = unroll_temporal_dag(*latent, {"Trait", "Y"})
    nodes = set(dag.nodes())
    assert "Trait" in nodes
    assert "Trait_t" not in nodes
    assert "Trait_{t-1}" not in nodes


def test_unroll_time_invariant_affects_both_timesteps():
    latent = make_graph(
        constructs=[
            {"name": "Trait", "temporal_status": "time_invariant"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[{"cause": "Trait", "effect": "Y"}],
    )
    dag = unroll_temporal_dag(*latent, {"Trait", "Y"})
    edges = list(dag.edges())
    assert ("Trait", "Y_t") in edges
    assert ("Trait", "Y_{t-1}") in edges


def test_unroll_hidden_labels_correct():
    latent = make_graph(
        constructs=[
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
        ],
        edges=[
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    )
    dag = unroll_temporal_dag(*latent, {"X", "Y"})
    assert dag.nodes["X_t"].get("hidden", False) is False
    assert dag.nodes["Y_t"].get("hidden", False) is False
    assert dag.nodes["U_t"].get("hidden", False) is True
    assert dag.nodes["U_{t-1}"].get("hidden", False) is True


# =============================================================================
# ADMG PROJECTION (different API: dag_to_admg)
# =============================================================================


def test_admg_bidirected_from_contemporaneous_confounder():
    latent = make_graph(
        constructs=[
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        edges=[
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    )
    admg, confounders = dag_to_admg(*latent, {"X", "Y"})
    assert "U" in confounders
    undirected = {tuple(sorted((str(e[0]), str(e[1])))) for e in admg.undirected.edges()}
    assert ("X_t", "Y_t") in undirected or ("X_{t-1}", "Y_{t-1}") in undirected


def test_admg_bidirected_from_lagged_confounder():
    latent = make_graph(
        constructs=[
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        edges=[
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    )
    admg, confounders = dag_to_admg(*latent, {"X", "Y"})
    assert "U" in confounders
    assert len(list(admg.undirected.edges())) > 0


def test_admg_no_bidirected_single_child():
    """Unobserved with a single observed child should not become a confounder."""
    latent = make_graph(
        constructs=[
            {"name": "X", "temporal_status": "time_varying"},
            {"name": "Y", "temporal_status": "time_varying"},
            {"name": "U", "temporal_status": "time_varying"},
        ],
        edges=[
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
        ],
    )
    _admg, confounders = dag_to_admg(*latent, {"X", "Y"})
    assert "U" not in confounders


# =============================================================================
# FIND BLOCKING CONFOUNDERS (different API: find_blocking_confounders)
# =============================================================================


def test_find_blocking_confounders_via_latent_chain():
    """U has only one direct observed child (X) but creates X<-U->V->Y backdoor."""
    latent = make_graph(
        constructs=[
            {"name": "U"},
            {"name": "V"},
            {"name": "X"},
            {"name": "Y"},
        ],
        edges=[
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "V"},
            {"cause": "V", "effect": "Y"},
            {"cause": "X", "effect": "Y"},
        ],
    )
    blockers = find_blocking_confounders(*latent, {"X", "Y"}, "X", "Y")
    assert "U" in blockers
    assert "V" not in blockers


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
