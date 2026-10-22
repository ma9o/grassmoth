"""Tests for causal design schema computed properties and utility functions.

Object-level construction validation (ConstructSpec, DynamicalModelSpec, IndicatorSpec,
MeasurementStructure) is covered by test_schema_validators.py via dict validators.
This file tests CausalDesign composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.identity import (
    ConstructId,
    DistributionId,
    EdgeId,
    IndicatorId,
    MechanismId,
    ParameterElementId,
    ParameterId,
)
from grassmoth.artifacts.model.construct import (
    CausalEdgeSpec,
    Role,
    TemporalStatus,
    replace_constructs,
)
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.duration import Duration
from grassmoth.artifacts.observations.preparation import (
    ComputedExtractionSpec,
    DataVariableSpec,
    SemanticExtractionSpec,
    WindowExpression,
    check_semantic_collisions,
)
from grassmoth.artifacts.observations.semantics import (
    AnchorPolicy,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
)
from grassmoth.artifacts.observations.spec import AuthoredObservationSpec
from tests.support.helpers import graph_constructs, make_model

pytestmark = pytest.mark.contract

_QUERY = {
    "start": "2026-05-15",
    "horizon": "9w",
    "interventions": [{"target": "construct:x", "value": 10}],
}


class TestModel:
    """Scientific membership follows the endpoints of one connected graph."""

    def test_valid_simple_structure(self, construct_factory):
        edge = CausalEdgeSpec(
            id="edge:stress-mood",
            cause=construct_factory("stress", Role.EXOGENOUS),
            effect=construct_factory("mood"),
            description="Stress affects mood",
        )
        dynamical_model_spec = DynamicalModelSpec.from_entities(edges=(edge,))
        assert [construct.name for construct in dynamical_model_spec.constructs] == [
            "stress",
            "mood",
        ]
        assert dynamical_model_spec.edges[0].cause is dynamical_model_spec.get_construct(
            edge.cause.id
        )

    @pytest.mark.parametrize("endpoint", ["cause", "effect"])
    def test_undefined_endpoint_reference_is_rejected(self, endpoint):
        payload = make_model(["stress", "mood"], [("stress", "mood")]).model_dump(mode="json")
        next(iter(payload["edges"].values()))[endpoint] = "construct:unknown"
        with pytest.raises(ValueError, match="Undefined construct endpoint"):
            DynamicalModelSpec.model_validate(payload).materialized()

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            DynamicalModelSpec.from_entities(
                edges=(
                    CausalEdgeSpec(
                        id="edge:mood-weather",
                        cause=construct_factory("mood"),
                        effect=construct_factory("weather", Role.EXOGENOUS),
                        description="Invalid edge",
                    ),
                )
            )

    def test_invalid_time_varying_to_time_invariant_edge(self, construct_factory):
        with pytest.raises(ValueError, match="cannot be a cause of time-invariant construct"):
            DynamicalModelSpec.from_entities(
                edges=(
                    CausalEdgeSpec(
                        id="edge:habit-trait",
                        cause=construct_factory("habit"),
                        effect=construct_factory(
                            "trait", temporal_status=TemporalStatus.TIME_INVARIANT
                        ),
                        description="Habit changes a fixed trait",
                    ),
                )
            )

    @pytest.mark.parametrize(
        ("question", "message"),
        [
            ({"text": "   ", "outcome": "construct:y"}, "at least 1 character"),
            ({"text": "Why?"}, "outcome"),
            ({"text": "Why?", "outcome": None}, "outcome"),
            ({"text": "Why?", "queries": {"q": _QUERY}}, "outcome"),
            (
                {
                    "text": "Why?",
                    "outcome": "construct:y",
                    "queries": {"q": {**_QUERY, "interventions": []}},
                },
                "needs an intervention",
            ),
            (
                {"text": "Why?", "outcome": "construct:x", "queries": {"q": _QUERY}},
                "intervenes on the outcome",
            ),
            ({"text": "Why?", "outcome": "construct:y", "queries": {" ": _QUERY}}, "at least 1"),
        ],
    )
    def test_question_names_an_outcome_and_contrasts_interventions(self, question, message):
        with pytest.raises(ValidationError, match=message):
            QuestionSpec.model_validate(question)
        assert QuestionSpec(text="  Why?  ", outcome=ConstructId("construct:y")).text == "Why?"


class TestDataVariable:
    """Tests for IndicatorSpec validation."""

    def test_invalid_aggregation(self):
        """Invalid aggregation is rejected."""
        with pytest.raises(ValueError, match="aggregation"):
            DataVariableSpec.model_validate(
                {
                    "observation": {
                        "id": "indicator:e05e217de7f4442abdc5",
                        "name": "mood_rating",
                        "measurement_dtype": "continuous",
                        "aggregation": "invalid_agg",
                    },
                    "extraction": {"kind": "semantic", "how_to_measure": "Extract mood"},
                }
            )

    def test_invalid_measurement_dtype(self):
        """Invalid measurement_dtype is rejected."""
        with pytest.raises(ValueError, match="measurement_dtype"):
            DataVariableSpec.model_validate(
                {
                    "observation": {
                        "id": "indicator:e05e217de7f4442abdc5",
                        "name": "mood_rating",
                        "measurement_dtype": "invalid_type",
                        "aggregation": "mean",
                    },
                    "extraction": {"kind": "semantic", "how_to_measure": "Extract mood"},
                }
            )

    def test_ordinal_requires_levels(self):
        """Ordinal dtype without ordinal_levels is rejected."""
        with pytest.raises(ValueError, match="ordinal_levels is required"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:036fd134b9ad32d7a2ca",
                    name="pain",
                    measurement_dtype="ordinal",
                    aggregation="last",
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract pain level"),
            )

    def test_ordinal_needs_at_least_two_levels(self):
        """Ordinal with only one level is rejected."""
        with pytest.raises(ValueError, match="at least 2 items"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:036fd134b9ad32d7a2ca",
                    name="pain",
                    measurement_dtype="ordinal",
                    aggregation="last",
                    ordinal_levels=["only_one"],
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract pain level"),
            )

    def test_ordinal_no_duplicate_levels(self):
        """Ordinal with duplicate levels is rejected."""
        with pytest.raises(ValueError, match="duplicate labels"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:036fd134b9ad32d7a2ca",
                    name="pain",
                    measurement_dtype="ordinal",
                    aggregation="last",
                    ordinal_levels=["low", "low", "high"],
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract pain level"),
            )

    def test_ordinal_valid_levels(self):
        """Ordinal with valid levels passes."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:036fd134b9ad32d7a2ca",
                name="pain",
                measurement_dtype="ordinal",
                aggregation="last",
                ordinal_levels=["low", "medium", "high"],
            ),
            extraction=SemanticExtractionSpec(how_to_measure="Extract pain level"),
        )
        assert ind.observation.ordinal_levels == ("low", "medium", "high")

    def test_categorical_requires_levels(self):
        with pytest.raises(ValueError, match="categorical_levels is required"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:73fed6c52a41057ef02f",
                    name="location",
                    measurement_dtype="categorical",
                    aggregation="last",
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract location"),
            )

    def test_categorical_needs_at_least_two_levels(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:73fed6c52a41057ef02f",
                    name="location",
                    measurement_dtype="categorical",
                    aggregation="last",
                    categorical_levels=["home"],
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract location"),
            )

    def test_categorical_rejects_duplicate_levels(self):
        with pytest.raises(ValueError, match="duplicate labels"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:73fed6c52a41057ef02f",
                    name="location",
                    measurement_dtype="categorical",
                    aggregation="last",
                    categorical_levels=["home", "home"],
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Extract location"),
            )

    def test_non_ordinal_ignores_levels(self):
        """Non-ordinal dtype doesn't require ordinal_levels."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:c5b118ae552981435d7b",
                name="weight",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            extraction=SemanticExtractionSpec(how_to_measure="Extract weight"),
        )
        assert ind.observation.ordinal_levels is None

    def test_semantic_default(self):
        """Extraction mode defaults to 'semantic'."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:e05e217de7f4442abdc5",
                name="mood_rating",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            extraction=SemanticExtractionSpec(how_to_measure="Extract mood"),
        )
        assert ind.extraction.kind == "semantic"

    def test_invalid_extraction_mode(self):
        """Invalid extraction_mode is rejected."""
        with pytest.raises(ValueError, match="extraction"):
            DataVariableSpec.model_validate(
                {
                    "observation": {
                        "id": "indicator:e05e217de7f4442abdc5",
                        "name": "mood_rating",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    "extraction": {"kind": "invalid", "how_to_measure": "Extract mood"},
                }
            )

    def test_computed_valid(self):
        """Computed indicator with single source column and continuous dtype passes."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:aa573b5cc0c0a1837e05",
                name="avg_heart_rate",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Use heart_rate column directly", source_columns=["heart_rate"]
            ),
        )
        assert ind.extraction.kind == "computed"

    def test_computed_count_dtype(self):
        """Computed indicator with count dtype passes."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:a0ce08437c19d06aafd1",
                name="total_steps",
                measurement_dtype="count",
                aggregation="sum",
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Use steps column directly", source_columns=["steps"]
            ),
        )
        assert ind.extraction.kind == "computed"

    def test_computed_binary_point_dtype(self):
        """Computed indicator with binary dtype passes for direct point aggregation."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:58ba7e8b022133d4764f",
                name="alarm_state",
                measurement_dtype="binary",
                aggregation="last",
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Use the last observed alarm_state value directly",
                source_columns=["alarm_state"],
            ),
        )
        assert ind.extraction.kind == "computed"

    def test_computed_ordinal_point_dtype(self):
        """Computed indicator with ordinal dtype passes for direct point aggregation."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:745132edf4775f59f221",
                name="mood_label",
                measurement_dtype="ordinal",
                aggregation="last",
                ordinal_levels=["bad", "ok", "good"],
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Use the last observed mood_label value directly",
                source_columns=["mood_label"],
            ),
        )
        assert ind.extraction.kind == "computed"

    def test_computed_categorical_point_dtype(self):
        """Computed indicator with categorical dtype passes for direct point aggregation."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:42f1f2a7a4c92ee606b6",
                name="care_setting",
                measurement_dtype="categorical",
                aggregation="first",
                categorical_levels=["home", "clinic"],
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Use the first observed care_setting value directly",
                source_columns=["care_setting"],
            ),
        )
        assert ind.extraction.kind == "computed"

    def test_computed_requires_single_source_column(self):
        """Direct computed indicators with 0 or 2+ source_columns are rejected."""
        with pytest.raises(ValueError, match="source_column"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:c21b43949b3712e734c8",
                    name="avg_hr",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Use heart_rate", source_columns=[]
                ),
            )
        with pytest.raises(ValueError, match="source_column"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:c21b43949b3712e734c8",
                    name="avg_hr",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Compute from systolic and diastolic",
                    source_columns=["systolic_bp", "diastolic_bp"],
                ),
            )

    def test_computed_rule_allows_multi_source_deterministic_formula(self):
        """Computed rules can reference multiple source columns deterministically."""
        ind = DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:e33fbf156ca312595e47",
                name="mean_arterial_pressure",
                measurement_dtype="continuous",
                aggregation="mean",
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Compute deterministically from systolic and diastolic blood pressure",
                source_columns=["systolic_bp", "diastolic_bp"],
                computed_rule="mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)",
            ),
        )
        assert ind.extraction.kind == "computed"
        assert ind.extraction.computed_rule is not None
        assert ind.extraction.computed_rule.dependencies == frozenset(
            {"systolic_bp", "diastolic_bp"}
        )
        assert ind.extraction.computed_rule.summary_operator == SummaryOperator.MEAN
        assert (
            TypeAdapter(WindowExpression).validate_python(ind.extraction.computed_rule)
            is ind.extraction.computed_rule
        )
        assert (
            ind.model_dump(mode="json")["extraction"]["computed_rule"]
            == ind.extraction.computed_rule.source
        )

    def test_computed_rule_rejects_semantic_mode(self):
        """Semantic extraction cannot carry a deterministic computation."""
        with pytest.raises(ValueError, match="computed_rule"):
            SemanticExtractionSpec.model_validate(
                {
                    "how_to_measure": "Read SpO2",
                    "source_columns": ["spo2_pct"],
                    "computed_rule": "last(spo2_pct)",
                }
            )

    def test_computed_rule_rejects_undeclared_source_column(self):
        """computed_rule must reference only declared source_columns."""
        with pytest.raises(ValueError, match="references undeclared source_columns"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:0c111e9b74f243fbc086",
                    name="glucose_out_of_range",
                    measurement_dtype="count",
                    aggregation="sum",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Count out-of-range glucose values deterministically",
                    source_columns=["glucose_mg_dl"],
                    computed_rule="None if count_non_null(glucose_mg_dl) == 0 else sum(1 if (glucose_mg_dl < 70 or serum_glucose > 180) else 0)",
                ),
            )

    def test_computed_rule_requires_source_reference(self):
        """computed_rule must actually use at least one declared source column."""
        with pytest.raises(ValueError, match="must reference at least 1 source_column"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:4aa7a5f09fd3489f4f2e",
                    name="constant_flag",
                    measurement_dtype="binary",
                    aggregation="last",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Always emit a constant flag",
                    source_columns=["spo2_pct"],
                    computed_rule="last(1)",
                ),
            )

    def test_computed_still_rejects_invalid_semantics(self):
        """Computed indicators still respect the measurement-semantics grid."""
        with pytest.raises(
            ValueError, match="aggregation 'mean' requires measurement_dtype='continuous'"
        ):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:58ba7e8b022133d4764f",
                    name="alarm_state",
                    measurement_dtype="binary",
                    aggregation="mean",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Use alarm_state directly", source_columns=["alarm_state"]
                ),
            )


class TestModelContainment:
    def test_construct_owns_its_indicators(self):
        dynamical_model_spec = make_model(["mood", "stress"], [("mood", "stress")])
        mood, stress = dynamical_model_spec.constructs
        assert dynamical_model_spec.get_construct(mood.id).indicators == mood.indicators
        assert dynamical_model_spec.indicator_owner(mood.indicators[0].observation.id) is mood
        assert dynamical_model_spec.indicator_owner(stress.indicators[0].observation.id) is stress
        assert "construct_id" not in mood.indicators[0].model_dump()

    def test_indicator_cannot_have_an_independent_unknown_owner(self):
        dynamical_model_spec = make_model(["mood"])
        value = dynamical_model_spec.model_dump(mode="json")
        next(iter(graph_constructs(value)[0]["indicators"].values()))["construct_id"] = (
            "construct:unknown"
        )
        with pytest.raises(ValidationError, match="Extra inputs"):
            DynamicalModelSpec.model_validate(value).materialized()

    def test_latent_construct_without_indicators_is_valid(self):
        dynamical_model_spec = make_model(["observed", "latent"], [("observed", "latent")])
        observed, latent = dynamical_model_spec.constructs
        result = dynamical_model_spec.with_entities(
            edges=replace_constructs(
                dynamical_model_spec.edges,
                (observed, latent.revised(indicators=())),
            )
        )
        assert result.get_construct(latent.id).indicators == ()

    def test_dynamic_feedback_is_valid_but_static_cycles_are_rejected(self):
        dynamical_model_spec = make_model(["X", "Y"], [("X", "Y"), ("Y", "X")])
        payload = dynamical_model_spec.model_dump(mode="json")
        for construct in graph_constructs(payload):
            construct["temporal_status"] = "time_invariant"
        with pytest.raises(ValidationError, match="Time-invariant edges form cycle"):
            DynamicalModelSpec.model_validate(payload).materialized()


class TestDuration:
    """Tests for fixed duration parsing and lowering."""

    def test_seconds(self):
        assert Duration("3600s").seconds / 3600 == 1.0
        duration = TypeAdapter(Duration).validate_python("03600s")
        assert duration.seconds == 3600
        assert TypeAdapter(Duration).validate_python(duration) is duration
        assert TypeAdapter(Duration).dump_json(duration) == b'"03600s"'

    def test_minutes(self):
        assert Duration("60m").seconds / 3600 == 1.0

    def test_hours(self):
        assert Duration("4h").seconds / 3600 == 4.0
        assert Duration("1h").seconds / 3600 == 1.0

    def test_days(self):
        assert Duration("1d").seconds / 3600 == 24.0
        assert Duration("7d").seconds / 3600 == 168.0

    def test_weeks(self):
        assert Duration("1w").seconds / 3600 == 168.0
        assert Duration("2w").seconds / 3600 == 336.0

    def test_months_are_rejected(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("1mo")

    def test_quarters_are_rejected(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("1q")

    def test_years_are_rejected(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("1y")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("abc")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("5x")

    def test_zero_duration(self):
        with pytest.raises(ValueError, match="positive"):
            Duration("0d")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Duration("d")

    def test_invalid_model_clock(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            make_model(["X"]).revised(measurement_clock="bad")


class TestDeriveObservationSemantics:
    """Tests for derive_indicator_observation_semantics."""

    def test_first_maps_to_point_at_window_start(self):
        semantics = derive_indicator_observation_semantics(SummaryOperator.FIRST, "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.FIRST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_START

    def test_last_maps_to_point_at_window_end(self):
        semantics = derive_indicator_observation_semantics(SummaryOperator.LAST, "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.LAST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_interval_summary_operator_maps_to_interval_support(self):
        semantics = derive_indicator_observation_semantics(SummaryOperator.SUM, "count")
        assert semantics.support_kind == SupportKind.INTERVAL
        assert semantics.summary_operator == SummaryOperator.SUM
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_std_requires_continuous_measurements(self):
        with pytest.raises(
            ValueError, match="aggregation 'std' requires measurement_dtype='continuous'"
        ):
            derive_indicator_observation_semantics(SummaryOperator.STD, "count")

    def test_ordinal_indicators_only_support_point_operators(self):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            derive_indicator_observation_semantics(SummaryOperator.MEAN, "ordinal")

    def test_unsupported_aggregations_fail_fast(self):
        with pytest.raises(ValueError, match="aggregation"):
            DataVariableSpec.model_validate(
                {
                    "observation": {
                        "id": "indicator:mood",
                        "name": "mood",
                        "measurement_dtype": "continuous",
                        "aggregation": "median",
                    },
                    "extraction": {"kind": "semantic", "how_to_measure": "Median score"},
                }
            )


class TestSemanticCollisions:
    """Tests for check_semantic_collisions function."""

    def test_count_text_mean_agg_collision(self):
        """'count' in how_to_measure + mean aggregation → warning."""
        warnings = check_semantic_collisions(
            "Count the number of exercise sessions", SummaryOperator.MEAN
        )
        assert len(warnings) >= 1
        assert "counting" in warnings[0].lower() or "count" in warnings[0].lower()

    def test_no_collision(self):
        """Consistent text and aggregation → no warnings."""
        warnings = check_semantic_collisions("Average daily mood rating", SummaryOperator.MEAN)
        assert len(warnings) == 0

    def test_total_text_mean_agg_collision(self):
        """'total' in text + mean aggregation → warning."""
        warnings = check_semantic_collisions(
            "Total steps walked during the day", SummaryOperator.MEAN
        )
        assert len(warnings) >= 1

    def test_last_text_sum_agg_collision(self):
        """'most recent' in text + sum aggregation → warning."""
        warnings = check_semantic_collisions(
            "The most recent blood pressure reading", SummaryOperator.SUM
        )
        assert len(warnings) >= 1


class TestIndicatorObservationSemantics:
    """Tests for IndicatorSpec computed observation semantics."""

    def test_interval_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("steps", aggregation="sum", dtype="count")
        assert ind.observation.support_kind == SupportKind.INTERVAL
        assert ind.observation.summary_operator == SummaryOperator.SUM
        assert ind.observation.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.observation.requires_interval_summary_measurement is True

    def test_point_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("last_bp", aggregation="last", dtype="continuous")
        assert ind.observation.support_kind == SupportKind.POINT
        assert ind.observation.summary_operator == SummaryOperator.LAST
        assert ind.observation.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.observation.requires_interval_summary_measurement is False

    def test_ordinal_indicator_uses_point_semantics(self, indicator_factory):
        ind = indicator_factory("pain_level", aggregation="last", dtype="ordinal")
        assert ind.observation.support_kind == SupportKind.POINT
        assert ind.observation.summary_operator == SummaryOperator.LAST
        assert ind.observation.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_unsupported_aggregation_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(ValueError, match="aggregation"):
            indicator_factory("median_hr", aggregation="median", dtype="continuous")

    def test_ordinal_interval_summary_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            indicator_factory("pain_level", aggregation="mean", dtype="ordinal")


class TestIndicatorObservationWindow:
    def test_valid_observation_window(self):
        indicator = DataVariableSpec(
            observation=AuthoredObservationSpec(
                id="indicator:b41c85c254676b4bc588",
                name="fortnightly_mood",
                measurement_dtype="continuous",
                aggregation="mean",
                observation_window="2w",
            ),
            extraction=SemanticExtractionSpec(how_to_measure="Average mood over two weeks"),
        )

        assert indicator.observation.observation_window is not None
        assert indicator.observation.observation_window.source == "2w"

    def test_invalid_observation_window(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    id="indicator:b41c85c254676b4bc588",
                    name="fortnightly_mood",
                    measurement_dtype="continuous",
                    aggregation="mean",
                    observation_window="monthly",
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Average mood over two weeks"),
            )


@pytest.mark.parametrize(
    "expression",
    ["mean(values", "values.mean()", "unknown_helper(values)", {"window_expr": "mean(values)"}],
)
def test_window_expression_is_validated_at_the_scalar_boundary(expression):
    with pytest.raises(ValidationError):
        TypeAdapter(WindowExpression).validate_python(expression)


def test_window_expression_serializes_without_a_wrapper():
    adapter = TypeAdapter(WindowExpression)
    expression = adapter.validate_python("mean(values)")
    assert adapter.dump_json(expression) == b'"mean(values)"'


@pytest.mark.parametrize(
    ("expression", "aggregation"),
    [
        ("sum(reading)", "mean"),
        ("mean(reading)", "last"),
        ("last(reading) / 10", "first"),
        ("None if count_non_null(reading) == 0 else sum(reading)", "mean"),
        ("1 if any(reading < 92) else 0", "last"),
        ("mean(sum(reading))", "mean"),
        ("sum(reading) / count_non_null(reading)", "sum"),
    ],
)
def test_computed_measurement_cannot_claim_different_summary_semantics(expression, aggregation):
    with pytest.raises(ValidationError, match="computed_rule"):
        DataVariableSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id="indicator:reading",
                name="reading",
                measurement_dtype="continuous",
                aggregation=aggregation,
            ),
            extraction=ComputedExtractionSpec(
                how_to_measure="Read the signal",
                source_columns=("reading",),
                computed_rule=expression,
            ),
        )


@pytest.mark.parametrize(
    ("id_type", "prefix"),
    [
        (ConstructId, "construct"),
        (EdgeId, "edge"),
        (IndicatorId, "indicator"),
        (MechanismId, "mechanism"),
        (DistributionId, "distribution"),
        (ParameterId, "parameter"),
        (ParameterElementId, "element"),
    ],
)
def test_nominal_ids_validate_and_serialize_as_strings(id_type, prefix):
    adapter = TypeAdapter(id_type)
    identity = f"{prefix}:{'a' * 64}"
    restored = adapter.validate_json(f'"{identity}"')
    assert restored == identity
    assert adapter.dump_json(restored) == f'"{identity}"'.encode()
    with pytest.raises(ValidationError, match="String should match pattern"):
        adapter.validate_python("unrelated:" + "a" * 64)
