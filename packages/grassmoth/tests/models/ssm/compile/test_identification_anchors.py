"""Guard tests for the per-construct identification-anchor invariant.

Every retained construct must have exactly one location anchor and one scale
anchor (docs/assumptions.md#parameter-anchors). These tests
enumerate the family/policy combinations so that any future eligibility change
that reopens an exact likelihood ridge fails at construction.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.identity import DistributionId, IndicatorId, ParameterId
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import (
    CallExpression,
    Expression,
    coefficient,
    restoring_potential,
    state,
)
from grassmoth.artifacts.model.indicator import IndicatorPolarity, IndicatorSpec
from grassmoth.artifacts.model.likelihood import (
    BernoulliLogitsLawSpec,
    CategoricalLawSpec,
    DeltaLawSpec,
    LikelihoodSpec,
    LinkFunction,
    NormalLawSpec,
    OrderedLogisticLawSpec,
)
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from grassmoth.artifacts.observations.semantics import SummaryOperator
from grassmoth.artifacts.observations.spec import AuthoredObservationSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.compilation_errors import AggregatedCompileError
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.observations import materialize_observation_laws
from grassmoth.models.structure import StructuralSelection, selected_state_ids
from grassmoth_test_support.helpers import fixture_entity_id, make_model
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import (
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    parameter_named,
    without_parameters,
)


def _categorical_loading_pinned_in_mixed_construct__with_likelihoods() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testcategoricalanchors_test_categorical_loading_pinned_in_mixed_construct__with_likelihoods.json"
    )


def _manifest_intercept_remains_free_for_raw_gaussian_sum_channel__with_likelihoods() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "identification_anchors/testlocationanchors_test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel__with_likelihoods.json"
    )


def _static_t0_mean_free_with_standardized_channel__with_likelihoods() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testlocationanchors_test_static_t0_mean_free_with_standardized_channel__with_likelihoods.json"
    )


def _all_categorical_construct_gets_anchor_slope__with_likelihoods() -> DynamicalModelSpec:
    dynamical_model_spec = _categorical_loading_pinned_in_mixed_construct__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    mood_kind = indicator_named(dynamical_model_spec, "mood_kind")
    obs_sd_mood_rating = parameter_named(dynamical_model_spec, "obs_sd_mood_rating")
    mood_revised = mood.revised(indicators=(mood_kind,))
    parameters, distributions = without_parameters(dynamical_model_spec, obs_sd_mood_rating)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_count() -> (
    DynamicalModelSpec
):
    _MOOD_RATING_INDICATOR_ID = IndicatorId("indicator:e05e217de7f4442abdc5")
    _CINT_MOOD_PARAMETER_ID = ParameterId(
        "parameter:403d1ba7a1dc2ea4b423d3918766b7d70dc5cab41ad22f6f0251ee151e43952d"
    )
    dynamical_model_spec = _all_categorical_construct_gets_anchor_slope__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    (mood_potential,) = mood.dynamics
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    obs_cat_intercepts = parameter_named(dynamical_model_spec, "obs_cat_intercepts")
    obs_cat_slopes = parameter_named(dynamical_model_spec, "obs_cat_slopes")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_revised = mood.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_MOOD_RATING_INDICATOR_ID,
                    name="mood_rating",
                    measurement_dtype="count",
                    aggregation=SummaryOperator.COUNT,
                    observation_window=None,
                ),
                likelihood=LikelihoodSpec(
                    law=DeltaLawSpec[Expression](v=state(mood.id)),
                    reasoning="Exact observation with an optional unknown calibration offset",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        ),
        dynamics=(
            mood_potential.revised(
                expression=restoring_potential(
                    mood.id, center=_CINT_MOOD_PARAMETER_ID, stiffness=rho_mood.id, quartic=0.0
                )
            ),
        ),
    )
    _parameters, distributions = without_parameters(
        dynamical_model_spec, obs_cat_intercepts, obs_cat_slopes
    )
    distributions = {
        identity: law
        for identity, law in distributions.items()
        if identity
        not in (
            rho_mood.distribution,
            sigma_mood.distribution,
        )
    }
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            ParameterSpec(
                id=_CINT_MOOD_PARAMETER_ID, name="cint_mood", description="center of cint_mood"
            ),
            rho_mood.revised(distribution=None),
            sigma_mood.revised(distribution=None),
        ),
        distributions=distributions,
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_sum() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_count()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_revised = mood_rating.revised(
        observation=mood_rating.observation.revised(
            measurement_dtype="continuous", aggregation=SummaryOperator.SUM
        )
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,))
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_last() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_sum()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_revised = mood_rating.revised(
        observation=mood_rating.observation.revised(aggregation=SummaryOperator.LAST)
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,))
    )


def _free_center_without_standardized_channel_fails__with_likelihoods() -> DynamicalModelSpec:
    _MOOD_FLAG_INDICATOR_ID = IndicatorId("indicator:f6e3cfa37e27221c32d7")
    _MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID = ParameterId(
        "parameter:5313b06cc9f7da6557b18d034bbbec6f84b3c074a293f6f005019171296895ac"
    )
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_sum()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_revised = mood.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_MOOD_FLAG_INDICATOR_ID,
                    name="mood_flag",
                    measurement_dtype="binary",
                    aggregation=SummaryOperator.LAST,
                    observation_window=None,
                ),
                likelihood=LikelihoodSpec(
                    law=BernoulliLogitsLawSpec[Expression](
                        logits=(
                            coefficient(
                                _MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID, "observation_intercept"
                            )
                            + (coefficient(1.0, "loading") * state(mood.id))
                        )
                    ),
                    reasoning="test",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID,
                name="manifest_mean_mood_flag",
                description="likelihood.intercept for manifest_mean_mood_flag",
            ),
        ),
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_true_sum() -> (
    DynamicalModelSpec
):
    _MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID = ParameterId(
        "parameter:bca1520f34995d14ca61465d185091a1771da172ba8bb846e3a977f55273ad4e"
    )
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_sum()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_likelihood = likelihood_named(dynamical_model_spec, "mood_rating")
    cint_mood = parameter_named(dynamical_model_spec, "cint_mood")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_rating_revised = mood_rating.revised(
        likelihood=mood_rating_likelihood.revised(
            law=DeltaLawSpec[Expression](
                v=(
                    coefficient(_MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(mood.id))
                )
            )
        )
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID,
                name="manifest_mean_mood_rating",
                description="Observation intercept for mood_rating",
            ),
            cint_mood,
            rho_mood,
            sigma_mood,
        ),
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_true_last() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_true_sum()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_revised = mood_rating.revised(
        observation=mood_rating.observation.revised(aggregation=SummaryOperator.LAST)
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,))
    )


def _exact_state_anchors_location_for_every_summary_complete_component_slots_true_count() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = (
        _exact_state_anchors_location_for_every_summary_complete_component_slots_true_sum()
    )
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_revised = mood_rating.revised(
        observation=mood_rating.observation.revised(
            measurement_dtype="count", aggregation=SummaryOperator.COUNT
        )
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,))
    )


def _manifest_intercept_is_rejected_for_categorical_channel__with_likelihoods() -> (
    DynamicalModelSpec
):
    _MANIFEST_MEAN_MOOD_KIND_PARAMETER_ID = ParameterId(
        "parameter:9ee7f16619dec0fee01eccacf7ced00544ea939bab64c6916764d6713ad0c0d2"
    )
    _MANIFEST_MEAN_MOOD_KIND_DISTRIBUTION_ID = DistributionId(
        "distribution:8fb0b4e3266ff97aba27d74c9c10662056a8c4d39d99176929f7646c4b91a73d"
    )
    dynamical_model_spec = _all_categorical_construct_gets_anchor_slope__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    mood_kind = indicator_named(dynamical_model_spec, "mood_kind")
    mood_kind_likelihood = likelihood_named(dynamical_model_spec, "mood_kind")
    obs_cat_intercepts = parameter_named(dynamical_model_spec, "obs_cat_intercepts")
    obs_cat_slopes = parameter_named(dynamical_model_spec, "obs_cat_slopes")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_kind_revised = mood_kind.revised(
        likelihood=mood_kind_likelihood.revised(
            law=CategoricalLawSpec[Expression](
                logits=CallExpression(
                    function="category_logits",
                    arguments=(
                        (
                            coefficient(
                                _MANIFEST_MEAN_MOOD_KIND_PARAMETER_ID, "observation_intercept"
                            )
                            + (coefficient(1.0, "loading") * state(mood.id))
                        ),
                        coefficient(obs_cat_intercepts.id, "category_intercepts"),
                        coefficient(obs_cat_slopes.id, "category_slopes"),
                    ),
                )
            )
        )
    )
    mood_revised = mood.revised(indicators=(mood_kind_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            rho_mood,
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_KIND_PARAMETER_ID,
                name="manifest_mean_mood_kind",
                description="Observation intercept for mood_kind",
                distribution=_MANIFEST_MEAN_MOOD_KIND_DISTRIBUTION_ID,
            ),
            sigma_mood,
            obs_cat_intercepts,
            obs_cat_slopes,
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _MANIFEST_MEAN_MOOD_KIND_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
        },
    )


def _static_t0_mean_gated_without_standardized_channel__with_likelihoods() -> DynamicalModelSpec:
    _TRAIT_FLAG_INDICATOR_ID = IndicatorId("indicator:2af50d06b16a559dee9e")
    _MANIFEST_MEAN_TRAIT_FLAG_PARAMETER_ID = ParameterId(
        "parameter:7604955ad068e49b834a6599b202b4de46728634b8aa30f2e7777cc713953a6c"
    )
    _MANIFEST_MEAN_TRAIT_FLAG_DISTRIBUTION_ID = DistributionId(
        "distribution:bec02fdd1551efc70d3c081bc3cdbad3004d831683e93df523591a64fa7e948a"
    )
    dynamical_model_spec = _static_t0_mean_free_with_standardized_channel__with_likelihoods()
    trait = construct_named(dynamical_model_spec, "trait")
    t0_sd_trait = parameter_named(dynamical_model_spec, "t0_sd_trait")
    t0_mean_trait = parameter_named(dynamical_model_spec, "t0_mean_trait")
    trait_revised = trait.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_TRAIT_FLAG_INDICATOR_ID,
                    name="trait_flag",
                    measurement_dtype="binary",
                    aggregation=SummaryOperator.LAST,
                    observation_window=None,
                ),
                likelihood=LikelihoodSpec(
                    law=BernoulliLogitsLawSpec[Expression](
                        logits=(
                            coefficient(
                                _MANIFEST_MEAN_TRAIT_FLAG_PARAMETER_ID, "observation_intercept"
                            )
                            + (coefficient(1.0, "loading") * state(trait.id))
                        )
                    ),
                    reasoning="test",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        ),
        coefficients=(
            coefficient(0.0, "initial_mean"),
            coefficient(t0_sd_trait.id, "initial_scale"),
        ),
    )
    parameters, distributions = without_parameters(dynamical_model_spec, t0_mean_trait)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (trait_revised,)),
        parameters=(
            *parameters,
            ParameterSpec(
                id=_MANIFEST_MEAN_TRAIT_FLAG_PARAMETER_ID,
                name="manifest_mean_trait_flag",
                description="likelihood.intercept for manifest_mean_trait_flag",
                distribution=_MANIFEST_MEAN_TRAIT_FLAG_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **distributions,
            _MANIFEST_MEAN_TRAIT_FLAG_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
        },
    )


def _manifest_intercept_remains_free_for_binary_channel__with_likelihoods() -> DynamicalModelSpec:
    _MOOD_FLAG_INDICATOR_ID = IndicatorId("indicator:f6e3cfa37e27221c32d7")
    _MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID = ParameterId(
        "parameter:92944ebc8cd24d4ef9e7c52a8cf36626b63e00598eba7e4a9781680d950652aa"
    )
    _MANIFEST_MEAN_MOOD_FLAG_DISTRIBUTION_ID = DistributionId(
        "distribution:12b1eb4208f179cfe7f24dc517d33059930f432b6492172f54c40e83e4ed69ab"
    )
    dynamical_model_spec = _all_categorical_construct_gets_anchor_slope__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    obs_cat_intercepts = parameter_named(dynamical_model_spec, "obs_cat_intercepts")
    obs_cat_slopes = parameter_named(dynamical_model_spec, "obs_cat_slopes")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_revised = mood.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_MOOD_FLAG_INDICATOR_ID,
                    name="mood_flag",
                    measurement_dtype="binary",
                    aggregation=SummaryOperator.LAST,
                    observation_window=None,
                ),
                likelihood=LikelihoodSpec(
                    law=BernoulliLogitsLawSpec[Expression](
                        logits=(
                            coefficient(
                                _MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID, "observation_intercept"
                            )
                            + (coefficient(1.0, "loading") * state(mood.id))
                        )
                    ),
                    reasoning="test",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        )
    )
    _parameters, distributions = without_parameters(
        dynamical_model_spec, obs_cat_intercepts, obs_cat_slopes
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            rho_mood,
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_FLAG_PARAMETER_ID,
                name="manifest_mean_mood_flag",
                description="Observation intercept for mood_flag",
                distribution=_MANIFEST_MEAN_MOOD_FLAG_DISTRIBUTION_ID,
            ),
            sigma_mood,
        ),
        distributions={
            **distributions,
            _MANIFEST_MEAN_MOOD_FLAG_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
        },
    )


def _manifest_intercept_is_rejected_for_standardized_channel__with_likelihoods() -> (
    DynamicalModelSpec
):
    _MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID = ParameterId(
        "parameter:bca1520f34995d14ca61465d185091a1771da172ba8bb846e3a977f55273ad4e"
    )
    _MANIFEST_MEAN_MOOD_RATING_DISTRIBUTION_ID = DistributionId(
        "distribution:d9087f182aea71973885f8cbf78f98441e37efbc8085c11f927f13088997b2e9"
    )
    dynamical_model_spec = _categorical_loading_pinned_in_mixed_construct__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    mood_rating = indicator_named(dynamical_model_spec, "mood_rating")
    mood_rating_likelihood = likelihood_named(dynamical_model_spec, "mood_rating")
    obs_sd_mood_rating = parameter_named(dynamical_model_spec, "obs_sd_mood_rating")
    obs_cat_intercepts = parameter_named(dynamical_model_spec, "obs_cat_intercepts")
    obs_cat_slopes = parameter_named(dynamical_model_spec, "obs_cat_slopes")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_rating_revised = mood_rating.revised(
        likelihood=mood_rating_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(_MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(mood.id))
                ),
                scale=coefficient(0.0, "observation_scale"),
            )
        )
    )
    mood_revised = mood.revised(indicators=(mood_rating_revised,))
    _parameters, distributions = without_parameters(
        dynamical_model_spec, obs_sd_mood_rating, obs_cat_intercepts, obs_cat_slopes
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            rho_mood,
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_RATING_PARAMETER_ID,
                name="manifest_mean_mood_rating",
                description="Observation intercept for mood_rating",
                distribution=_MANIFEST_MEAN_MOOD_RATING_DISTRIBUTION_ID,
            ),
            sigma_mood,
        ),
        distributions={
            **distributions,
            _MANIFEST_MEAN_MOOD_RATING_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
        },
    )


def _cutpoints_keep_free_base_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testorderedthresholds_test_cutpoints_keep_free_base_model_fixture.json"
    )


def _ordinal_only_construct_compiles__with_likelihoods() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testorderedthresholds_test_ordinal_only_construct_compiles__with_likelihoods.json"
    )


def _manifest_intercept_is_rejected_for_threshold_channel__with_likelihoods() -> DynamicalModelSpec:
    _MANIFEST_MEAN_MOOD_LEVEL_PARAMETER_ID = ParameterId(
        "parameter:0bdc340bfd8c14cfd0b29c37f345a0d3d12539d90d56333f5caff8e5aefa8faa"
    )
    _MANIFEST_MEAN_MOOD_LEVEL_DISTRIBUTION_ID = DistributionId(
        "distribution:a115b821036afd88a9926c11f8b6bb63d71a7ce1924ab784573f94d24eddf28b"
    )
    dynamical_model_spec = _ordinal_only_construct_compiles__with_likelihoods()
    mood = construct_named(dynamical_model_spec, "mood")
    mood_level = indicator_named(dynamical_model_spec, "mood_level")
    mood_level_likelihood = likelihood_named(dynamical_model_spec, "mood_level")
    obs_ordered_base_mood_level = parameter_named(
        dynamical_model_spec, "obs_ordered_base_mood_level"
    )
    obs_ordered_gaps_mood_level = parameter_named(
        dynamical_model_spec, "obs_ordered_gaps_mood_level"
    )
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    mood_level_revised = mood_level.revised(
        likelihood=mood_level_likelihood.revised(
            law=OrderedLogisticLawSpec[Expression](
                predictor=(
                    coefficient(_MANIFEST_MEAN_MOOD_LEVEL_PARAMETER_ID, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(mood.id))
                ),
                cutpoints=CallExpression(
                    function="ordered_cutpoints",
                    arguments=(
                        coefficient(obs_ordered_base_mood_level.id, "cutpoint_base"),
                        coefficient(obs_ordered_gaps_mood_level.id, "cutpoint_gaps"),
                    ),
                ),
            )
        )
    )
    mood_revised = mood.revised(indicators=(mood_level_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (mood_revised,)),
        parameters=(
            rho_mood,
            ParameterSpec(
                id=_MANIFEST_MEAN_MOOD_LEVEL_PARAMETER_ID,
                name="manifest_mean_mood_level",
                description="Observation intercept for mood_level",
                distribution=_MANIFEST_MEAN_MOOD_LEVEL_DISTRIBUTION_ID,
            ),
            sigma_mood,
            obs_ordered_base_mood_level,
            obs_ordered_gaps_mood_level,
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _MANIFEST_MEAN_MOOD_LEVEL_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
        },
    )


def _reference_prefers_continuous_over_ordinal__with_likelihoods() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testanchorsurfaces_test_reference_prefers_continuous_over_ordinal__with_likelihoods.json"
    )


def _free_center_with_standardized_channel_compiles__with_likelihoods() -> DynamicalModelSpec:
    return load_model_fixture(
        "identification_anchors/testlocationanchors_test_free_center_with_standardized_channel_compiles__with_likelihoods.json"
    )


pytestmark = pytest.mark.contract

# ═══════════════════════════════════════════════════════════════════════
# Fixture builders
# ═══════════════════════════════════════════════════════════════════════


def _indicator(
    name: str,
    construct_name: str,
    dtype: str,
    *,
    polarity: str = "positive",
) -> dict[str, Any]:
    levels: dict[str, list[str]] = {
        "ordinal": {"ordinal_levels": ["low", "medium", "high"]},
        "categorical": {"categorical_levels": ["a", "b", "c"]},
    }.get(dtype, {})
    return {
        "observation": {
            "id": fixture_entity_id("indicator", name),
            "name": name,
            "measurement_dtype": dtype,
            "aggregation": "mean" if dtype == "continuous" else "last",
            **levels,
        },
        "construct_id": fixture_entity_id("construct", construct_name),
        "construct_polarity": polarity,
    }


def _structure(
    construct_names: list[str],
    indicators: list[dict[str, Any]],
    *,
    time_invariant: set[str] | None = None,
) -> DynamicalModelSpec:
    from grassmoth.artifacts.model.indicator import IndicatorSpec

    dynamical_model_spec = make_model(construct_names)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(
                    temporal_status="time_invariant"
                    if construct.name in (time_invariant or set())
                    else "time_varying",
                    indicators=tuple(
                        IndicatorSpec.model_validate(
                            {key: value for key, value in row.items() if key != "construct_id"}
                        )
                        for row in indicators
                        if row["construct_id"] == construct.id
                    ),
                )
                for construct in dynamical_model_spec.constructs
            ),
        )
    )


_LIKELIHOOD_BY_DTYPE = {
    "continuous": (DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY),
    "binary": (DistributionFamily.BERNOULLI, LinkFunction.LOGIT),
    "ordinal": (DistributionFamily.ORDERED_LOGISTIC, LinkFunction.CUMULATIVE_LOGIT),
    "categorical": (DistributionFamily.CATEGORICAL, LinkFunction.SOFTMAX),
}


# ═══════════════════════════════════════════════════════════════════════
# Ordered-logistic: free threshold base, no centering
# ═══════════════════════════════════════════════════════════════════════


class TestOrderedThresholds:
    def test_cutpoints_keep_free_base(self):
        """The threshold base shifts the cutpoints instead of cancelling out."""
        dynamical_model_spec = _cutpoints_keep_free_base_model_fixture()
        laws = materialize_observation_laws(
            compile_model_fixture(dynamical_model_spec),
            {
                "obs_ordered_base": jnp.array([0.7]),
                "obs_ordered_gaps": jnp.array([[0.5]]),
            },
        )
        ordered = laws[0]
        assert isinstance(ordered, OrderedLogisticLawSpec)
        np.testing.assert_allclose(
            np.asarray(ordered.cutpoints.evaluate(jnp.zeros(()), jnp.ones(())))[None, :],
            np.array([[0.7, 1.2]]),
        )

    def test_ordinal_only_construct_compiles(self):
        """Well-at-zero anchors location; the fixed logistic link anchors scale."""
        dynamical_model_spec = _ordinal_only_construct_compiles__with_likelihoods()
        assert numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec)) is not None
        assert not any(numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec)))
        assert (
            float(compile_model_fixture(dynamical_model_spec).loading_block.template[0, 0]) == 1.0
        )
        assert not compile_model_fixture(dynamical_model_spec).loading_block.free_support[0, 0]

    def test_manifest_intercept_is_rejected_for_threshold_channel(self):
        with pytest.raises(AggregatedCompileError, match=r"Observation intercept.*is inactive"):
            compile_model_fixture(
                _manifest_intercept_is_rejected_for_threshold_channel__with_likelihoods()
            )


# ═══════════════════════════════════════════════════════════════════════
# Location anchors: equilibrium center and static t0 mean
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAnchors:
    def test_manifest_intercept_is_rejected_for_standardized_channel(self):
        with pytest.raises(AggregatedCompileError, match=r"Observation intercept.*is inactive"):
            compile_model_fixture(
                _manifest_intercept_is_rejected_for_standardized_channel__with_likelihoods()
            )

    def test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel(self):
        indicator = _indicator("fill_quantity", "dose", "continuous")
        indicator["observation"]["aggregation"] = "sum"
        dynamical_model_spec = (
            _manifest_intercept_remains_free_for_raw_gaussian_sum_channel__with_likelihoods()
        )

        assert numeric.observation_standardized(compile_model_fixture(dynamical_model_spec)) == (
            False,
        )
        assert compile_model_fixture(
            dynamical_model_spec
        ).observation_mean_block.free_support.tolist() == [True]

    def test_manifest_intercept_remains_free_for_binary_channel(self):
        dynamical_model_spec = (
            _manifest_intercept_remains_free_for_binary_channel__with_likelihoods()
        )
        assert compile_model_fixture(
            dynamical_model_spec
        ).observation_mean_block.free_support.tolist() == [True]

    def test_free_center_without_standardized_channel_fails(self):
        dynamical_model_spec = _free_center_without_standardized_channel_fails__with_likelihoods()
        with pytest.raises(ValueError, match="Construct 'mood' has no location anchor"):
            StructuralSelection(dynamical_model_spec, None)

    def test_free_center_with_standardized_channel_compiles(self):
        dynamical_model_spec = _free_center_with_standardized_channel_compiles__with_likelihoods()
        assert (
            numeric.observation_standardized(compile_model_fixture(dynamical_model_spec))
            is not None
        )
        assert numeric.observation_standardized(compile_model_fixture(dynamical_model_spec))[0]

    @pytest.mark.parametrize(
        ("affine", "model_payload"),
        [
            pytest.param(
                False,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_last,
                id="False-last",
            ),
            pytest.param(
                False,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_sum,
                id="False-sum",
            ),
            pytest.param(
                False,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_2_false_count,
                id="False-count",
            ),
            pytest.param(
                True,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_true_last,
                id="True-last",
            ),
            pytest.param(
                True,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_true_sum,
                id="True-sum",
            ),
            pytest.param(
                True,
                _exact_state_anchors_location_for_every_summary_complete_component_slots_true_count,
                id="True-count",
            ),
        ],
    )
    def test_exact_state_anchors_location_for_every_summary(self, affine, model_payload):
        payload = model_payload().model_dump_json()
        dynamical_model_spec = DynamicalModelSpec.model_validate_json(payload).materialized()
        if affine:
            with pytest.raises(ValueError, match="Construct 'mood' has no location anchor"):
                StructuralSelection(dynamical_model_spec, None)
        else:
            StructuralSelection(dynamical_model_spec, None)

    def test_static_t0_mean_gated_without_standardized_channel(self):
        dynamical_model_spec = (
            _static_t0_mean_gated_without_standardized_channel__with_likelihoods()
        )
        assert numeric.state_names(compile_model_fixture(dynamical_model_spec)) is not None
        trait_index = numeric.state_names(compile_model_fixture(dynamical_model_spec)).index(
            "trait"
        )
        assert not compile_model_fixture(dynamical_model_spec).initial_mean_block.free_support[
            trait_index
        ]

    def test_static_t0_mean_free_with_standardized_channel(self):
        dynamical_model_spec = _static_t0_mean_free_with_standardized_channel__with_likelihoods()
        assert numeric.state_names(compile_model_fixture(dynamical_model_spec)) is not None
        trait_index = numeric.state_names(compile_model_fixture(dynamical_model_spec)).index(
            "trait"
        )
        assert compile_model_fixture(dynamical_model_spec).initial_mean_block.free_support[
            trait_index
        ]

    def test_unmeasured_construct_stays_scientific_without_an_unidentified_state(self):
        plan = _structure(["mood", "ghost"], [_indicator("mood_rating", "mood", "continuous")])
        assert plan.get_construct(fixture_entity_id("construct", "ghost")).indicators == ()
        assert fixture_entity_id("construct", "ghost") not in selected_state_ids(
            StructuralSelection(plan, None)
        )


# ═══════════════════════════════════════════════════════════════════════
# Categorical: pinned loadings and anchor slopes
# ═══════════════════════════════════════════════════════════════════════


class TestCategoricalAnchors:
    def test_categorical_loading_pinned_in_mixed_construct(self):
        dynamical_model_spec = _categorical_loading_pinned_in_mixed_construct__with_likelihoods()
        assert numeric.observation_names(compile_model_fixture(dynamical_model_spec)) is not None
        assert numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec)) is not None
        cat_row = numeric.observation_names(compile_model_fixture(dynamical_model_spec)).index(
            "mood_kind"
        )
        assert (
            float(compile_model_fixture(dynamical_model_spec).loading_block.template[cat_row, 0])
            == 1.0
        )
        assert not compile_model_fixture(dynamical_model_spec).loading_block.free_support[
            cat_row, 0
        ]
        assert not numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec))[cat_row]

    def test_all_categorical_construct_gets_anchor_slope(self):
        dynamical_model_spec = _all_categorical_construct_gets_anchor_slope__with_likelihoods()
        assert numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec)) == (True,)
        assert numeric.observation_level_counts(compile_model_fixture(dynamical_model_spec)) == (3,)

        laws = materialize_observation_laws(
            compile_model_fixture(dynamical_model_spec),
            {
                "obs_cat_intercepts": jnp.array([[0.3, -0.4]]),
                "obs_cat_slopes": jnp.array([[9.9, 2.0]]),
            },
        )
        categorical = laws[0]
        assert isinstance(categorical, CategoricalLawSpec)
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(categorical.logits.evaluate)(jnp.zeros(()), jnp.ones(())))[
                None, 1:
            ],
            np.array([[1.0, 2.0]]),
        )

    def test_manifest_intercept_is_rejected_for_categorical_channel(self):
        with pytest.raises(AggregatedCompileError, match=r"Observation intercept.*is inactive"):
            compile_model_fixture(
                _manifest_intercept_is_rejected_for_categorical_channel__with_likelihoods()
            )


# ═══════════════════════════════════════════════════════════════════════
# Reference indicator preference and prior-surface activation
# ═══════════════════════════════════════════════════════════════════════


class TestAnchorSurfaces:
    def test_reference_prefers_continuous_over_ordinal(self):
        dynamical_model_spec = _reference_prefers_continuous_over_ordinal__with_likelihoods()
        assert numeric.observation_names(compile_model_fixture(dynamical_model_spec)) is not None
        continuous_row = numeric.observation_names(
            compile_model_fixture(dynamical_model_spec)
        ).index("mood_rating")
        ordinal_row = numeric.observation_names(compile_model_fixture(dynamical_model_spec)).index(
            "mood_level"
        )
        assert (
            float(
                compile_model_fixture(dynamical_model_spec).loading_block.template[
                    continuous_row, 0
                ]
            )
            == 1.0
        )
        assert compile_model_fixture(dynamical_model_spec).loading_block.free_support[
            ordinal_row, 0
        ]
