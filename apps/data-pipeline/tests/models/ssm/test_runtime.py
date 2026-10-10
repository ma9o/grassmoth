"""Tests for SSM runtime preparation helpers.

Covers: semantic prior binding and fit-input preparation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.expressions import Expression, coefficient, restoring_force, state
from nof1_causal_lab.artifacts.identity import DistributionId, IndicatorId, ParameterId
from nof1_causal_lab.artifacts.indicator import IndicatorPolarity, IndicatorSpec
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, NormalLawSpec
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.observation_data import ObservationDataset
from nof1_causal_lab.artifacts.observations import AuthoredObservationSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import (
    IdentityTransformSpec,
    InitialCorrelationTransformSpec,
    ParameterSpec,
)
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.models.ssm.runtime import (
    BoundPanel,
    bind_panel,
    prepare_fit_inputs,
    project_observation_data,
)
from nof1_causal_lab.sampler_config import SamplerSpec
from nof1_causal_lab.utils.observation_semantics import SummaryOperator
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.inference_fixtures import bind_panel_fixture, compile_fit_fixture, compile_model_fixture
from tests.model_fixtures import (
    _two_state_fixed_drift_model,
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    one_state_gaussian_model,
    parameter_for,
    parameter_laws,
    replace_parameters,
    without_parameters,
)


def _initial_state_correlation_prior_indices_are_dense_after_mask_filtering__make_spec() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "runtime/testbuilderpriorconversion_test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering__make_spec.json"
    )


def _gamma_emission_rejects_zero_observations__make_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "runtime/testobservationsupportvalidation_test_gamma_emission_rejects_zero_observations__make_spec.json"
    )


def _compile_distinguishes_incomplete_unsupported_and_bugs_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = one_state_gaussian_model()
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    return dynamical_model_spec.with_entities(
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                latent_0_dynamics_decay.id: dist.Delta(
                    v=0.5, log_density=0.0, event_dim=0, validate_args=False
                )
            },
        )
    )


def _initial_state_correlation_priors_are_bounded_to_correlation_scale_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = load_model_fixture(
        "dynamics_config/scientific_model_roundtrip_preserves_derived_dynamics_model_fixture.json"
    )
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    (latent_0_potential,) = latent_0.dynamics
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_latent_1_hill_emax = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EMAX, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_n = parameter_for(
        dynamical_model_spec, SiteKind.HILL_N, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_ec50 = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EC50, "latent_0", "latent_1"
    )
    latent_0_latent_1_t0_var_lower = parameter_for(
        dynamical_model_spec, SiteKind.T0_VAR_LOWER, "latent_0", "latent_1"
    )
    latent_0_revised = latent_0.revised(
        dynamics=(
            DriftMechanismSpec(
                id=latent_0_potential.id,
                expression=restoring_force(
                    latent_0.id, center=0.0, stiffness=latent_0_dynamics_decay.id, quartic=0.0
                ),
            ),
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec,
        latent_0_latent_1_hill_emax,
        latent_0_latent_1_hill_n,
        latent_0_latent_1_hill_ec50,
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            tuple(
                edge
                for edge in dynamical_model_spec.edges
                if (edge.cause.name, edge.effect.name) not in (("latent_0", "latent_1"),)
            ),
            (latent_0_revised,),
        ),
        parameters=replace_parameters(
            parameters,
            latent_0_latent_1_t0_var_lower.revised(transform=InitialCorrelationTransformSpec()),
        ),
        distributions={
            key: law
            for key, law in parameter_laws(
                dynamical_model_spec,
                {
                    latent_0_latent_1_t0_var_lower.id: dist.Normal(
                        loc=0.2, scale=0.8, validate_args=False
                    )
                },
            ).items()
            if key in distributions
        },
    )


def _initial_state_mean_and_sd_priors_bind_to_t0_sites_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = _initial_state_correlation_priors_are_bounded_to_correlation_scale_with_parameter_distributions()
    latent_0_latent_1_t0_var_lower = parameter_for(
        dynamical_model_spec, SiteKind.T0_VAR_LOWER, "latent_0", "latent_1"
    )
    latent_0_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_0")
    latent_0_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_0")
    latent_1_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_1")
    latent_1_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_1")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            latent_0_latent_1_t0_var_lower.revised(transform=IdentityTransformSpec()),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                latent_0_t0_means.id: dist.Normal(loc=0.2, scale=0.3, validate_args=False),
                latent_0_t0_var_diag.id: dist.HalfNormal(scale=0.7, validate_args=False),
                latent_1_t0_means.id: dist.Normal(loc=0.4, scale=0.5, validate_args=False),
                latent_1_t0_var_diag.id: dist.HalfNormal(scale=0.9, validate_args=False),
                latent_0_latent_1_t0_var_lower.id: dist.Uniform(
                    low=-1.0, high=1.0, validate_args=False
                ),
            },
        ),
    )


def _manifest_standardization_of_constant_column_centers_without_scaling__make_spec() -> (
    DynamicalModelSpec
):
    _X_INDICATOR_ID = IndicatorId("indicator:1f4c67cecb9238ee1a80")
    _LATENT_0_X_MANIFEST_VAR_DIAG_PARAMETER_ID = ParameterId(
        "parameter:ec27e91f16b480062ddc51afd32ac245399b05df25eb8feb93e3fad16abc9ccd"
    )
    _LATENT_0_X_MANIFEST_VAR_DIAG_DISTRIBUTION_ID = DistributionId(
        "distribution:62b85664eaabfedf901244cf2eb42462b317bb4c970ac7c04e89cdf83d1b79e5"
    )
    dynamical_model_spec = one_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    latent_0_manifest_0_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "manifest_0"
    )
    latent_0_revised = latent_0.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_X_INDICATOR_ID,
                    name="x",
                    measurement_dtype="continuous",
                    aggregation=SummaryOperator.LAST,
                    observation_window=None,
                ),
                likelihood=LikelihoodSpec(
                    law=NormalLawSpec[Expression](
                        loc=(
                            coefficient(0.0, "observation_intercept")
                            + (coefficient(1.0, "loading") * state(latent_0.id))
                        ),
                        scale=coefficient(
                            _LATENT_0_X_MANIFEST_VAR_DIAG_PARAMETER_ID, "observation_scale"
                        ),
                    ),
                    standardized=True,
                    reasoning="Test likelihood",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec, latent_0_manifest_0_manifest_var_diag
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=(
            *parameters,
            ParameterSpec(
                id=_LATENT_0_X_MANIFEST_VAR_DIAG_PARAMETER_ID,
                name=_LATENT_0_X_MANIFEST_VAR_DIAG_PARAMETER_ID,
                description="Fixture quantity",
                distribution=_LATENT_0_X_MANIFEST_VAR_DIAG_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **distributions,
            _LATENT_0_X_MANIFEST_VAR_DIAG_DISTRIBUTION_ID: dist.HalfNormal(
                scale=1.0, validate_args=False
            ),
        },
    )


def _stress_interval_model(window="31d") -> DynamicalModelSpec:
    _STRESS_SCORE_INDICATOR_ID = IndicatorId("indicator:3696aef3ff6f446744e5")
    _LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_PARAMETER_ID = ParameterId(
        "parameter:cfa73aad8f98fefa2c03e109a6c5619b67bc4f1ab79bcaf100e70f7cfe7abc80"
    )
    _LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_DISTRIBUTION_ID = DistributionId(
        "distribution:6e81bd018a8f326fe633d69e5b75bb5a5b47a3f4480e625579f1b3aa986def0b"
    )
    dynamical_model_spec = one_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    latent_0_manifest_0_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "manifest_0"
    )
    latent_0_revised = latent_0.revised(
        indicators=(
            IndicatorSpec(
                observation=AuthoredObservationSpec(
                    id=_STRESS_SCORE_INDICATOR_ID,
                    name="stress_score",
                    measurement_dtype="continuous",
                    aggregation=SummaryOperator.MEAN,
                    observation_window=window,
                ),
                likelihood=LikelihoodSpec(
                    law=NormalLawSpec[Expression](
                        loc=(
                            coefficient(0.0, "observation_intercept")
                            + (coefficient(1.0, "loading") * state(latent_0.id))
                        ),
                        scale=coefficient(
                            _LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_PARAMETER_ID,
                            "observation_scale",
                        ),
                    ),
                    reasoning="Test likelihood",
                ),
                construct_polarity=IndicatorPolarity.POSITIVE,
            ),
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec, latent_0_manifest_0_manifest_var_diag
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=(
            *parameters,
            ParameterSpec(
                id=_LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_PARAMETER_ID,
                name=_LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_PARAMETER_ID,
                description="Fixture quantity",
                distribution=_LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **distributions,
            _LATENT_0_STRESS_SCORE_MANIFEST_VAR_DIAG_DISTRIBUTION_ID: dist.HalfNormal(
                scale=1.0, validate_args=False
            ),
        },
    )


def _ar_prior_rejects_negative_support_make_prior_model() -> DynamicalModelSpec:
    return load_model_fixture(
        "runtime/testbuilderpriorconversion_test_ar_prior_rejects_negative_support_make_prior_model.json"
    )


def _sparse_wide_nulls_become_nan_without_fill_forward__make_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "runtime/testpreparefitinputs_test_sparse_wide_nulls_become_nan_without_fill_forward__make_spec.json"
    )


def _manifest_standardization_applies_only_to_standardized_channels__make_spec() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = _sparse_wide_nulls_become_nan_without_fill_forward__make_spec()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    x = indicator_named(dynamical_model_spec, "x")
    x_likelihood = likelihood_named(dynamical_model_spec, "x")
    x_revised = x.revised(likelihood=x_likelihood.revised(standardized=True))
    latent_0_revised = latent_0.revised(indicators=(x_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,))
    )


def _stress_mood_model() -> DynamicalModelSpec:
    return load_model_fixture("runtime/stress_mood_model.json")


# =============================================================================
# normalize_prior_params
# =============================================================================


@pytest.mark.contract
class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        with pytest.raises(ValueError, match=r"support within \[0, 1\]"):
            compile_priors(
                compile_model_fixture(_ar_prior_rejects_negative_support_make_prior_model()),
                StructuralSelection(
                    _ar_prior_rejects_negative_support_make_prior_model(),
                    None,
                ),
            )

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        dynamical_model_spec = _initial_state_correlation_priors_are_bounded_to_correlation_scale_with_parameter_distributions()
        law = compile_priors(
            compile_model_fixture(dynamical_model_spec),
            StructuralSelection(dynamical_model_spec, None),
        )[0]["t0_var_lower_free"]
        np.testing.assert_allclose(law.base_dist.loc, [0.2])
        np.testing.assert_allclose(law.base_dist.scale, [0.8])
        np.testing.assert_allclose(law.low, [-1.0])
        np.testing.assert_allclose(law.high, [1.0])

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        dynamical_model_spec = _two_state_fixed_drift_model()
        means = [
            p
            for p in dynamical_model_spec.parameters
            if dynamical_model_spec.parameter_context(p.id).quantity == SiteKind.T0_MEANS
        ]
        dynamical_model_spec = (
            _initial_state_mean_and_sd_priors_bind_to_t0_sites_with_parameter_distributions()
        )
        priors, bindings, _ = compile_priors(
            compile_model_fixture(dynamical_model_spec),
            StructuralSelection(dynamical_model_spec, None),
        )
        np.testing.assert_allclose(priors["t0_means_free"].loc, [0.2, 0.4])
        np.testing.assert_allclose(priors["t0_var_diag_free"].scale, [0.7, 0.9])
        assert [
            {binding.parameter_id: binding for binding in bindings}[p.id].flat_index for p in means
        ] == [0, 1]

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        dynamical_model_spec = (
            _initial_state_correlation_prior_indices_are_dense_after_mask_filtering__make_spec()
        )
        _, bindings, _ = compile_priors(
            compile_model_fixture(dynamical_model_spec),
            StructuralSelection(dynamical_model_spec, None),
        )
        correlation = next(
            p
            for p in dynamical_model_spec.parameters
            if dynamical_model_spec.parameter_context(p.id).quantity == SiteKind.T0_VAR_LOWER
        )
        assert {binding.parameter_id: binding for binding in bindings}[
            correlation.id
        ].flat_index == 0
        assert compile_model_fixture(
            dynamical_model_spec
        ).initial_covariance_block.correlation_positions == ((2, 1),)

    def test_component_dynamics_parameters_bind_to_their_own_terms(self):
        from nof1_causal_lab.models.ssm.structure.sites import (
            CompiledEdgeTarget,
            CompiledNodeTarget,
        )

        dynamical_model_spec = _stress_mood_model()
        compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
        _, bindings, _ = compile_priors(
            compiled_dynamical_model, StructuralSelection(dynamical_model_spec, None)
        )
        by_parameter = {binding.parameter_id: binding for binding in bindings}
        sites = {site.name: site for site in compiled_dynamical_model.site_registry}
        for parameter in dynamical_model_spec.parameters:
            if any(
                owner.kind == "mechanism"
                for owner in dynamical_model_spec.parameter_context(parameter.id).owners
            ):
                binding = by_parameter[parameter.id]
                assert binding.site is sites[binding.site.name]
                target = binding.target
                assert isinstance(target, (CompiledNodeTarget, CompiledEdgeTarget))
                component = next(
                    item
                    for item in compiled_dynamical_model.dynamics.spec.components
                    if parameter.id in item.coefficients.values()
                )
                assert target.target_index == component.target
                if component.source is None:
                    assert isinstance(target, CompiledNodeTarget)
                else:
                    assert isinstance(target, CompiledEdgeTarget)
                    assert target.source_index == component.source

    def test_cross_lag_prior_requires_the_declared_measurement_clock(self):
        dynamical_model_spec = _stress_mood_model().revised(measurement_clock=None)
        with pytest.raises(ValueError, match="measurement clock"):
            compile_priors(
                compile_model_fixture(dynamical_model_spec),
                StructuralSelection(dynamical_model_spec, None),
            )


@pytest.mark.contract
class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})
        dynamical_model_spec = _gamma_emission_rejects_zero_observations__make_spec()

        from nof1_causal_lab.models.ssm.observation_support import validate_observation_support

        failure = validate_observation_support(compile_model_fixture(dynamical_model_spec), X)
        assert isinstance(failure, ObservationPreflightFailure)
        assert "Observation support check failed" in failure.message


@pytest.mark.contract
class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        dynamical_model_spec = _sparse_wide_nulls_become_nan_without_fill_forward__make_spec()
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "x": [10.0, None],
                "y": [None, 30.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(
            compile_model_fixture(dynamical_model_spec), wide
        )

        assert manifest_names == ("x", "y")
        assert jnp.allclose(times, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.isclose(observations[0, 0], 10.0)
        assert jnp.isnan(observations[0, 1])
        assert jnp.isnan(observations[1, 0])
        assert jnp.isclose(observations[1, 1], 30.0)

    def test_manifest_standardization_applies_only_to_standardized_channels(self):
        """prepare_fit_inputs should deterministically standardize only marked manifests."""
        dynamical_model_spec = (
            _manifest_standardization_applies_only_to_standardized_channels__make_spec()
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "x": [10.0, 12.0, 14.0],
                "y": [5.0, 6.0, 7.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(
            compile_model_fixture(dynamical_model_spec), wide
        )

        assert manifest_names == ("x", "y")
        np.testing.assert_allclose(np.asarray(times), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.asarray(observations[:, 0]), np.array([-1.0, 0.0, 1.0]), rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(observations[:, 1]), np.array([5.0, 6.0, 7.0]))

    def test_manifest_standardization_of_constant_column_centers_without_scaling(self):
        """A zero-variance standardized column becomes exactly zero (divisor 1)."""
        dynamical_model_spec = (
            _manifest_standardization_of_constant_column_centers_without_scaling__make_spec()
        )
        wide = pl.DataFrame({"time": [0.0, 1.0], "x": [4.2, 4.2]})

        observations, _times, _names, _wide = prepare_fit_inputs(
            compile_model_fixture(dynamical_model_spec), wide
        )

        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([0.0, 0.0]))


class TestPrepareModelRuntime:
    @pytest.mark.contract
    def test_selected_indicators_keep_the_panel_origin_and_initial_grid_point(self):
        from nof1_causal_lab.models.ssm.observation_support import (
            augment_wide_data_with_support_boundaries,
        )
        from nof1_causal_lab.utils.observation_rows import prepared_time_origin

        early = one_state_gaussian_model()
        late = DynamicalModelSpec.model_validate_json(
            early.model_dump_json()
            .replace(str(early.indicators[0].observation.id), "indicator:late")
            .replace(early.indicators[0].observation.name, "late_obs")
        ).materialized()
        rows = pl.DataFrame(
            {
                "indicator_id": [
                    early.indicators[0].observation.id,
                    late.indicators[0].observation.id,
                ],
                "value": [1.0, 2.0],
                "anchor_time": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
                "support_start": [datetime(2024, 1, 1), datetime(2024, 1, 11)],
                "support_end": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
                "support_kind": ["point", "point"],
            }
        )
        origin = prepared_time_origin(rows, None)
        assert origin == datetime(2024, 1, 1, tzinfo=UTC)
        for model, expected in ((early, 1.0), (late, 11.0)):
            projected = project_observation_data(
                rows, compiled_dynamical_model=compile_model_fixture(model), time_origin=origin
            )
            assert not isinstance(projected, ObservationPreflightFailure)
            (wide, selected) = projected
            assert wide["time"].to_list() == [expected]
            augmented = augment_wide_data_with_support_boundaries(
                selected, wide, time_origin=origin
            )
            assert augmented["time"].to_list() == [0.0, expected]
            assert augmented[model.indicators[0].observation.name][0] is None

    @pytest.mark.contract
    def test_preserves_long_observation_metadata_and_augments_support_boundaries(self, caplog):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["31d"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )

        inputs = compile_fit_fixture(_stress_interval_model())

        with caplog.at_level("INFO"):
            runtime = bind_panel(
                ObservationDataset.from_frame(
                    data_for_model,
                    tuple(
                        item.observation for item in inputs.compiled_dynamical_model.observations
                    ),
                    time_origin=datetime(2024, 1, 1, tzinfo=UTC),
                ),
                time_origin=datetime(2024, 1, 1, tzinfo=UTC),
                compiled_dynamical_model=inputs.compiled_dynamical_model,
            )

        assert isinstance(runtime, BoundPanel)
        assert runtime.rows.column_names == data_for_model.columns
        assert pl.DataFrame(runtime.rows)["observation_window"][0] == "31d"
        assert pl.DataFrame(runtime.rows)["support_end"][0] == datetime(2024, 2, 1)
        assert pl.DataFrame(runtime.rows)["anchor_time"][0] == datetime(2024, 2, 1)
        assert isinstance(runtime, BoundPanel)
        assert runtime.times.tolist() == [0.0, 31.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.manifest_names == ("stress_score",)
        assert runtime.observation_support.support_kinds == ("interval",)
        assert runtime.observation_support.summary_operators == ("mean",)
        assert runtime.observation_support.anchor_policies == ("support_end",)
        assert runtime.observation_support.observation_windows == ("31d",)
        assert runtime.observation_support.requires_interval_summary_handling is True
        assert runtime.observation_support.interval_summary_manifest_names == ("stress_score",)
        assert runtime.observation_support.support_start_times.shape == (2, 1)
        assert runtime.observation_support.support_end_times.shape == (2, 1)
        assert runtime.observation_support.support_start_times[1, 0] == pytest.approx(0.0)
        assert runtime.observation_support.support_end_times[1, 0] == pytest.approx(31.0)
        assert runtime.observation_support.interval_prev_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_curr_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_weights.shape == (2, 1, 1)
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [0]]
        assert runtime.observation_support.interval_prev_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_curr_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(31.0)
        assert numeric.observation_names(runtime.compiled_dynamical_model) == ("stress_score",)

    @pytest.mark.contract
    def test_compiles_overlapping_interval_windows_into_concurrent_slots(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:3696aef3ff6f446744e5",
                    "indicator:3696aef3ff6f446744e5",
                ],
                "value": [3.0, 5.0],
                "anchor_time": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
                "support_kind": ["interval", "interval"],
                "summary_operator": ["mean", "mean"],
                "anchor_policy": ["support_end", "support_end"],
                "observation_window": ["2d", "2d"],
                "support_start": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                "support_end": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
            }
        )

        inputs = compile_fit_fixture(_stress_interval_model("2d"))

        runtime = bind_panel(
            ObservationDataset.from_frame(
                data_for_model,
                tuple(item.observation for item in inputs.compiled_dynamical_model.observations),
                time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            compiled_dynamical_model=inputs.compiled_dynamical_model,
        )

        assert isinstance(runtime, BoundPanel)
        assert runtime.times.tolist() == [0.0, 1.0, 2.0, 3.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [-1], [0], [1]]
        assert runtime.observation_support.interval_weights.shape == (4, 1, 2)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 1] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[3, 0, 1] == pytest.approx(1.0)

    @pytest.mark.inference(concern="predictive")
    def test_prior_predictive_reuses_prepared_support_schedule(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["31d"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )
        model = compile_fit_fixture(_stress_interval_model())
        runtime = bind_panel(
            ObservationDataset.from_frame(
                data_for_model,
                tuple(item.observation for item in model.compiled_dynamical_model.observations),
                time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            compiled_dynamical_model=model.compiled_dynamical_model,
        )

        from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch

        assert isinstance(runtime, BoundPanel)
        batch = generate_simulation_batch(
            runtime,
            start=float(runtime.times[0]),
            end=float(runtime.times[-1]),
            draws=3,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            compute_budget=COMPUTE_BUDGET,
        )
        from nof1_causal_lab.models.ssm.predictive.simulation import SimulationBatch

        assert isinstance(batch, SimulationBatch)
        samples = batch.prediction

        assert samples.trajectory.observations.shape == (3, 2, 1)
        assert samples.trajectory.observations_mask.shape == (3, 2, 1)
        assert jnp.isnan(samples.trajectory.observations[:, 0, 0]).all()
        assert jnp.isfinite(samples.trajectory.observations[:, 1, 0]).all()
        assert (~samples.trajectory.observations_mask[:, 0, 0]).all()
        assert samples.trajectory.observations_mask[:, 1, 0].all()


@pytest.mark.contract
def test_compiled_inputs_own_runtime_derivations(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import prior_compilation

    inputs = compile_fit_fixture(one_state_gaussian_model())

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("runtime recompiled its evidence")

    monkeypatch.setattr(prior_compilation, "compile_priors", unexpected_compile)
    monkeypatch.setattr(prior_compilation, "bind_parameters", unexpected_compile)
    panel = bind_panel_fixture(inputs.compiled_dynamical_model, jnp.zeros((2, 1)), jnp.arange(2.0))
    assert panel.compiled_dynamical_model is inputs.compiled_dynamical_model
    assert panel.indicator_ids == tuple(
        observation.id for observation in inputs.compiled_dynamical_model.observations
    )
    assert panel.rows.column("indicator_id").to_pylist() == [str(panel.indicator_ids[0])] * 2
    with pytest.raises(AttributeError):
        panel.compiled_dynamical_model = inputs.compiled_dynamical_model
    with pytest.raises(ValueError, match="read-only"):
        panel.observation_support.anchor_times[0] = 7


@pytest.mark.contract
def test_compile_distinguishes_incomplete_unsupported_and_bugs(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import inputs as compiler

    incomplete = compiler.compile_ssm_inputs_from_model(
        StructuralSelection(DynamicalModelSpec(), None)
    )
    assert isinstance(incomplete, compiler.IncompleteModel)
    dynamical_model_spec = one_state_gaussian_model()
    unsupported = (
        _compile_distinguishes_incomplete_unsupported_and_bugs_with_parameter_distributions()
    )
    assert isinstance(
        compiler.compile_ssm_inputs_from_model(StructuralSelection(unsupported, None)),
        compiler.UnsupportedFit,
    )

    def broken_compiler(_compiled, _authored):
        raise ValueError("internal compiler bug")

    monkeypatch.setattr(compiler, "compile_priors", broken_compiler)
    with pytest.raises(ValueError, match="internal compiler bug"):
        compiler.compile_ssm_inputs_from_model(StructuralSelection(dynamical_model_spec, None))


@pytest.mark.contract
def test_fit_resolves_incomplete_model_before_panel_preparation(monkeypatch):
    from nof1_causal_lab.actions.inference import fit as fitting
    from nof1_causal_lab.models.ssm import runtime

    def unexpected_panel(*_args, **_kwargs):
        raise AssertionError("panel prepared before fit capability was resolved")

    monkeypatch.setattr(runtime, "bind_panel", unexpected_panel)
    result = fitting.fit_model(
        StructuralSelection(DynamicalModelSpec(), None),
        ObservationDataset(series={}, time_origin=datetime(2024, 1, 1, tzinfo=UTC)),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        sampler=SamplerSpec(),
    )
    assert not result["fitted"]
    assert result["error"]
