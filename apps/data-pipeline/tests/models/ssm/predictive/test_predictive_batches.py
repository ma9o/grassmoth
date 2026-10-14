"""Predictive batch boundaries and exact discrete diagnostic statistics."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.identity import DistributionId, MechanismId, ParameterId
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.expressions import (
    Expression,
    coefficient,
    restoring_force,
    state,
)
from nof1_causal_lab.artifacts.model.likelihood import StudentTLawSpec
from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model.parameter import SiteKind
from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.predictive.statistics import observation_signal_and_variance
from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
from tests.support.inference import compile_model_fixture
from tests.support.model import (
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    one_state_gaussian_model,
    parameter_for,
    replace_parameters,
    without_parameters,
)


def _diagnostic_noise_matches_point_and_interval_execution_model_fixture_families0_false_1_0001e_08() -> (
    DynamicalModelSpec
):
    _LATENT_0_DRIFT_MECHANISM_ID = MechanismId("mechanism:5ced39cf0fae44636f2b")
    _LATENT_0_DIFFUSION_DIAG_DISTRIBUTION_ID = DistributionId(
        "distribution:6bbcc6530d30b7047a9232a9f85ac4c291df198857e92cbaa2b6a87f718be3a4"
    )
    _LATENT_0_T0_MEANS_DISTRIBUTION_ID = DistributionId(
        "distribution:a7ffde8dd9d9963518bc2ebe62ba9e41e53ee39bca8f4fa9d4d105004c82f5d1"
    )
    _LATENT_0_T0_VAR_DIAG_DISTRIBUTION_ID = DistributionId(
        "distribution:e9b0f86010c60b0e69c22ed8d0c1d7f416881c51406b24202ec0007c519d2aeb"
    )
    _LATENT_0_MANIFEST_0_MANIFEST_VAR_DIAG_DISTRIBUTION_ID = DistributionId(
        "distribution:e00bc5d0d99932524d21f2db2e76aa813087b7b29e2a3195d42f8b28199798d8"
    )
    dynamical_model_spec = one_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_diffusion_diag = parameter_for(
        dynamical_model_spec, SiteKind.DIFFUSION_DIAG, "latent_0"
    )
    latent_0_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_0")
    latent_0_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_0")
    latent_0_manifest_0_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "manifest_0"
    )
    latent_0_revised = latent_0.revised(
        dynamics=(
            DriftMechanismSpec(
                id=_LATENT_0_DRIFT_MECHANISM_ID,
                expression=restoring_force(latent_0.id, center=0.0, stiffness=1.0, quartic=0.0),
            ),
        )
    )
    parameters, distributions = without_parameters(dynamical_model_spec, latent_0_dynamics_decay)
    distributions = {
        identity: law
        for identity, law in distributions.items()
        if identity
        not in (
            latent_0_t0_means.distribution,
            latent_0_t0_var_diag.distribution,
            latent_0_diffusion_diag.distribution,
            latent_0_manifest_0_manifest_var_diag.distribution,
        )
    }
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=replace_parameters(
            parameters,
            latent_0_diffusion_diag.revised(distribution=_LATENT_0_DIFFUSION_DIAG_DISTRIBUTION_ID),
            latent_0_t0_means.revised(distribution=_LATENT_0_T0_MEANS_DISTRIBUTION_ID),
            latent_0_t0_var_diag.revised(distribution=_LATENT_0_T0_VAR_DIAG_DISTRIBUTION_ID),
            latent_0_manifest_0_manifest_var_diag.revised(
                distribution=_LATENT_0_MANIFEST_0_MANIFEST_VAR_DIAG_DISTRIBUTION_ID
            ),
        ),
        distributions={
            **distributions,
            _LATENT_0_DIFFUSION_DIAG_DISTRIBUTION_ID: dist.HalfNormal(
                scale=1.0, validate_args=False
            ),
            _LATENT_0_T0_MEANS_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=1.0, validate_args=False
            ),
            _LATENT_0_T0_VAR_DIAG_DISTRIBUTION_ID: dist.HalfNormal(scale=1.0, validate_args=False),
            _LATENT_0_MANIFEST_0_MANIFEST_VAR_DIAG_DISTRIBUTION_ID: dist.HalfNormal(
                scale=1.0, validate_args=False
            ),
        },
    )


def _student_t_model() -> DynamicalModelSpec:
    _INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_PARAMETER_ID = ParameterId(
        "parameter:5d755d2d8d1fafa18d1718ae376e47cb39c4fa4db501e16c8a098eff75cb51b3"
    )
    _INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_DISTRIBUTION_ID = DistributionId(
        "distribution:f183bae941d1033fb3c0fcbfa1f5898921d81e1789273b58f0f526eff9629fe8"
    )
    dynamical_model_spec = _diagnostic_noise_matches_point_and_interval_execution_model_fixture_families0_false_1_0001e_08()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    manifest_0 = indicator_named(dynamical_model_spec, "manifest_0")
    manifest_0_likelihood = likelihood_named(dynamical_model_spec, "manifest_0")
    latent_0_manifest_0_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "manifest_0"
    )
    manifest_0_revised = manifest_0.revised(
        likelihood=manifest_0_likelihood.revised(
            law=StudentTLawSpec[Expression](
                df=coefficient(
                    _INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_PARAMETER_ID,
                    "degrees_of_freedom",
                ),
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(latent_0.id))
                ),
                scale=coefficient(latent_0_manifest_0_manifest_var_diag.id, "observation_scale"),
            )
        )
    )
    latent_0_revised = latent_0.revised(indicators=(manifest_0_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_PARAMETER_ID,
                name="indicator:a6c99a629d09c47f8b69/degrees_of_freedom",
                description="Explicit coefficient for the diagnostic fixture.",
                distribution=_INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _INDICATOR_A6C99A629D09C47F8B69_DEGREES_OF_FREEDOM_DISTRIBUTION_ID: dist.HalfNormal(
                scale=1.0, validate_args=False
            ),
        },
    )


def _scalar_diagnostics_use_projected_means_at_supported_times_model_fixture() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "predictive_batches/scalar_diagnostics_use_projected_means_at_supported_times_model_fixture.json"
    )


def _gaussian_poisson_model() -> DynamicalModelSpec:
    return load_model_fixture("predictive_batches/gaussian_poisson_model.json")


def _undefined_student_moments_produce_an_explicit_diagnostic_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "predictive_batches/undefined_student_moments_produce_an_explicit_diagnostic_model_fixture.json"
    )


def _discrete_diagnostics_share_cutpoints_anchors_and_padded_probabilities_model_fixture() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "predictive_batches/discrete_diagnostics_share_cutpoints_anchors_and_padded_probabilities_model_fixture.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


def _trajectory():
    observations = jnp.zeros((2, 3, 2))
    return PredictiveTrajectory(
        latents=jnp.zeros((2, 3, 1)),
        linear_predictors=observations,
        observations=observations,
        observations_mask=jnp.ones(observations.shape, dtype=bool),
        expected_observations=observations,
    )


@pytest.mark.contract
@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("latents", jnp.zeros((2, 1))),
        ("latents", jnp.zeros((2, 4, 1))),
        ("observations", jnp.zeros((1, 3, 2))),
        ("linear_predictors", jnp.zeros((2, 3, 1))),
        ("expected_observations", jnp.zeros((2, 4, 2))),
        ("observations_mask", jnp.ones((2, 3, 2))),
    ],
)
def test_predictive_trajectory_rejects_misaligned_axes(field, values):
    with pytest.raises(ValueError, match=r"axes|axis|boolean"):
        replace(_trajectory(), **{field: values})


@pytest.mark.contract
def test_predictive_batch_preserves_pairing_and_rejects_misaligned_parameters():
    batch = PredictiveDraws(
        parameters={"coefficient": jnp.array([1.0, 2.0])},
        trajectory=_trajectory(),
        reference=_trajectory(),
    )
    assert batch.n_draws == 2
    assert jax.block_until_ready(batch).parameters.keys() == {"coefficient"}
    with pytest.raises(ValueError, match="draw axis"):
        replace(batch, parameters={"coefficient": jnp.ones(3)})
    with pytest.raises(ValueError, match="Reference"):
        replace(batch, reference=replace(_trajectory(), latents=jnp.zeros((2, 3, 2))))


@pytest.mark.inference(concern="predictive")
def test_discrete_diagnostics_share_cutpoints_anchors_and_declared_probabilities():
    dynamical_model_spec = (
        _discrete_diagnostics_share_cutpoints_anchors_and_padded_probabilities_model_fixture()
    )
    raw = {
        "obs_ordered_base": jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        "obs_ordered_gaps": jnp.ones((2, 2, 2)),
        "obs_cat_intercepts": jnp.zeros((2, 2, 3)),
        # The categorical anchor must replace 99 by +1, and padded entries vanish.
        "obs_cat_slopes": jnp.full((2, 2, 3), 99.0),
    }
    assert numeric.categorical_anchors(compile_model_fixture(dynamical_model_spec))[1]
    predictors = jnp.broadcast_to(jnp.array([-100.0, 0.0, 100.0])[None, :, None], (2, 3, 2))
    batch = PredictiveDraws(
        parameters={**raw, "manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2))},
        trajectory=replace(_trajectory(), linear_predictors=predictors),
    )
    indices = np.arange(3)
    ordered, variance = observation_signal_and_variance(
        compile_model_fixture(dynamical_model_spec), batch, 0, indices
    )
    assert variance is None
    np.testing.assert_allclose(ordered.sum(axis=2), 1.0, atol=1e-7)
    np.testing.assert_array_equal(ordered[:, 0], [[1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(ordered[:, -1], [[0.0, 0.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(ordered[:, 1, 0], jax.nn.sigmoid(jnp.array([0.0, 1.0])))

    categorical, variance = observation_signal_and_variance(
        compile_model_fixture(dynamical_model_spec), batch, 1, indices
    )
    assert variance is None
    assert categorical.shape == (2, 3, 2)
    np.testing.assert_allclose(categorical[:, 1, :2], 0.5)
    np.testing.assert_array_equal(categorical[:, 0, :2], [[1.0, 0.0]] * 2)
    np.testing.assert_array_equal(categorical[:, -1, :2], [[0.0, 1.0]] * 2)


@pytest.mark.inference(concern="predictive")
def test_scalar_diagnostics_use_projected_means_at_supported_times():
    means = jnp.array([[[jnp.nan, 0.0], [1e-12, 0.0], [4e-12, 0.0]]] * 2)
    batch = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2))},
        trajectory=replace(
            _trajectory(),
            linear_predictors=jnp.full((2, 3, 2), 10.0),
            expected_observations=means,
            observations_mask=jnp.isfinite(means),
        ),
    )
    signal, variance = observation_signal_and_variance(
        compile_model_fixture(
            _scalar_diagnostics_use_projected_means_at_supported_times_model_fixture()
        ),
        batch,
        0,
        np.array([1, 2]),
    )
    np.testing.assert_array_equal(signal, np.asarray(means[:, 1:, 0]))
    np.testing.assert_array_equal(variance, signal)


@pytest.mark.inference(concern="predictive")
def test_undefined_student_moments_produce_an_explicit_diagnostic():
    from nof1_causal_lab.models.ssm.diagnostics.reachability import check_transmission

    paths = _trajectory()
    batch = PredictiveDraws(
        parameters={
            "manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            "obs_df": jnp.array([0.5, 5.0]),
        },
        trajectory=paths,
    )
    signal, variance = observation_signal_and_variance(
        compile_model_fixture(
            _undefined_student_moments_produce_an_explicit_diagnostic_model_fixture()
        ),
        batch,
        0,
        np.arange(3),
    )
    result = check_transmission("heavy_tail", signal, variance)
    assert not result.passed
    assert np.isnan(signal[0]).all()
    assert variance is not None
    assert np.isnan(variance[0]).all()
    assert result.assessment.kind == "not_evaluated"
    assert result.assessment.reason == "NONFINITE_SIGNAL"


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("families", "interval", "expected_variance", "model_fixture_payload"),
    [
        pytest.param(
            ["gaussian"],
            False,
            1.0001e-08,
            _diagnostic_noise_matches_point_and_interval_execution_model_fixture_families0_false_1_0001e_08,
            id="families0-False-1.0001e-08",
        ),
        pytest.param(
            ["gaussian", "poisson"],
            False,
            1e-10,
            _gaussian_poisson_model,
            id="families1-False-1e-10",
        ),
        pytest.param(
            ["gaussian", "poisson"],
            True,
            1.0001e-08,
            _gaussian_poisson_model,
            id="families2-True-1.0001e-08",
        ),
        pytest.param(
            ["student_t"],
            False,
            1.6666666666666669e-10,
            _student_t_model,
            id="families3-False-1.6666666666666669e-10",
        ),
        pytest.param(
            ["student_t"],
            True,
            1.6666666666666666e-12,
            _student_t_model,
            id="families4-True-1.6666666666666666e-12",
        ),
    ],
)
def test_diagnostic_noise_matches_point_and_interval_execution(
    families, interval, expected_variance, model_fixture_payload
):
    from tests.support.inference import make_observation_support_runtime

    channels = len(families)
    spec = model_fixture_payload()
    values = jnp.zeros((2, 3, channels))
    prediction = PredictiveDraws(
        parameters={
            "manifest_cov": jnp.broadcast_to(jnp.eye(channels) * 1e-12, (2, channels, channels)),
            "obs_df": jnp.full(2, 5.0),
        },
        trajectory=PredictiveTrajectory(
            values, values, values, jnp.ones_like(values, dtype=bool), values
        ),
    )
    support = None
    if interval:
        support = make_observation_support_runtime(
            anchor_times=np.arange(3.0),
            manifest_names=numeric.observation_names(compile_model_fixture(spec)),
            support_kinds=("interval",) * channels,
            summary_operators=("mean",) * channels,
            anchor_policies=("support_end",) * channels,
            observation_windows=("2d",) * channels,
            support_start_times=np.tile([[np.nan], [np.nan], [0.0]], (1, channels)),
            support_end_times=np.tile([[np.nan], [np.nan], [2.0]], (1, channels)),
        )
    _, variance = observation_signal_and_variance(
        compile_model_fixture(spec), prediction, 0, np.array([2]), observation_support=support
    )
    assert variance is not None
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-6, atol=0.0)
