"""Tests for the fit-time observation/prior preflight checks."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.identity import DistributionId, ParameterId
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import (
    CallExpression,
    Expression,
    coefficient,
    state,
)
from grassmoth.artifacts.model.likelihood import NegativeBinomial2LawSpec, NormalLawSpec
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from grassmoth.artifacts.sampling import SamplerSpec
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.preparation.preflight import (
    ObservationPreflightFailure,
    validate_observations_for_fit,
)
from grassmoth_server.compute.compilation_cache import compile_staged, prepare_program
from tests.support.compute import COMPUTE_BUDGET
from tests.support.inference import bind_panel_fixture, compile_fit_fixture
from tests.support.model import (
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    parameter_for,
    parameter_laws,
    without_parameters,
)


def _fixed_manifest_means_are_not_judged__model() -> DynamicalModelSpec:
    return load_model_fixture("fit_preflight/fixed_manifest_means_are_not_judged__model.json")


def _standardized_channel_model() -> DynamicalModelSpec:
    return load_model_fixture("fit_preflight/standardized_channel_model.json")


def _raw_channels_model() -> DynamicalModelSpec:
    _LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_PARAMETER_ID = ParameterId(
        "parameter:4830af3c39eaf88213e6b9046faa495a365138395bd1bdd299dac6047fa22056"
    )
    _LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_DISTRIBUTION_ID = DistributionId(
        "distribution:c22817758b2820e1bbf9445f5265eea6a6550afbd7870dfaa0acaef47553a227"
    )
    dynamical_model_spec = _standardized_channel_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    raw_channel = indicator_named(dynamical_model_spec, "raw_channel")
    raw_channel_likelihood = likelihood_named(dynamical_model_spec, "raw_channel")
    latent_0_raw_channel_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "raw_channel"
    )
    small_channel = indicator_named(dynamical_model_spec, "small_channel")
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_diffusion_diag = parameter_for(
        dynamical_model_spec, SiteKind.DIFFUSION_DIAG, "latent_0"
    )
    latent_0_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_0")
    latent_0_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_0")
    latent_0_small_channel_manifest_means = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_MEANS, "latent_0", "small_channel"
    )
    latent_0_small_channel_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "small_channel"
    )
    raw_channel_revised = raw_channel.revised(
        likelihood=raw_channel_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(
                        _LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_PARAMETER_ID, "observation_intercept"
                    )
                    + (coefficient(1.0, "loading") * state(latent_0.id))
                ),
                scale=coefficient(latent_0_raw_channel_manifest_var_diag.id, "observation_scale"),
            ),
            standardized=False,
        )
    )
    latent_0_revised = latent_0.revised(
        indicators=(
            raw_channel_revised,
            small_channel,
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=(
            latent_0_dynamics_decay,
            latent_0_diffusion_diag,
            latent_0_t0_means,
            latent_0_t0_var_diag,
            ParameterSpec(
                id=_LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_PARAMETER_ID,
                name=_LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_PARAMETER_ID,
                description="Fixture quantity",
                distribution=_LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_DISTRIBUTION_ID,
            ),
            latent_0_raw_channel_manifest_var_diag,
            latent_0_small_channel_manifest_means,
            latent_0_small_channel_manifest_var_diag,
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _LATENT_0_RAW_CHANNEL_MANIFEST_MEANS_DISTRIBUTION_ID: dist.Normal(
                loc=0.0, scale=0.5, validate_args=False
            ),
        },
    )


def _preflight_uses_compiled_authored_location_laws_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = _raw_channels_model()
    latent_0_raw_channel_manifest_means = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_MEANS, "latent_0", "raw_channel"
    )
    latent_0_small_channel_manifest_means = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_MEANS, "latent_0", "small_channel"
    )
    return dynamical_model_spec.with_entities(
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                latent_0_raw_channel_manifest_means.id: dist.Normal(
                    loc=jnp.array(87.0, dtype=jnp.float32),
                    scale=jnp.array(10.0, dtype=jnp.float32),
                    validate_args=True,
                ),
                latent_0_small_channel_manifest_means.id: dist.Normal(
                    loc=jnp.array(0.0, dtype=jnp.float32),
                    scale=jnp.array(2.0, dtype=jnp.float32),
                    validate_args=True,
                ),
            },
        )
    )


def _non_identity_links_are_not_judged__model() -> DynamicalModelSpec:
    _OBS_R_PARAMETER_ID = ParameterId(
        "parameter:e72638eadfbc4808dff2da0d50aa4e15ce849387527175e0b32d8d5ad7f89144"
    )
    _OBS_R_DISTRIBUTION_ID = DistributionId(
        "distribution:d4e849f468b99afbc0e3ffa301d28666f7c4f08729c0c9b26630bd008884d6f3"
    )
    dynamical_model_spec = _raw_channels_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    raw_channel = indicator_named(dynamical_model_spec, "raw_channel")
    raw_channel_likelihood = likelihood_named(dynamical_model_spec, "raw_channel")
    latent_0_raw_channel_manifest_means = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_MEANS, "latent_0", "raw_channel"
    )
    small_channel = indicator_named(dynamical_model_spec, "small_channel")
    latent_0_raw_channel_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "raw_channel"
    )
    raw_channel_revised = raw_channel.revised(
        observation=raw_channel.observation.revised(measurement_dtype="count"),
        likelihood=raw_channel_likelihood.revised(
            law=NegativeBinomial2LawSpec[Expression](
                mean=CallExpression(
                    function="exp",
                    arguments=(
                        (
                            coefficient(
                                latent_0_raw_channel_manifest_means.id, "observation_intercept"
                            )
                            + (coefficient(1.0, "loading") * state(latent_0.id))
                        ),
                    ),
                ),
                concentration=coefficient(_OBS_R_PARAMETER_ID, "dispersion"),
            )
        ),
    )
    latent_0_revised = latent_0.revised(
        indicators=(
            raw_channel_revised,
            small_channel,
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec, latent_0_raw_channel_manifest_var_diag
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=(
            *parameters,
            ParameterSpec(
                id=_OBS_R_PARAMETER_ID,
                name="obs_r",
                description="dispersion for obs_r",
                distribution=_OBS_R_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **distributions,
            _OBS_R_DISTRIBUTION_ID: dist.Gamma(concentration=2.0, rate=0.5, validate_args=False),
        },
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.contract

RNG = np.random.default_rng(7)


def _observations(mean_a, mean_b, n=200):
    return np.column_stack([RNG.normal(mean_a, 1.0, size=n), RNG.normal(mean_b, 1.0, size=n)])


def _panel(inputs, observations):
    return bind_panel_fixture(
        inputs.compiled_dynamical_model,
        observations,
        jnp.arange(len(observations), dtype=jnp.float32),
    )


def test_returns_rejection_for_unreachable_free_manifest_mean():
    fit_inputs = compile_fit_fixture(_raw_channels_model())
    failure = validate_observations_for_fit(
        fit_inputs.prior_runtime_bundle, _panel(fit_inputs, _observations(87.0, 0.1))
    )
    assert isinstance(failure, ObservationPreflightFailure)
    assert "raw_channel" in failure.message


def test_passes_when_free_mean_is_within_prior_reach():
    fit_inputs = compile_fit_fixture(_raw_channels_model())
    assert (
        validate_observations_for_fit(
            fit_inputs.prior_runtime_bundle, _panel(fit_inputs, _observations(0.5, -0.3))
        )
        is None
    )


def test_preflight_uses_compiled_authored_location_laws():

    dynamical_model_spec = (
        _preflight_uses_compiled_authored_location_laws_with_parameter_distributions()
    )
    fit_inputs = compile_fit_fixture(dynamical_model_spec)
    assert (
        validate_observations_for_fit(
            fit_inputs.prior_runtime_bundle, _panel(fit_inputs, _observations(87.0, 0.1))
        )
        is None
    )


def test_fixed_manifest_means_are_not_judged():
    fit_inputs = compile_fit_fixture(_fixed_manifest_means_are_not_judged__model())
    assert (
        validate_observations_for_fit(
            fit_inputs.prior_runtime_bundle, _panel(fit_inputs, _observations(87.0, 0.1))
        )
        is None
    )


def test_binding_standardizes_flagged_channels_before_preflight():
    fit_inputs = compile_fit_fixture(_standardized_channel_model())
    obs = _observations(87.0, 0.1)
    assert (
        validate_observations_for_fit(fit_inputs.prior_runtime_bundle, _panel(fit_inputs, obs))
        is None
    )


def test_non_identity_links_are_not_judged():
    fit_inputs = compile_fit_fixture(_non_identity_links_are_not_judged__model())
    obs = _observations(0.0, 0.1)
    obs[:, 0] = RNG.poisson(80.0, size=obs.shape[0]).astype(np.float64)
    assert (
        validate_observations_for_fit(fit_inputs.prior_runtime_bundle, _panel(fit_inputs, obs))
        is None
    )


def test_nan_only_channels_are_skipped():
    fit_inputs = compile_fit_fixture(_raw_channels_model())
    obs = _observations(0.2, 0.1)
    obs[:, 0] = np.nan
    assert (
        validate_observations_for_fit(fit_inputs.prior_runtime_bundle, _panel(fit_inputs, obs))
        is None
    )


def test_fit_runs_preflight_before_dispatch():
    fit_inputs = compile_fit_fixture(_raw_channels_model())
    obs = _observations(87.0, 0.1)
    failure = fit(
        fit_inputs.prior_runtime_bundle,
        bind_panel_fixture(
            fit_inputs.compiled_dynamical_model,
            jnp.asarray(obs),
            jnp.arange(obs.shape[0], dtype=jnp.float32),
        ),
        sampler=SamplerSpec(),
        clock=time.monotonic,
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_staged,
        prepare_program=prepare_program,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert isinstance(failure, ObservationPreflightFailure)
    assert "raw_channel" in failure.message
