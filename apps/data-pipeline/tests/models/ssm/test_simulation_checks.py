"""Shared exact simulation reductions, without construct admission."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    evaluate_law,
    safe_native,
    with_response,
)
from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    DesignInfo,
    measure_construct_simulation,
)
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.helpers import fixture_entity_id
from tests.inference_fixtures import compile_model_fixture
from tests.model_fixtures import fixed_hill_model, load_model_fixture
from tests.observation_fixtures import observation_laws


def _time_invariant_construct_omits_temporal_transmission_check_model_fixture() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "simulation_checks/time_invariant_construct_omits_temporal_transmission_check_model_fixture.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.inference(concern="predictive")


@pytest.mark.parametrize(
    ("family", "means", "scale", "extra", "expected"),
    [
        ("delta", [0.0, 2.0], 0.0, {}, [0.0, 0.0]),
        ("gaussian", [0.0, 2.0], 0.5, {}, [0.25, 0.25]),
        (
            "student_t",
            [0.0, 1.0, 2.0],
            1.0,
            {"obs_df": [5.0, 1.5, 0.5]},
            [5.0 / 3.0, np.inf, np.nan],
        ),
        ("poisson", [0.0, 1e-12, 2.0], 1.0, {}, [0.0, 1e-12, 2.0]),
        ("gamma", [1.0, 2.0], 1.0, {"obs_shape": 1e-10}, [1e10, 4e10]),
        ("bernoulli", [0.0, 1e-12, 0.5, 1.0], 1.0, {}, [0.0, 1e-12, 0.25, 0.0]),
        ("negative_binomial", [0.0, 1.0, 2.0], 1.0, {"obs_r": 1e-10}, [0.0, 1e10, 4e10]),
        ("beta", [1e-12, 0.5], 1.0, {"obs_concentration": 9.0}, [1e-13, 0.025]),
    ],
)
def test_conditional_variance_uses_observation_family_moments(
    family, means, scale, extra, expected
):
    bound = observation_laws(
        [DistributionFamily(family)],
        parameters={name: jnp.asarray(value) for name, value in extra.items()},
    )[0]
    mean = jnp.asarray(means)
    baseline = jnp.ones_like(mean) if family == "gamma" else jnp.zeros_like(mean)
    law = with_response(evaluate_law(bound, baseline, jnp.full_like(mean, scale)), mean)
    native, valid = safe_native(law)
    actual = jnp.where(valid, native.variance, jnp.nan)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0, equal_nan=True)


def test_time_invariant_construct_omits_temporal_transmission_check():
    draws = 200
    times = 20
    static_values = np.linspace(-1.5, 1.5, draws)
    latent = np.broadcast_to(static_values[:, None, None], (draws, times, 1))
    expected = latent.copy()
    pred = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.array([[[0.25]]]), (draws, 1, 1))},
        trajectory=PredictiveTrajectory(
            latents=jnp.asarray(latent),
            linear_predictors=jnp.asarray(expected),
            observations=jnp.asarray(expected),
            expected_observations=jnp.asarray(expected),
            observations_mask=jnp.ones(expected.shape, dtype=bool),
        ),
    )
    dynamical_model_spec = (
        _time_invariant_construct_omits_temporal_transmission_check_model_fixture()
    )
    obs_idx = np.arange(times)
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "static_indicator"),),
        t_grid=jnp.arange(times, dtype=float),
        obs_index_by_indicator={fixture_entity_id("indicator", "static_indicator"): obs_idx},
        values_by_indicator={
            fixture_entity_id("indicator", "static_indicator"): np.linspace(-1.0, 1.0, times)
        },
    )
    target = ConstructSimulationTarget(
        construct=compile_model_fixture(dynamical_model_spec).states[0],
    )

    results, _timings = measure_construct_simulation(
        compile_model_fixture(dynamical_model_spec),
        pred,
        design,
        target,
        clock=time.monotonic,
        edge_contrasts=COMPUTE_BUDGET,
    )
    checks = {result.check for result in results}
    assert {"C5a location reach", "C5b width"} <= checks
    transmission = next(
        result for result in results if result.check == "C5c transmission"
    ).assessment
    assert transmission.kind == "not_evaluated"
    assert transmission.reason == "STATIC_CONSTRUCT"


@pytest.mark.parametrize(
    ("dynamics", "measurement", "model_fixture_payload"),
    [
        pytest.param(
            True,
            True,
            fixed_hill_model,
            id="True-True",
        ),
        pytest.param(
            True,
            False,
            fixed_hill_model,
            id="True-False",
        ),
        pytest.param(
            False,
            True,
            fixed_hill_model,
            id="False-True",
        ),
        pytest.param(
            False,
            False,
            fixed_hill_model,
            id="False-False",
        ),
    ],
)
def test_fixed_hill_coefficients_participate_in_checks_and_edge_off(
    monkeypatch, dynamics, measurement, model_fixture_payload
):
    """Fixed coefficients still affect the Hill checks and the exact edge-off contrast."""
    from nof1_causal_lab.artifacts.expressions import LiteralExpression, hill_applications
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    draws, ticks = 4, 21
    spec = model_fixture_payload()
    latents = np.broadcast_to(np.linspace(0.2, 2, ticks)[None, :, None], (draws, ticks, 2)).copy()
    predictive = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.eye(2) * 0.25, (draws, 2, 2))},
        trajectory=PredictiveTrajectory(
            latents=jnp.asarray(latents),
            linear_predictors=jnp.asarray(latents),
            observations=jnp.asarray(latents),
            expected_observations=jnp.asarray(latents),
            observations_mask=jnp.ones(latents.shape, dtype=bool),
        ),
    )
    captured = []
    original = compile_model_fixture(spec).dynamics.spec.components[2]

    def exact_resimulation(intervened_spec, samples, _times, **_kwargs):
        component = _kwargs["dynamics"].components[2]
        assert component.expression == LiteralExpression(value=0)
        assert samples is predictive.parameters
        for name, values in predictive.parameters.items():
            np.testing.assert_array_equal(samples[name], jnp.asarray(values))
        captured.append(intervened_spec)
        return jnp.asarray(latents * 0.8), jnp.asarray(latents[:, :, 1:])

    monkeypatch.setattr(
        registry_runtime, "_simulate_vector_field_predictive_latents", exact_resimulation
    )
    indicator_id = fixture_entity_id("indicator", "y1")
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "x1"), indicator_id),
        t_grid=jnp.arange(ticks, dtype=float),
        obs_index_by_indicator={
            identity: np.arange(ticks)
            for identity in numeric.observation_ids(compile_model_fixture(spec))
        },
        values_by_indicator={
            identity: np.linspace(0.2, 2, ticks)
            for identity in numeric.observation_ids(compile_model_fixture(spec))
        },
        n_draws=draws,
    )
    results, _ = measure_construct_simulation(
        compile_model_fixture(spec),
        predictive,
        design,
        ConstructSimulationTarget(
            construct=compile_model_fixture(spec).states[1],
            edge_parents=("X",),
            hill_parents=("X",),
        ),
        dynamics=dynamics,
        measurement=measurement,
        clock=time.monotonic,
        edge_contrasts=COMPUTE_BUDGET,
    )
    assert len(captured) == int(dynamics)
    assert compile_model_fixture(spec).dynamics.spec.components[2] == original
    assert len(tuple(hill_applications(original.expression))) == 1
    checks = {result.check: result for result in results}
    assert ("C5c transmission" in checks) == measurement
    assert ("C3 resolvability" in checks) == dynamics
    if dynamics:
        assert checks["C3 resolvability"].passed
        assert checks["C4c saturation"].passed
        evidence = checks["C4c saturation"].evidence
        assert evidence is not None
        np.testing.assert_array_equal(evidence["hill_n"], np.full(draws, 2))
