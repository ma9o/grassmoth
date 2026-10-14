"""Predictive runtime and diagnostic-report integration on a small mixed model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import PosteriorPredictiveChecks
from nof1_causal_lab.models.diagnostics.history_comparison import measure_predictive_checks
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    predictive_keys,
    simulate_predictive_draws,
)
from tests.support.compute import COMPUTE_BUDGET
from tests.support.inference import compile_model_fixture
from tests.support.model import load_model_fixture


def _predictive_draws_feed_mixed_family_diagnostics_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "posterior_predictive_integration/predictive_draws_feed_mixed_family_diagnostics_model_fixture.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.inference(concern="predictive")


def test_predictive_draws_feed_mixed_family_diagnostics():
    # Exhaustive family/link domains are checked by the observation-sampling
    # test. This case checks the complete prior/runtime/report connection.
    dynamical_model_spec = _predictive_draws_feed_mixed_family_diagnostics_model_fixture()
    times = jnp.array([0.0, 0.1, 0.25, 0.4, 0.7, 1.0], dtype=jnp.float32)
    compiled = compile_model_fixture(dynamical_model_spec)
    parameters = sample_model_laws(compiled, draws=3, key=predictive_keys(7).parameters).parameters
    samples = simulate_predictive_draws(
        compiled,
        parameters,
        times,
        seed=7,
        compute_budget=COMPUTE_BUDGET,
    )
    assert samples.trajectory.latents.shape == (3, 6, 1)
    assert samples.trajectory.observations.shape == (3, 6, 2)
    assert samples.trajectory.observations_mask.shape == (3, 6, 2)
    assert bool(samples.trajectory.observations_mask.all())
    assert all(bool(jnp.isfinite(value).all()) for value in jax.tree.leaves(samples))
    counts = samples.trajectory.observations[..., 1]
    assert bool((counts >= 0).all())
    np.testing.assert_array_equal(counts, jnp.floor(counts))

    # Known observations avoid a second simulation just to construct test data.
    observations = jnp.array(
        [[0.2, 1.0], [jnp.nan, 2.0], [-0.1, 0.0], [0.1, 3.0], [-0.2, 2.0], [0.3, 4.0]]
    )
    indicator_ids = [IndicatorId("indicator:signal"), IndicatorId("indicator:count")]
    result = measure_predictive_checks(
        samples.trajectory.observations,
        observations,
        indicator_ids,
        times=tuple(float(t) for t in times),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        standardized=(False, False),
    )

    assert isinstance(result, PosteriorPredictiveChecks)
    assert result.n_subsample == 3
    assert [overlay.indicator_id for overlay in result.overlays] == indicator_ids
    assert len(result.test_stats) == 8
    assert {(warning.subject.target.id, warning.code) for warning in result.findings} == {
        (indicator_id, check)
        for indicator_id in indicator_ids
        for check in ("calibration", "autocorrelation", "variance")
    }
