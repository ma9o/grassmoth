"""Weight diagnostics distinguish support loss, concentration, and invalid laws."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from grassmoth.artifacts.reports.posterior_diagnostics import (
    DsmcWeightDiagnostics,
    ParticleWeightMetric,
)
from grassmoth.models.ssm.inference.samplers.particle.weights import (
    summarize_dsmc_weights,
    weight_statistics,
)


@pytest.mark.inference(concern="sampling")
def test_weight_statistics_match_categorical_laws_and_ignore_additive_offsets():
    logits = np.array(
        [[0, 0, 0, 0], [0, -2, -7, -np.inf], [0, -np.inf, -np.inf, -np.inf]],
        dtype=np.float32,
    )
    statistics = jax.jit(weight_statistics)(jnp.asarray(logits))
    shifted = jax.jit(weight_statistics)(jnp.asarray(logits + 1000))
    assert set(statistics) == set(ParticleWeightMetric)
    for metric, values in statistics.items():
        np.testing.assert_allclose(values, shifted[metric], atol=2e-5, rtol=2e-5)
    for index, row in enumerate(logits):
        scores = np.asarray(row)
        finite = np.isfinite(scores)
        log_probs = scores.astype(np.float64) - logsumexp(scores.astype(np.float64))
        probabilities = np.exp(log_probs)
        ess = 1 / np.square(probabilities).sum()
        expected = {
            ParticleWeightMetric.ESS: ess,
            ParticleWeightMetric.ESS_FRACTION: ess / len(scores),
            ParticleWeightMetric.ENTROPY: -(probabilities[finite] * log_probs[finite]).sum(),
            ParticleWeightMetric.MAX_PROBABILITY: probabilities.max(),
            ParticleWeightMetric.REFERENCE_PROBABILITY: probabilities[0],
            ParticleWeightMetric.FINITE_LOG_WEIGHT_RANGE: np.ptp(scores[finite]),
            ParticleWeightMetric.FINITE_LOG_WEIGHT_VARIANCE: scores[finite].var(),
            ParticleWeightMetric.FINITE_FRACTION: finite.mean(),
        }
        for metric, value in expected.items():
            np.testing.assert_allclose(statistics[metric][index], value, rtol=2e-6, atol=1e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("invalid", [-np.inf, np.inf, np.nan])
def test_invalid_weight_laws_are_reported_instead_of_replaced(invalid):
    statistics = jax.jit(weight_statistics)(jnp.full((2, 4), invalid))
    for metric in (
        ParticleWeightMetric.ESS,
        ParticleWeightMetric.ENTROPY,
        ParticleWeightMetric.MAX_PROBABILITY,
        ParticleWeightMetric.REFERENCE_PROBABILITY,
    ):
        assert np.isnan(statistics[metric]).all()
    np.testing.assert_array_equal(statistics[ParticleWeightMetric.FINITE_FRACTION], 0)


@pytest.mark.contract
def test_weight_payload_preserves_nodes_aggregates_and_nonfinite_evidence():
    # Five leaves, three pairs, two four-step nodes, the root and terminal selection.
    history = {
        metric.history_key: jnp.arange(72, dtype=jnp.float32).reshape(2, 3, 12) / 100
        for metric in ParticleWeightMetric
    }
    ess_key = ParticleWeightMetric.ESS.history_key
    history[ess_key] = history[ess_key].at[0, 0, 0].set(jnp.nan)
    result = summarize_dsmc_weights(history, num_steps=5, num_particles=8, phase="post_warmup")
    assert [(node.level, node.start, node.stop) for node in result.tree] == [
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (1, 0, 2),
        (1, 2, 4),
        (1, 4, 5),
        (2, 0, 4),
        (2, 4, 5),
        (3, 0, 5),
    ]
    assert [node.weights.candidates for node in result.tree] == [8] * 5 + [64] * 6
    assert result.terminal.candidates == 8
    missing = result.tree[0].weights.metrics[ParticleWeightMetric.ESS]
    assert missing.mean is missing.minimum is missing.maximum is None
    assert missing.nonfinite_fraction == pytest.approx(1 / 6)
    terminal = result.terminal.metrics[ParticleWeightMetric.ESS]
    assert terminal.mean == pytest.approx(0.41)
    assert terminal.minimum == pytest.approx(0.11)
    assert terminal.maximum == pytest.approx(0.71)
    assert terminal.nonfinite_fraction == 0
    payload = json.dumps(result.model_dump(mode="json"), allow_nan=False)
    assert DsmcWeightDiagnostics.model_validate_json(payload) == result
