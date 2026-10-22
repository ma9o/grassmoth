"""Reported predictive plots contain server-computed statistics, including exact bin totals."""

import jax.numpy as jnp
import pytest

from grassmoth.artifacts.identity import IndicatorId
from grassmoth.models.diagnostics.history_comparison import _compute_test_stats

pytestmark = pytest.mark.inference(concern="predictive")


def test_predictive_stat_p_value_and_histogram_use_replications():
    observed = jnp.array([[0.0], [1.0], [2.0]])
    replications = jnp.array([[[-1.0], [0.0], [1.0]], [[1.0], [2.0], [3.0]]])
    stats = _compute_test_stats(replications, observed, [IndicatorId("indicator:outcome")])
    mean = next(stat for stat in stats if stat.stat_name == "mean")
    assert mean.observed_value == 1.0
    assert mean.rep_values == (0.0, 2.0)
    assert mean.p_value == 0.5
    assert sum(histogram_bin.count for histogram_bin in mean.histogram) == 2
    assert mean.histogram[-1].bin_end == 2.0
