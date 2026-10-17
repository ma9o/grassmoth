"""Literal coordinates stay literal when float32 fits retain different draw counts."""

from unittest.mock import Mock

import numpy as np
import pytest
from evaluation.declarations import Tolerances

from evaluation import comparison

pytestmark = pytest.mark.inference(concern="sampling")


def test_identical_point_laws_compare_equal_across_draw_counts(monkeypatch):
    layout = Mock()
    result, reference = (
        Mock(inference=Mock(core=Mock(inference_metadata=Mock(num_chains=4)))) for _ in range(2)
    )
    values = {
        id(result): np.full((16000, 1), 0.1, dtype=np.float32),
        id(reference): np.full((24000, 1), 0.1, dtype=np.float32),
    }
    monkeypatch.setattr(
        comparison, "_posterior_draws", lambda body, _segment=None: (layout, values[id(body)])
    )
    report = comparison.posterior_equivalence(
        result, reference, Tolerances(mean_sd=0.1, sd_relative=0.1)
    )
    assert report["summary_equivalent"] is True
    assert report["fixed_coordinates_equal"] is True
    scores = comparison._start_scores(values[id(result)][:4], values[id(reference)])
    assert scores["start_rms_reference_sd"] == [None] * 4
    assert scores["start_fixed_coordinates_equal"] == [True] * 4
