"""Descriptive movement agrees with dense reductions without crossing chain boundaries."""

import numpy as np
import pytest
from scipy.fft import dct

from grassmoth.artifacts.probability.joint_layout import JointLawLayout
from grassmoth.artifacts.reports.scorecard import MovementWindow
from grassmoth.models.ssm.inference.diagnostics.scorecard import (
    _refresh_ratio,
    frequency_refresh,
    regional_movement,
)
from grassmoth_test_support.model import x_y_model

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("frequency", [False, True])
def test_blocked_refresh_matches_dense_pooled_variance_and_adjacent_jumps(frequency):
    paths = np.random.default_rng(0).normal(size=(3, 277, 19)).astype(np.float32)
    paths[1] += 10
    values = (
        dct(paths.astype(np.float64), type=2, norm="ortho", axis=-1)
        if frequency
        else paths.astype(np.float64)
    )
    expected = np.mean(np.diff(values, axis=1) ** 2, axis=(0, 1)) / (
        2 * np.var(values, axis=(0, 1))
    )
    actual = _refresh_ratio(paths, frequency=frequency)
    assert actual is not None
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_frozen_chains_do_not_gain_refresh_from_other_chains_or_zero_variance():
    paths = np.zeros((2, 257, 4))
    paths[1, :, :2] = 5
    np.testing.assert_array_equal(_refresh_ratio(paths, frequency=False), np.zeros(4))
    assert _refresh_ratio(paths[:, :1], frequency=True) is None


@pytest.mark.parametrize("steps", [9, 2049, 4097])
def test_frequency_bands_cover_short_standard_and_refined_grids(steps):
    model = x_y_model()
    identity = model.constructs[0].id
    layout = JointLawLayout(
        parameters=(), constructs=(identity,), time_points=tuple(range(steps)), labels={}
    )
    (report,) = frequency_refresh(
        np.zeros((4, steps)), layout, {identity: model.get_construct(identity).name}, chains=2
    )
    assert report.subject.id == identity
    assert report.bands[0].start_mode == 0
    assert report.bands[-1].stop_mode == steps
    assert sum(band.stop_mode - band.start_mode for band in report.bands) == steps
    assert all(band.median_refresh == 0 for band in report.bands)


def test_regional_movement_respects_windows_free_mask_and_worst_time_order():
    model = x_y_model()
    identity = model.constructs[0].id
    layout = JointLawLayout(
        parameters=(), constructs=(identity,), time_points=(0, 1, 2, 3, 4), labels={}
    )
    paths = np.zeros((2, 4, 5))
    paths[1, :, 1] = 1
    paths[:, :, 2] = np.arange(4)
    paths[:, :, 3] = (0, 1, 0, 1)
    (report,) = regional_movement(
        paths.reshape(8, 5),
        layout,
        {identity: model.get_construct(identity).name},
        2,
        free_mask=np.array([[False], [True], [True], [True], [False]]),
        final_delta=np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]),
        windows=(
            MovementWindow(name="early", start_day=0, end_day=3),
            MovementWindow(name="empty", start_day=10, end_day=20),
        ),
        observed_rows=np.array([True, True, False, True, False]),
    )
    assert [point.time_index for point in report.worst_times] == [1, 2, 3]
    early, empty = report.regions
    assert early.coordinates == 2
    assert early.refresh_quantiles is not None
    assert early.delta_quantiles is not None
    assert early.refresh_quantiles.model_dump() == {"minimum": 0, "median": 0.2, "maximum": 0.4}
    assert early.delta_quantiles.model_dump() == {"minimum": 1, "median": 6.5, "maximum": 12}
    assert empty.coordinates == 0
    assert empty.refresh_quantiles is None
    assert empty.delta_quantiles is None

    covered, uncovered = report.reading_coverage
    assert covered.has_readings
    assert not uncovered.has_readings
    assert (covered.coordinates, covered.stretches) == (2, 2)
    assert (uncovered.coordinates, uncovered.stretches) == (1, 1)
    assert uncovered.refresh_quantiles is not None
    assert uncovered.refresh_quantiles.median == 0.4
