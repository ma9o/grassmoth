"""Batched reporting preserves native diagnostics on the same float32 draws."""

import numpy as np
import pytest
from arviz_base import from_dict
from arviz_stats.sampling_diagnostics import ess, mcse, rhat

from grassmoth.models.ssm.inference.diagnostics.chain_stats import chain_statistics

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(("chains", "draws"), [(1, 3), (1, 4), (2, 5), (2, 64), (4, 257)])
def test_batched_diagnostics_match_native_rank_tail_and_mcse(chains, draws):
    rng = np.random.default_rng(16)
    samples = rng.normal(size=(chains, draws, 263)).astype(np.float32)
    samples[:, :, 1::3] = np.cumsum(samples[:, :, 1::3], axis=1)
    samples[:, 1::2, 2::3] = -samples[:, ::2, 2::3][:, : draws // 2]
    samples[:, :, 0] = 4
    samples[:, :, 3] = np.arange(chains, dtype=np.float32)[:, None]
    samples[:, :, 6] = np.round(samples[:, :, 6])
    samples[0, 0, 9] = np.nan
    accepted = rng.random((chains, draws)) < 0.2
    accepted[:, 0] = True
    indices = np.maximum.accumulate(np.where(accepted, np.arange(draws), 0), axis=1)
    samples[:, :, 12] = np.take_along_axis(samples[:, :, 12], indices, axis=1)
    data = from_dict({"posterior": {"state": samples}})
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.stack(
            (
                rhat(data)["state"].values,
                ess(data, method="bulk")["state"].values,
                ess(data, method="tail")["state"].values,
                mcse(data, method="mean")["state"].values,
            ),
            axis=-1,
        )
        actual = chain_statistics(samples)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=5e-5,  # Float32 rank transforms and FFT reductions differ by about 1e-5 at 320,000 draws.
        atol=0,
        equal_nan=True,
    )


def test_tail_quantile_indicators_do_not_round_up_to_an_adjacent_float32_draw():
    # The low quantile falls between adjacent floats and rounds up in float32.
    values = 1 + np.arange(4 * 257, dtype=np.float32) * np.finfo(np.float32).eps
    samples = values.reshape(4, 257, 1)
    # Promote the same stored values so NumPy does not round the scalar quantile.
    reference = from_dict({"posterior": {"state": samples.astype(np.float64)}})
    expected = ess(reference, method="tail")["state"].values
    np.testing.assert_allclose(chain_statistics(samples)[:, 2], expected, rtol=5e-5)
