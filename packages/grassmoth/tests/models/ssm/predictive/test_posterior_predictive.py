"""Tests for posterior predictive checks (PPCs)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.model.likelihood import OrderedLogisticLawSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.diagnostics.history_comparison import (
    _check_calibration,
    _check_residual_autocorrelation,
    _check_variance_ratio,
    _compute_overlays,
    _compute_test_stats,
)
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.predictive.observations import PredictiveObservationMeanOverflow
from tests.support.compute import COMPUTE_BUDGET
from tests.support.inference import compile_model_fixture, make_observation_support_runtime
from tests.support.model import load_model_fixture
from tests.support.predictive import sample_observation_fixture
from tests.support.ssm import complex_mixed_family_config


def _posterior_runtime_assembles_ordered_cutpoints_from_sample_sites_model_fixture() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "posterior_predictive/testforwardsimulation_test_posterior_runtime_assembles_ordered_cutpoints_from_sample_sites_model_fixture.json"
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime


def _make_lp_and_samples(
    n_draws: int,
    n_timepoints: int,
    n_manifest: int,
    *,
    obs_sd: float = 0.5,
    lp: float = 0.0,
    **extra: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Build observation linear predictors + the minimal ``samples`` the exact
    observation sampler needs (``manifest_cov`` + any emission extra-params).

    The latent simulation is exercised separately (prior/posterior predictive via
    the Diffrax vector field); these tests pin only the emission-family sampling
    given precomputed linear predictors, so no drift/spec is required.
    """
    linear_predictors = jnp.full((n_draws, n_timepoints, n_manifest), lp, dtype=jnp.float32)
    manifest_cov = jnp.broadcast_to(
        jnp.eye(n_manifest, dtype=jnp.float32) * (obs_sd**2),
        (n_draws, n_manifest, n_manifest),
    )
    samples: dict[str, jnp.ndarray] = {"manifest_cov": manifest_cov, **extra}
    return linear_predictors, samples


class TestForwardSimulation:
    """Tests for shared predictive observation simulation."""

    @staticmethod
    def _window_average_support() -> ObservationSupportRuntime:
        nan = np.nan
        return make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            manifest_names=("y",),
            support_kinds=("interval",),
            summary_operators=("mean",),
            anchor_policies=("support_end",),
            observation_windows=("2d",),
            support_start_times=np.array([[nan], [nan], [0.0]], dtype=np.float32),
            support_end_times=np.array([[nan], [nan], [2.0]], dtype=np.float32),
        )

    @pytest.mark.contract
    def test_unknown_family_rejected_at_parse_boundary(self):
        """Serialized distribution names are validated before predictive dispatch."""
        with pytest.raises(ValueError, match="is not a valid DistributionFamily"):
            DistributionFamily("nonexistent_distribution")

    @pytest.mark.inference(concern="predictive")
    def test_mixed_families_preserve_means_and_sample_domains(self):
        families, links, levels, _ = complex_mixed_family_config()
        families += ["bernoulli", "gamma"]
        links += ["probit", "inverse"]
        levels += [0, 0]
        predictors, samples = _make_lp_and_samples(
            2,
            3,
            12,
            obs_df=jnp.array(6.0),
            obs_shape=jnp.array(3.0),
            obs_r=jnp.array(8.0),
            obs_concentration=jnp.array(14.0),
            obs_ordered_cutpoints=jnp.broadcast_to(jnp.array([-1.0, 0.0, 1.0]), (2, 12, 3)),
            obs_cat_intercepts=jnp.zeros((2, 12, 3)),
            obs_cat_slopes=jnp.zeros((2, 12, 3)),
        )
        # Equal nonzero predictors distinguish probit from logit. The final
        # inverse-link predictor is invalid and must remain NaN through sampling.
        predictors = predictors.at[..., 1].set(1.0).at[..., 10].set(1.0)
        predictors = predictors.at[..., 11].set(2.0).at[:, -1, 11].set(-1.0)
        draws, mask, expected = sample_observation_fixture(
            predictors,
            samples,
            jnp.arange(3, dtype=jnp.float32),
            manifest_dists=families,
            manifest_links=links,
            manifest_level_counts=levels,
            n_subsample=2,
        )

        assert draws.shape == (2, 3, 12)
        assert bool(mask.all())
        means = np.broadcast_to(
            [0.0, 0.731058579, 1.0, 0.0, 1.0, 0.5, 1.5, 1.5, 1.0, 0.0, 0.841344746, 0.5],
            (2, 3, 12),
        ).copy()
        means[:, -1, 11] = np.nan
        np.testing.assert_allclose(expected, means, atol=1e-6, equal_nan=True)
        np.testing.assert_array_equal(jnp.isfinite(draws), np.isfinite(means))
        assert bool(jnp.isnan(draws[:, -1, 11]).all())
        assert bool((draws[:, :-1, 11] > 0).all())
        binary = draws[..., [1, 10]]
        assert bool(((binary == 0) | (binary == 1)).all())
        counts = draws[..., [2, 8]]
        assert bool(((counts >= 0) & (counts == jnp.floor(counts))).all())
        assert bool((draws[..., 4] > 0).all())
        assert bool(((draws[..., 5] >= 0) & (draws[..., 5] <= 1)).all())
        categories = draws[..., [6, 7]]
        assert bool(((categories >= 0) & (categories <= 3)).all())
        np.testing.assert_array_equal(categories, jnp.floor(categories))

    @pytest.mark.inference(concern="predictive")
    def test_forward_simulate_support_aware_window_average_respects_emission_schedule(self):
        """Interval-summary PPC emits only on anchor rows and uses aggregated means."""
        # Latent held at 1.0, so the observation linear predictor is 1.0 every row.
        lp = jnp.ones((1, 3, 1), dtype=jnp.float32)
        samples = {"manifest_cov": jnp.array([[[0.0]]], dtype=jnp.float32)}
        times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        obs_mask = jnp.array([[False], [False], [True]])

        y_sim, _, expected = sample_observation_fixture(
            lp,
            samples,
            times,
            manifest_dists=["gaussian"],
            observation_support=self._window_average_support(),
            observation_mask=obs_mask,
            n_subsample=1,
            rng_key=random.PRNGKey(0),
        )

        assert y_sim.shape == (1, 3, 1)
        assert jnp.isnan(y_sim[0, 0, 0])
        assert jnp.isnan(y_sim[0, 1, 0])
        assert abs(float(y_sim[0, 2, 0]) - 1.0) < 0.05
        assert jnp.isnan(expected[0, 0, 0])
        assert jnp.isnan(expected[0, 1, 0])
        assert float(expected[0, 2, 0]) == pytest.approx(1.0)

    @pytest.mark.contract
    def test_forward_simulate_raises_on_log_link_mean_overflow(self):
        """Overflowing log-link means fail before observation sampling."""
        lp, samples = _make_lp_and_samples(
            1, 3, 1, lp=1000.0, obs_shape=jnp.array(2.0, dtype=jnp.float32)
        )
        times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

        with pytest.raises(PredictiveObservationMeanOverflow, match="log-link mean overflow"):
            sample_observation_fixture(
                lp,
                samples,
                times,
                manifest_dists=["gamma"],
                manifest_names=["monthly_eveningness_activity_timing"],
                n_subsample=1,
                rng_key=random.PRNGKey(0),
            )

    @pytest.mark.contract
    def test_posterior_runtime_assembles_ordered_cutpoints_from_sample_sites(self, monkeypatch):
        """Posterior PPC derives cutpoints from sampled threshold bases and gaps."""
        from grassmoth.models.ssm.parameterization import (
            assemble_deterministics_from_registry,
            build_site_registry,
        )
        from grassmoth.models.ssm.predictive import registry_runtime

        dynamical_model_spec = (
            _posterior_runtime_assembles_ordered_cutpoints_from_sample_sites_model_fixture()
        )
        n_draws = 2
        ordered_base = jnp.zeros((n_draws, 2), dtype=jnp.float32)
        ordered_base = ordered_base.at[:, 1].set(-1.0)
        samples = {
            **{
                site.name: jnp.full((n_draws, *site.shape), 0.5)
                for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
            },
            "obs_ordered_base": ordered_base,
            "obs_ordered_gaps": jnp.ones((n_draws, 2, 1), dtype=jnp.float32),
        }
        samples.update(
            assemble_deterministics_from_registry(
                samples, compile_model_fixture(dynamical_model_spec)
            )
        )
        captured = {}

        def _fake_latents(_spec, _samples, times, **_kwargs):
            return (
                jnp.zeros(
                    (
                        n_draws,
                        times.shape[0],
                        numeric.n_states(compile_model_fixture(dynamical_model_spec)),
                    )
                ),
                jnp.zeros(
                    (
                        n_draws,
                        times.shape[0],
                        numeric.n_observations(compile_model_fixture(dynamical_model_spec)),
                    )
                ),
            )

        def _fake_observations(models, linear_predictors, *_args, **_kwargs):

            bound = models.observation_model.laws[1]
            assert isinstance(bound, OrderedLogisticLawSpec)
            captured["cutpoints"] = eqx.filter_vmap(
                lambda operand: operand.evaluate(jnp.zeros(()), jnp.ones(()))
            )(bound.cutpoints)
            shape = linear_predictors.shape
            return jnp.zeros(shape), jnp.ones(shape, dtype=bool), jnp.zeros(shape)

        monkeypatch.setattr(
            registry_runtime,
            "_simulate_vector_field_predictive_latents",
            _fake_latents,
        )
        monkeypatch.setattr(
            registry_runtime,
            "sample_model_observations",
            _fake_observations,
        )

        registry_runtime.simulate_predictive_draws(
            compile_model_fixture(dynamical_model_spec),
            samples,
            jnp.arange(3, dtype=jnp.float32),
            compute_budget=COMPUTE_BUDGET,
        )

        np.testing.assert_allclose(
            captured["cutpoints"],
            np.array([[-1.0, 0.0], [-1.0, 0.0]]),
        )


@pytest.mark.inference(concern="predictive")
class TestDiagnosticChecks:
    """Known arrays separate diagnostic decisions from observation simulation."""

    @pytest.mark.parametrize(
        ("thresholds", "passed"),
        [
            ({}, [False, True, False]),
            ({"low_threshold": 0.5, "high_threshold": 0.9}, [True, True, False]),
        ],
        ids=["defaults", "inclusive-boundaries"],
    )
    def test_calibration_reports_exact_coverage_and_ignores_missing_rows(self, thresholds, passed):
        names = [
            IndicatorId("indicator:under"),
            IndicatorId("indicator:calibrated"),
            IndicatorId("indicator:over"),
            IndicatorId("indicator:short"),
            IndicatorId("indicator:missing"),
        ]
        y_sim = jnp.broadcast_to(jnp.array([-1.0, 1.0])[:, None, None], (2, 10, 5))
        observations = np.zeros((10, 5))
        observations[5:, 0] = 2.0
        observations[[0, 5], 0] = np.nan  # Four of eight observed values are covered.
        observations[-1, 1] = 2.0
        observations[1:, 3] = np.nan
        observations[:, 4] = np.nan

        warnings = _check_calibration(y_sim, jnp.asarray(observations), names, **thresholds)

        assert [warning.subject.target.id for warning in warnings] == names
        unevaluated = [warning for warning in warnings if warning.kind == "not_evaluated"]
        assert [warning.subject.target.id for warning in unevaluated] == names[3:]
        warnings = [warning for warning in warnings if warning.kind == "evaluated"]
        assert [warning.subject.target.id for warning in warnings] == names[:3]
        assert [warning.code for warning in warnings] == ["calibration"] * 3
        np.testing.assert_allclose(
            [warning.evidence.value for warning in warnings], [0.5, 0.9, 1.0]
        )
        assert [(warning.outcome == "passed") for warning in warnings] == passed

    def test_autocorrelation_uses_residuals_and_detects_both_signs(self):
        names: list[IndicatorId] = [
            IndicatorId(f"indicator:{name}")
            for name in ("trend", "alternating", "uncorrelated", "constant", "short", "missing")
        ]
        mean = np.broadcast_to(np.arange(10)[:, None] * 2.0, (10, 6))
        y_sim = jnp.asarray(np.stack([mean - 0.5, mean + 0.5]))
        observations = mean.copy()
        observations[1:9, 0] += np.arange(8)
        observations[1:9, 1] += [-1, 1] * 4
        observations[1:9, 2] += [1, -1, -1, 1] * 2
        observations[[0, 9], :] = np.nan
        observations[5:, 4] = np.nan
        observations[:, 5] = np.nan

        warnings = _check_residual_autocorrelation(y_sim, jnp.asarray(observations), names)

        assert [warning.subject.target.id for warning in warnings] == names
        unevaluated = [warning for warning in warnings if warning.kind == "not_evaluated"]
        assert [warning.subject.target.id for warning in unevaluated] == names[3:]
        warnings = [warning for warning in warnings if warning.kind == "evaluated"]
        assert [warning.subject.target.id for warning in warnings] == names[:3]
        assert [warning.code for warning in warnings] == ["autocorrelation"] * 3
        np.testing.assert_allclose(
            [warning.evidence.value for warning in warnings], [5 / 7, -1, -1 / 7], atol=1e-6
        )
        assert [(warning.outcome == "passed") for warning in warnings] == [False, False, True]

    def test_variance_ratio_uses_temporal_variation_within_each_draw(self):
        names: list[IndicatorId] = [
            IndicatorId(f"indicator:{name}")
            for name in ("low", "matched", "high", "short", "constant")
        ]
        base = np.asarray([-1.0, 1.0] * 4)[:, None]
        temporal = base * np.asarray([0.1, 1.0, 10.0, 1.0, 1.0])
        # Between-draw level shifts must not count as temporal variation.
        y_sim = jnp.asarray(np.stack([temporal - 10.0, temporal + 10.0]))
        observations = np.broadcast_to(base, (8, 5)).copy()
        observations[:2, 0] = np.nan
        observations[2:, 3] = np.nan
        observations[:, 4] = 0.0

        warnings = _check_variance_ratio(y_sim, jnp.asarray(observations), names)

        assert [warning.subject.target.id for warning in warnings] == names
        unevaluated = [warning for warning in warnings if warning.kind == "not_evaluated"]
        assert [warning.subject.target.id for warning in unevaluated] == names[3:]
        warnings = [warning for warning in warnings if warning.kind == "evaluated"]
        assert [warning.subject.target.id for warning in warnings] == names[:3]
        assert [warning.code for warning in warnings] == ["variance"] * 3
        np.testing.assert_allclose(
            [warning.evidence.value for warning in warnings], [0.1, 1.0, 10.0], atol=1e-6
        )
        assert [(warning.outcome == "passed") for warning in warnings] == [False, True, False]


@pytest.mark.inference(concern="predictive")
def test_overlays_preserve_quantiles_observations_and_all_trajectories():
    # Draws are deliberately unordered, with different scales across time and
    # variables. Quantiles below are hand-computed linear interpolations.
    draws = jnp.array(
        [
            [[0.0, 30.0], [10.0, 0.0], [4.0, 20.0]],
            [[4.0, 10.0], [12.0, 8.0], [0.0, 24.0]],
            [[8.0, 20.0], [14.0, 4.0], [8.0, 28.0]],
            [[12.0, 40.0], [16.0, 12.0], [12.0, 32.0]],
        ]
    )
    observations = jnp.array([[2.0, 25.0], [jnp.nan, -1.0], [9.0, 27.0]])
    ids = [IndicatorId("indicator:z"), IndicatorId("indicator:a")]
    result = _compute_overlays(
        draws,
        observations,
        ids,
        times=(0.0, 0.25, 9.0),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        standardized=(True, False),
    )

    assert [overlay.indicator_id for overlay in result] == ids
    assert [overlay.observed for overlay in result] == [(2.0, None, 9.0), (25.0, -1.0, 27.0)]
    expected_medians = [[6.0, 13.0, 6.0], [25.0, 6.0, 26.0]]
    for column, overlay in enumerate(result):
        np.testing.assert_allclose(
            np.asarray(overlay.median, dtype=float), expected_medians[column], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(overlay.spaghetti_draws, draws[:, :, column])


@pytest.mark.inference(concern="predictive")
def test_single_draw_has_exact_median_and_one_trajectory():
    draws = jnp.array([[[2.0, -1.0], [4.0, 8.0]]])
    result = _compute_overlays(
        draws,
        jnp.zeros((2, 2)),
        [IndicatorId("indicator:x"), IndicatorId("indicator:y")],
        times=(0.0, 1.0),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        standardized=(False, False),
    )

    assert len(result) == 2
    for column, overlay in enumerate(result):
        np.testing.assert_array_equal(overlay.median, draws[0, :, column])
        np.testing.assert_array_equal(overlay.spaghetti_draws, draws[:, :, column])


@pytest.mark.inference(concern="predictive")
def test_test_stats_match_masked_observations_and_each_replicate():
    # Huge values at missing observation rows must be ignored in each replicate.
    # The final variable has only two observed rows and must be omitted.
    observations = jnp.array(
        [[1.0, -4.0, 8.0], [3.0, jnp.nan, jnp.nan], [jnp.nan, 0.0, jnp.nan], [5.0, 4.0, 10.0]]
    )
    draws = jnp.array(
        [
            [[2.0, -2.0, 10.0], [4.0, 999.0, 11.0], [999.0, 2.0, 12.0], [6.0, 6.0, 13.0]],
            [[-1.0, 2.0, 20.0], [1.0, -999.0, 21.0], [-999.0, 0.0, 22.0], [3.0, -2.0, 23.0]],
        ]
    )
    result = _compute_test_stats(
        draws,
        observations,
        [IndicatorId("indicator:x"), IndicatorId("indicator:y"), IndicatorId("indicator:z")],
    )
    small_sd, large_sd = np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)
    # Each entry gives observed value, one statistic per draw, and P(rep >= obs).
    expected = {
        (IndicatorId("indicator:x"), "mean"): (3.0, [4.0, 1.0], 0.5),
        (IndicatorId("indicator:x"), "sd"): (small_sd, [small_sd, small_sd], 1.0),
        (IndicatorId("indicator:x"), "min"): (1.0, [2.0, -1.0], 0.5),
        (IndicatorId("indicator:x"), "max"): (5.0, [6.0, 3.0], 0.5),
        (IndicatorId("indicator:y"), "mean"): (0.0, [2.0, 0.0], 1.0),
        (IndicatorId("indicator:y"), "sd"): (large_sd, [large_sd, small_sd], 0.5),
        (IndicatorId("indicator:y"), "min"): (-4.0, [-2.0, -2.0], 1.0),
        (IndicatorId("indicator:y"), "max"): (4.0, [6.0, 2.0], 0.5),
    }
    assert len(result) == len(expected)
    assert {(stat.indicator_id, stat.stat_name) for stat in result} == set(expected)
    for stat in result:
        observed, replicas, p_value = expected[stat.indicator_id, stat.stat_name]
        np.testing.assert_allclose(stat.observed_value, observed, rtol=1e-6)
        np.testing.assert_allclose(stat.rep_values, replicas, rtol=1e-6)
        assert stat.p_value == pytest.approx(p_value)


@pytest.mark.contract
@pytest.mark.parametrize(
    "compute", [_compute_overlays, _compute_test_stats], ids=["overlays", "stats"]
)
def test_predictive_summaries_require_complete_indicator_axis(compute):
    with pytest.raises(ValueError, match="shorter"):
        compute(
            jnp.ones((2, 3, 2)),
            jnp.ones((3, 2)),
            [IndicatorId("indicator:x")],
            **(
                {
                    "times": (0, 1, 2),
                    "time_origin": datetime(2026, 1, 1, tzinfo=UTC),
                    "standardized": (False, False),
                }
                if compute is _compute_overlays
                else {}
            ),
        )
