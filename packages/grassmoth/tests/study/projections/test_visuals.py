"""The workbench retains irregular timing, modes and paired draws."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import ConstructId, IndicatorId
from grassmoth.artifacts.observations.spec import ResolvedObservationSpec
from grassmoth.artifacts.reports.checks import NotEvaluated
from grassmoth.study.projections.visuals import observation_history
from grassmoth_test_support.helpers import fixture_entity_id


@pytest.mark.contract
def test_observations_keep_irregular_anchors_support_missingness_and_empirical_mass():
    origin = datetime(2026, 1, 1)
    identity = fixture_entity_id("indicator", "observed")
    table = pl.DataFrame(
        {
            "indicator_id": [identity] * 5,
            "anchor_time": [origin + timedelta(days=t) for t in (0, 0.25, 8, 10, 11)],
            "support_start": [origin + timedelta(days=t) for t in (-1, 0, 7, 9, 10)],
            "support_end": [origin + timedelta(days=t) for t in (0, 0.25, 8, 10, 11)],
            "value": [-4.0, -4.0, None, 5.0, 5.0],
        }
    )
    view = observation_history(origin.replace(tzinfo=UTC), table)
    assert view.times == (0, 0.25, 8, 10, 11)
    assert view.values == (-4, -4, None, 5, 5)
    assert view.support_start == (-1, 0, 7, 9, 10)
    assert [(p.value, p.probability) for p in view.empirical] == [(-4, 0.5), (5, 1)]


@pytest.mark.contract
def test_full_draw_summaries_preserve_modes_and_masked_category_counts():
    from grassmoth.actions.simulate.summaries import simulation_summary

    state_id = ConstructId("construct:state")
    indicator = IndicatorId("indicator:observed")
    action = np.array([[-5.0, -4.0, -5.0], [5.0, 4.0, 5.0], [8.0, 9.0, 8.0]])[:, :, None]
    reference = action - np.array([1.0, 3.0, 7.0])[:, None, None]
    observed = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])[:, :, None]
    mask = np.ones_like(action, dtype=bool)
    mask[:, 1, 0] = False
    from grassmoth.artifacts.simulation import (
        PairedArmSimulation,
        SimulationArm,
        SimulationEvidence,
        SimulationObservationLayout,
        SimulationSpec,
    )

    evidence = SimulationEvidence(
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 0.25, 9),
        draws=3,
        seed=0,
        state_ids=(state_id,),
        parameter_draws={},
        arms=PairedArmSimulation(
            action=SimulationArm(
                latent_paths=NumericalArray.from_numpy(action),
                observations=NumericalArray.from_numpy(observed),
            ),
            reference=SimulationArm(
                latent_paths=NumericalArray.from_numpy(reference),
                observations=NumericalArray.from_numpy(observed),
            ),
            causal=NotEvaluated(
                code="causal_effect",
                subject="causal_effect",
                reason="CAUSAL_EVALUATION_FAILED",
                detail="No certified effect in this fixture.",
            ),
        ),
        observation_layout=SimulationObservationLayout(
            variables=(
                ResolvedObservationSpec(
                    id=indicator,
                    name="Observed",
                    measurement_dtype="binary",
                    aggregation="last",
                    observation_window="1d",
                ),
            ),
            mask=NumericalArray.from_numpy(mask),
            support_start_times=NumericalArray.from_numpy(np.asarray([[0], [0.25], [9]])),
            support_end_times=NumericalArray.from_numpy(np.asarray([[0], [0.25], [9]])),
        ),
        assignments=SimulationSpec(
            start="2026-01-01", horizon="9d", interventions=({"target": state_id, "value": 1},)
        ).assignments(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    summary = simulation_summary(evidence, action, observed, mask, reference, observed)
    assert summary.state_frames[state_id] == pytest.approx((-5.875, 8.375))
    probabilities = summary.action_category_probabilities[indicator]
    assert probabilities.n_draws == (3, 0, 3)
    assert probabilities.probabilities["1"] == (1 / 3, None, 2 / 3)
    assert summary.reference_category_probabilities == summary.action_category_probabilities
    assert summary.indicator_frames[indicator] == (0, 1)
