"""Factory fixtures for ConstructSpec/IndicatorSpec schema tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grassmoth.artifacts.model.construct import ConstructSpec, Role, TemporalStatus
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.indicator import IndicatorPolarity, IndicatorSpec
from grassmoth.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from grassmoth.artifacts.observations.semantics import SummaryOperator
from grassmoth.artifacts.observations.spec import (
    AuthoredObservationSpec,
    LabeledMeasurement,
    NumericMeasurement,
)
from grassmoth_test_support.helpers import fixture_entity_id

if TYPE_CHECKING:
    from grassmoth.artifacts.observations.dtypes import MeasurementDtype


@pytest.fixture
def construct_factory():
    """Factory for creating ConstructSpec objects.

    Usage:
        def test_something(construct_factory):
            stress = construct_factory("stress", Role.EXOGENOUS)
            mood = construct_factory("mood", Role.ENDOGENOUS)
    """

    def _make(
        name: str,
        role: Role = Role.ENDOGENOUS,
        temporal_status: TemporalStatus = TemporalStatus.TIME_VARYING,
    ) -> ConstructSpec:
        identity = fixture_entity_id("construct", name)
        return ConstructSpec(
            id=identity,
            name=name,
            description=f"{name} description",
            role=role,
            temporal_status=temporal_status,
            indicators=(
                IndicatorSpec(
                    observation=AuthoredObservationSpec(
                        observation_window=None,
                        id=fixture_entity_id("indicator", name),
                        name=name + "_reading",
                        measurement=NumericMeasurement(dtype="continuous"),
                        aggregation="last",
                    ),
                    construct_polarity="positive",
                    likelihood=LikelihoodSpec(
                        law=DeltaLawSpec(v=state(identity)),
                        reasoning="Given reading",
                    ),
                ),
            )
            if role == Role.EXOGENOUS
            else (),
        )

    return _make


@pytest.fixture
def indicator_factory():
    """Factory for creating IndicatorSpec objects.

    Usage:
        def test_something(indicator_factory):
            ind = indicator_factory("mood_rating")
    """

    def _make(
        name: str,
        dtype: MeasurementDtype = "continuous",
        aggregation: SummaryOperator = SummaryOperator.MEAN,
        construct_polarity: IndicatorPolarity = IndicatorPolarity.POSITIVE,
    ) -> IndicatorSpec:
        return IndicatorSpec(
            observation=AuthoredObservationSpec(
                observation_window=None,
                id=fixture_entity_id("indicator", name),
                name=name,
                measurement=LabeledMeasurement(dtype=dtype, levels=("low", "medium", "high"))
                if dtype in {"ordinal", "categorical"}
                else NumericMeasurement(dtype=dtype),
                aggregation=aggregation,
            ),
            construct_polarity=construct_polarity,
        )

    return _make
