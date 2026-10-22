"""Tests for worker indicator formatting and extraction prompts."""

import pytest

from grassmoth.artifacts.identity import IndicatorId
from grassmoth_server.actions.prepare_data.context import MeasurementContext
from grassmoth_server.actions.prepare_data.messages import WorkerMessages, _format_indicators

pytestmark = pytest.mark.contract


def _measurement_structure():
    """Minimal MeasurementStructure for testing."""
    return MeasurementContext.model_validate(
        {
            "source": {"files": ["source.csv"]},
            "model_clock": "1d",
            "indicators": [
                {
                    "observation": {
                        "id": IndicatorId("indicator:6bde869aba53fb51e0f4"),
                        "name": "pss_score",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    "extraction": {
                        "kind": "semantic",
                        "how_to_measure": "Perceived Stress Scale score",
                    },
                },
                {
                    "observation": {
                        "id": IndicatorId("indicator:9866c549bd1c25f0a5d7"),
                        "name": "sleep_hours",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    "extraction": {
                        "kind": "semantic",
                        "how_to_measure": "Self-reported hours of sleep",
                    },
                },
            ],
        }
    )


# =============================================================================
# _format_indicators
# =============================================================================


class TestFormatIndicators:
    def test_basic_formatting(self):
        result = _format_indicators(_measurement_structure())
        assert "pss_score" in result
        assert "sleep_hours" in result
        assert "continuous" in result
        assert "Perceived Stress Scale" in result
        assert "support=interval" in result
        assert "operator=mean" in result
        assert "window=1d" in result

    def test_empty_indicators(self):
        with pytest.raises(ValueError, match="at least 1"):
            MeasurementContext.model_validate(
                {"source": {"files": ["source.csv"]}, "model_clock": "1d", "indicators": []}
            )

    def test_missing_instructions_are_rejected(self):
        with pytest.raises(ValueError, match="how_to_measure"):
            MeasurementContext.model_validate(
                {
                    "source": {"files": ["source.csv"]},
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "observation": {
                                "id": "indicator:x",
                                "name": "x",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "extraction": {"kind": "semantic"},
                        }
                    ],
                }
            )

    def test_indicator_specific_window_overrides_model_clock(self):
        spec = MeasurementContext.model_validate(
            {
                "source": {"files": ["source.csv"]},
                "model_clock": "1d",
                "indicators": [
                    {
                        "observation": {
                            "id": IndicatorId("indicator:fortnightly_pss_score"),
                            "name": "fortnightly_pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "observation_window": "2w",
                        },
                        "extraction": {
                            "kind": "semantic",
                            "how_to_measure": "Average perceived stress over two weeks",
                        },
                    }
                ],
            }
        )

        result = _format_indicators(spec)
        assert "window=2w" in result


# =============================================================================
# WorkerMessages
# =============================================================================


class TestWorkerMessages:
    def _sample_window_text(self):
        return "## Window Start: 2024-01-01\n\n08:00  pss=25, sleep=6.5\n09:00  pss=18, sleep=7.2"

    def test_user_message_contains_context(self):
        wm = WorkerMessages(
            question="Does stress affect sleep?",
            measurement_structure=_measurement_structure(),
            window_text=self._sample_window_text(),
            n_windows=1,
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "stress" in user_msg.lower() or "sleep" in user_msg.lower()
        assert "25" in user_msg
        assert "6.5" in user_msg

    def test_indicators_in_prompt(self):
        wm = WorkerMessages(
            question="test",
            measurement_structure=_measurement_structure(),
            window_text=self._sample_window_text(),
            n_windows=1,
        )
        msgs = wm.extraction_messages()
        user_msg = msgs[1]["content"]
        assert "pss_score" in user_msg
        assert "sleep_hours" in user_msg
