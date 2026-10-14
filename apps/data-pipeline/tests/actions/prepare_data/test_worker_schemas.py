"""Worker output contracts: datatype validation, error collection, and frame conversion."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nof1_causal_lab.actions.prepare_data.context import MeasurementContext
from nof1_causal_lab.actions.prepare_data.worker_output import (
    _check_dtype_match,
    validate_worker_output,
)

pytestmark = pytest.mark.contract


def _measurement_structure(*indicators):
    """Build a minimal MeasurementStructure dict with given indicator tuples (name, dtype)."""
    default_aggregations = {
        "continuous": "mean",
        "binary": "last",
        "count": "count",
        "ordinal": "last",
        "categorical": "last",
    }
    return MeasurementContext.model_validate(
        {
            "source": {"files": ["source.csv"]},
            "model_clock": "1d",
            "indicators": [
                {
                    "observation": {
                        "name": name,
                        "id": "indicator:" + name,
                        "measurement_dtype": dtype,
                        "aggregation": default_aggregations[dtype],
                        **(
                            {"ordinal_levels": ["low", "medium", "high"]}
                            if dtype == "ordinal"
                            else {}
                        ),
                        **(
                            {"categorical_levels": ["walking", "running"]}
                            if dtype == "categorical"
                            else {}
                        ),
                    },
                    "extraction": {"kind": "semantic", "how_to_measure": "Measure " + name},
                }
                for name, dtype in indicators
            ],
        }
    )


@pytest.mark.parametrize(
    ("dtype", "accepted", "rejected"),
    [
        ("continuous", [None, 42, 3.14], ["hello"]),
        (
            "binary",
            [None, True, False, 0, 1, "0", "1", "true", "false", "True", "False"],
            ["maybe", 2],
        ),
        ("count", [None, 5, 0, 3.0], [-1, 3.5]),
        ("ordinal", [None, 3, 2.0], ["moderate", True, 1.5]),
        ("categorical", [None, "category_a"], [42]),
    ],
)
def test_dtype_acceptance_boundary(dtype, accepted, rejected):
    for value in accepted:
        assert _check_dtype_match(value, dtype), (dtype, value)
    for value in rejected:
        assert not _check_dtype_match(value, dtype), (dtype, value)


@pytest.mark.parametrize("payload", [{}, {"extractions": []}])
def test_empty_output_keeps_typed_frame_columns(payload):
    output, errors = validate_worker_output(payload, _measurement_structure(("mood", "continuous")))
    assert errors == []
    assert output is not None
    assert_frame_equal(
        output.to_dataframe(),
        pl.DataFrame(
            schema={"indicator_id": pl.String, "value": pl.String, "timestamp": pl.String}
        ),
    )


def test_mixed_output_preserves_identity_windows_missingness_and_normalized_values():
    spec = _measurement_structure(
        ("mood", "continuous"),
        ("smoking", "binary"),
        ("steps", "count"),
        ("severity", "ordinal"),
        ("activity", "categorical"),
    )
    rows = [
        ("mood", "2024-01-01", 7.5),
        ("mood", "2024-01-02", None),
        ("smoking", "2024-01-01", True),
        ("smoking", "2024-01-02", False),
        ("steps", "2024-01-01", 3),
        ("steps", "2024-01-02", None),
        ("severity", "2024-01-01", 2.0),
        ("severity", "2024-01-02", None),
        ("activity", "2024-01-01", " Walking "),
        ("activity", "2024-01-02", "RUNNING"),
    ]
    payload = {
        "extractions": [
            {"indicator_id": f"indicator:{name}", "window_start": window, "value": value}
            for name, window, value in rows
        ]
    }
    output, errors = validate_worker_output(
        payload, spec, expected_window_starts=["2024-01-01", "2024-01-02"]
    )
    assert errors == []
    assert output is not None
    assert_frame_equal(
        output.to_dataframe(),
        pl.DataFrame(
            {
                "indicator_id": [f"indicator:{name}" for name, _, _ in rows],
                "value": [
                    "7.5",
                    None,
                    "True",
                    "False",
                    "3",
                    None,
                    "2",
                    None,
                    " Walking ",
                    "RUNNING",
                ],
                "timestamp": [window for _, window, _ in rows],
            }
        ),
    )


class TestValidateWorkerOutput:
    def test_not_dict_returns_error(self):
        spec = _measurement_structure(("mood", "continuous"))
        output, errors = validate_worker_output("not a dict", spec)
        assert output is None
        assert len(errors) == 1
        assert "dictionary" in errors[0].lower()

    def test_extractions_not_list(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {"extractions": "bad"}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("list" in e.lower() for e in errors)

    def test_extraction_not_dict(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {"extractions": [42]}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_unknown_indicator_error(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {
            "extractions": [
                {
                    "window_start": "2024-01-01",
                    "indicator_id": "indicator:nonexistent",
                    "value": 1.0,
                }
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("nonexistent" in e for e in errors)
        assert any("mood" in e for e in errors)  # suggests valid indicators

    def test_dtype_mismatch_error(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {
            "extractions": [
                {
                    "window_start": "2024-01-01",
                    "indicator_id": "indicator:mood",
                    "value": "not_a_number",
                }
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("dtype" in e for e in errors)

    def test_multiple_errors_collected(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator_id": "indicator:bad1", "value": 1.0},
                {"window_start": "2024-01-01", "indicator_id": "indicator:bad2", "value": 2.0},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert len(errors) >= 2

    def test_structural_errors_include_missing_expected_pairs(self):
        output, errors = validate_worker_output(
            {
                "extractions": [
                    {"window_start": [], "indicator_id": "indicator:mood", "value": 1.0},
                    {"window_start": "2024-01-01", "indicator_id": {}, "value": 2.0},
                ]
            },
            _measurement_structure(("mood", "continuous")),
            expected_window_starts=["2024-01-01"],
        )
        assert output is None
        assert len(errors) == 3
        assert "extractions[0]" in errors[0]
        assert "extractions[1]" in errors[1]
        assert errors[2] == (
            "Missing extractions (use null for unavailable values): "
            "[('2024-01-01', 'indicator:mood')]"
        )

    def test_duplicate_window_start_indicator_rejected(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator_id": "indicator:mood", "value": 3.0},
                {"window_start": "2024-01-01", "indicator_id": "indicator:mood", "value": 4.0},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_unexpected_window_start_rejected_when_expected_window_starts_given(self):
        spec = _measurement_structure(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-99", "indicator_id": "indicator:mood", "value": 3.0},
            ]
        }
        output, errors = validate_worker_output(
            data, spec, expected_window_starts=["2024-01-01", "2024-01-02"]
        )
        assert output is None
        assert any("not in expected support windows" in e for e in errors)

    def test_ordinal_requires_numeric_code(self):
        spec = _measurement_structure(("severity", "ordinal"))
        data = {
            "extractions": [
                {
                    "window_start": "2024-01-01",
                    "indicator_id": "indicator:severity",
                    "value": "high",
                }
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("expected dtype 'ordinal'" in e for e in errors)

    def test_ordinal_code_must_be_in_range(self):
        spec = _measurement_structure(("severity", "ordinal"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator_id": "indicator:severity", "value": 3}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("must be in 0..2" in e for e in errors)
