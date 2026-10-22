"""LAPS and particle sampling use the exact joint density.

Guard against reintroducing a linearized Gaussian likelihood backend. This is a
static source scan, with no JAX execution or model build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[3] / "src" / "grassmoth"

# Symbols that construct/return the linearised (EKF/IEKS + Laplace) marginal-
# likelihood backend.
_LAPLACE_BACKEND_SYMBOLS = (
    "make_laplace_backend",
    "build_laplace_backend",
    "LaplaceLikelihood",
    "LocalLinearizationConfig",
    "ExactAffineConfig",
    "linearize_drift",
)


def test_linearized_backends_are_absent_from_production() -> None:
    offenders: dict[str, list[str]] = {}
    for py_path in _SRC.rglob("*.py"):
        rel = py_path.relative_to(_SRC).as_posix()
        text = py_path.read_text(encoding="utf-8")
        hits = [symbol for symbol in _LAPLACE_BACKEND_SYMBOLS if symbol in text]
        if hits:
            offenders[rel] = hits

    assert not offenders, (
        "LAPS initialization and particle sampling must use the exact joint density. "
        "Linearized backend references found in: "
        f"{offenders}"
    )
