"""Pure derivations of numerical support from scientific structure and policies."""

from __future__ import annotations

from nof1_causal_lab.compilation_errors import AggregatedCompileError


class NumericalSupportError(AggregatedCompileError):
    """Aggregate failures to derive numerical support from DynamicalModelSpec."""

    header = "Numerical derivation failed"
