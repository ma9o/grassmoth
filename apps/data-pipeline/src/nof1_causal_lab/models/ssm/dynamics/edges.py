"""The contribution protocol implemented by compiled scientific mechanisms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array


@runtime_checkable
class VectorFieldComponent(Protocol):
    """Anything that contributes to ``dη/dt``.

    ``contribute`` returns the updated vector-field value. ``eta`` is
    the raw state; ``eta_per_edge`` is the ``(n_target, n_source)``
    matrix with edge-input overrides already applied (single-target
    edges read one entry; full-vector terms typically use one or the
    other depending on whether they care about per-target source
    overrides).
    """

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        eta_per_edge: Array,
        t: Array,
        params: Mapping[str, Array],
        /,
    ) -> Array:
        """Add this component's drift contribution to the accumulated state derivative.

        Args:
            accumulator: Derivative contributions already accumulated for each state.
            eta: Current state vector.
            eta_per_edge: Source states presented to each target after edge-specific
                intervention handling.
            t: Current model time.
            params: Numerical coefficients bound to this component.

        Returns:
            A new derivative vector with this component's contribution included.
        """
        ...
