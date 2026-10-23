"""Shared inference helpers that do not own concrete backend implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Set

    from numpyro.primitives import Message


from typing import TYPE_CHECKING

import jax
from numpyro import handlers

from grassmoth.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES


def _trace_public_sites(
    model_fn: Callable[[], None],
    *,
    exclude: Set[str] | None = None,
) -> frozenset[str]:
    """Trace the original prior and select its parameter sample sites for reporting.

    Reparameterization may later turn these sites into deterministics; replay
    still restores their original names. Assembled matrices are deterministic
    outputs of this original trace, not additional scientific parameters.
    """
    excluded = set(INTERNAL_DIAGNOSTIC_SITES)
    if exclude is not None:
        excluded.update(exclude)

    def public_values() -> dict[str, jax.Array]:
        with handlers.seed(rng_seed=0):
            trace: dict[str, Message] = handlers.trace(model_fn).get_trace()
        return {
            name: site["value"]
            for name, site in trace.items()
            if site["type"] == "sample"
            and not site.get("is_observed", False)
            and name not in excluded
        }

    shapes: dict[str, jax.ShapeDtypeStruct] = jax.eval_shape(public_values)
    return frozenset(shapes)


def _filter_public_samples[Sample](
    samples: Mapping[str, Sample], public_sites: Set[str]
) -> dict[str, Sample]:
    """Drop internal handler sites, keeping only original model outputs."""
    return {name: values for name, values in samples.items() if name in public_sites}
