"""Render diagnostics retained by the fit action for its latest segment."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from grassmoth_server.public.contracts import FitOutput, JsonObject


def segment_summary(result: FitOutput) -> JsonObject:
    """Pair this segment's ESS and convergence with this segment's sampling interval."""
    segment = result.inference.core.inference_metadata.segments[-1]
    diagnostics = segment.diagnostics
    gate = segment.convergence
    seconds = sum(p.end_seconds - p.start_seconds for p in segment.phases if p.stage == "sampling")
    rows = (
        () if diagnostics is None else (*diagnostics.per_parameter, *(diagnostics.per_latent or ()))
    )
    bulk = [row.ess_bulk for row in rows if row.ess_bulk is not None]
    tail = [row.ess_tail for row in rows if row.ess_tail is not None]
    hats = [row.r_hat for row in rows if row.r_hat is not None]
    minimum = min((*bulk, *tail)) if bulk and tail else None
    return {
        "gate": gate.status,
        "warnings": list(gate.warnings),
        "sampling_seconds": seconds,
        "min_bulk_ess": min(bulk) if bulk else None,
        "min_tail_ess": min(tail) if tail else None,
        "max_r_hat": max(hats) if hats else None,
        "min_ess_per_sampling_second": minimum / seconds
        if minimum is not None and seconds > 0
        else None,
        "min_ess_per_sweep": minimum / (segment.stop - segment.start)
        if minimum is not None
        else None,
        "num_divergences": diagnostics.num_divergences if diagnostics is not None else None,
        "bfmi": list(diagnostics.energy.bfmi)
        if diagnostics is not None and diagnostics.energy is not None
        else None,
        "segment_diagnostics": None
        if diagnostics is None
        else cast("JsonObject", diagnostics.model_dump(mode="json")),
    }
