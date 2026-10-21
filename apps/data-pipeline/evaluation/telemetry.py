"""Project retained production telemetry without rerunning scientific computations."""

from typing import cast

import numpy as np

from nof1_causal_lab.public.contracts import FitOutput, JsonObject, JsonValue


def sampler_outcomes(result: FitOutput) -> JsonObject:
    """Keep movement, adaptation and phase measurements apart from assigned controls."""
    evidence = result.inference.evidence
    metadata = result.inference.core.inference_metadata
    segment = metadata.segments[-1]
    row: dict[str, JsonValue] = {
        "draw_start": segment.start,
        "draw_stop": segment.stop,
        "phases": [phase.model_dump(mode="json") for phase in segment.phases],
        "parameterization": None
        if segment.parameterization is None
        else segment.parameterization.model_dump(mode="json"),
    }
    diagnostics = metadata.sampler_diagnostics
    if diagnostics is not None and segment.stop > segment.start:
        row["sampler_outcomes"] = cast(
            "JsonObject", diagnostics.model_dump(mode="json", exclude={"settings"})
        )
    if evidence.final_latent_delta is not None:
        row["final_latent_delta"] = np.asarray(evidence.final_latent_delta.values).tolist()
    return row
