"""The rules for continuing a previous fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.continuation import TransferredTuning

if TYPE_CHECKING:
    from grassmoth.actions.contracts.io import FitInputs
    from grassmoth.artifacts.continuation import SamplerCheckpoint


class ContinuationRefusal(Value):
    """A continuation that the previous fit's chains cannot resume."""

    code: Literal["INCOMPATIBLE_CONTINUATION", "CONTINUATION_REQUIRES_WARMUP"]
    message: str


def resumed_checkpoint(inputs: FitInputs) -> SamplerCheckpoint | ContinuationRefusal | None:
    """The previous fit's checkpoint; a new kernel keeps the chain states but re-tunes them."""
    checkpoint = None if inputs.previous is None else inputs.previous.inference.evidence.checkpoint
    if checkpoint is None:
        return None
    sampler = inputs.sampler
    if (
        sampler.num_chains != checkpoint.settings.num_chains
        or sampler.seed != checkpoint.settings.seed
    ):
        return ContinuationRefusal(
            code="INCOMPATIBLE_CONTINUATION",
            message="Continuation inherits its chain count and random stream",
        )
    if sampler.kernel_identity == checkpoint.settings.kernel_identity:
        return checkpoint
    if sampler.num_warmup == 0:
        return ContinuationRefusal(
            code="CONTINUATION_REQUIRES_WARMUP",
            message="Changing the sampler kernel requires positive warmup",
        )
    return checkpoint.with_tuning(tuning=TransferredTuning())
