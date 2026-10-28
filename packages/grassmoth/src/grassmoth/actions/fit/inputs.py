"""One fit's resolved inputs and the rules for continuing a previous fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.continuation import TransferredTuning

if TYPE_CHECKING:
    from grassmoth.actions.contracts.bodies import FitOutput
    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.models.structure import StructuralSelection


@dataclass(frozen=True)
class FitInputs:
    """The scoped model, its question, the data and sampler, and any fit this one continues.

    ``compute_loo_diagnostics`` retains pointwise log-likelihoods for LOO.
    """

    selection: StructuralSelection
    question: QuestionSpec
    data_for_model: ObservationDataset
    sampler: SamplerSpec
    compute_loo_diagnostics: bool
    previous: FitOutput | None = None


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
            message="Switching algorithm or joint parameterization requires positive warmup",
        )
    return checkpoint.revised(tuning=TransferredTuning())
