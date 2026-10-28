"""One simulation's resolved inputs and the ancestry of its laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import GitRef
    from grassmoth.artifacts.reports.inference import InferenceReportCore
    from grassmoth.artifacts.reports.predictive_provenance import (
        AuthoredLawProvenance,
        FittedLawProvenance,
        MixedLawProvenance,
        UnknownLawProvenance,
    )
    from grassmoth.artifacts.simulation import SimulationSpec
    from grassmoth.models.structure import StructuralSelection


@dataclass(frozen=True)
class FittedLaw:
    """Laws retained from a committed fit, with that fit's report."""

    provenance: FittedLawProvenance | MixedLawProvenance
    report: InferenceReportCore


type PredictiveLaw = AuthoredLawProvenance | UnknownLawProvenance | FittedLaw


@dataclass(frozen=True)
class SimulationInputs:
    """The scoped model, the design with its seed and replicate count, and its laws' ancestry.

    ``fitted`` pairs this revision with its own fit report when a fit produced it.
    """

    selection: StructuralSelection
    design: SimulationSpec
    seed: int
    num_replicates: int
    law: PredictiveLaw
    fitted: tuple[GitRef, InferenceReportCore] | None = None
