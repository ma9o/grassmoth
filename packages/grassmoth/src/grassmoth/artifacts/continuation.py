"""Persisted last-draw state and frozen tuning for production fit continuation."""

from typing import Annotated, Literal, Self

from pydantic import Field

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.joint_parameterization import JointParameterizationRecord
from grassmoth.artifacts.reports.posterior_diagnostics import InitializationDiagnostics
from grassmoth.artifacts.sampling import (
    SamplerSpec,
)


class TransferredTuning(Value):
    """Only scientific values transfer across kernels; warmup must prepare new tuning."""

    kind: Literal["transferred"] = "transferred"


class ParticleTuning(Value):
    """Frozen proposal geometry; these values are not posterior draws."""

    kind: Literal["particle"] = "particle"
    parameter_chol: NumericalArray
    parameter_reference_path: NumericalArray
    initialization: InitializationDiagnostics

    def with_parameter_cholesky(self, *, parameter_chol: NumericalArray) -> Self:
        """Return particle tuning with its parameter proposal factor."""
        return self._revised(parameter_chol=parameter_chol)


class JointTuning(Value):
    """Frozen native joint coordinates, including the mixed reading mask."""

    kind: Literal["joint"] = "joint"
    parameterization: JointParameterizationRecord


type FrozenSamplerTuning = Annotated[
    ParticleTuning | JointTuning | TransferredTuning, Field(discriminator="kind")
]


class SamplerCheckpoint(Value):
    """An exact native state at the last retained draw, with its next random key."""

    settings: SamplerSpec
    parameters: NumericalArray
    paths: NumericalArray
    random_key: NumericalArray
    native_state: tuple[NumericalArray, ...]
    tuning: FrozenSamplerTuning

    def with_tuning(self, *, tuning: FrozenSamplerTuning) -> Self:
        """Return the checkpoint with its retained sampler tuning."""
        return self._revised(tuning=tuning)
