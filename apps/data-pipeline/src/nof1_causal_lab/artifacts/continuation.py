"""Persisted last-draw state and frozen tuning for production fit continuation."""

from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.joint_parameterization import JointParameterizationRecord
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import InitializationDiagnostics
from nof1_causal_lab.artifacts.sampling import (
    MarginalParticleGibbsSpec,
    NutsDASpec,
    ParticleGibbsSpec,
    SamplerSpec,
)


class JointTuning(Value):
    """Joint coordinates are reconstructed from the original model and trace seed."""

    kind: Literal["joint"] = "joint"
    parameterization: JointParameterizationRecord


class TransferredTuning(Value):
    """Only scientific values transfer across kernels; warmup must prepare new tuning."""

    kind: Literal["transferred"] = "transferred"


class ParticleTuning(Value):
    """Frozen proposal geometry; these values are not posterior draws."""

    kind: Literal["particle"] = "particle"
    parameter_chol: NumericalArray
    pilot_mean: NumericalArray
    pilot_variance: NumericalArray
    parameter_reference_path: NumericalArray
    gaussian_reference: tuple[NumericalArray, ...] = ()
    initialization: InitializationDiagnostics


type FrozenSamplerTuning = Annotated[
    JointTuning | ParticleTuning | TransferredTuning, Field(discriminator="kind")
]


class SamplerCheckpoint(Value):
    """An exact native state at the last retained draw, with its next random key."""

    settings: SamplerSpec
    parameters: NumericalArray
    paths: NumericalArray
    random_key: NumericalArray
    native_state: tuple[NumericalArray, ...]
    tuning: FrozenSamplerTuning


def sampler_identity(settings: SamplerSpec) -> tuple[str | int, ...]:
    """Identify kernels whose private state has the same scientific interpretation."""
    algorithm = settings.algorithm
    if isinstance(algorithm, NutsDASpec):
        return (algorithm.method, algorithm.trajectory.kind, algorithm.parameterization)
    if isinstance(algorithm, ParticleGibbsSpec):
        return (algorithm.method, algorithm.parameter_kernel.method)
    assert isinstance(algorithm, MarginalParticleGibbsSpec)
    return (
        algorithm.method,
        algorithm.parameter_proposal,
        1 if algorithm.tempering is None else algorithm.tempering.num_temperatures,
    )
