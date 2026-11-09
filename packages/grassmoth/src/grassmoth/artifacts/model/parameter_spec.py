"""An uncertain scientific quantity participates in a probability law."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import DistributionId, ParameterId
from grassmoth.artifacts.model.parameter import PriorAuthoringTransform
from grassmoth.artifacts.model.reasoning import ReasoningSpec


class IdentityTransformSpec(Value):
    """Keep the authored probability law on the scientific quantity's native scale."""

    kind: Literal[PriorAuthoringTransform.IDENTITY] = PriorAuthoringTransform.IDENTITY


class PersistenceTransformSpec(Value):
    """Map persistence p to -log(p) divided by its explicit interval in days."""

    kind: Literal[PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY] = (
        PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY
    )
    interval_days: Annotated[float, Field(gt=0, allow_inf_nan=False)] | Literal["model_clock"]


class IntervalEffectTransformSpec(Value):
    """Divide an interval effect by its explicit duration in days."""

    kind: Literal[PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE] = (
        PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE
    )
    interval_days: Annotated[float, Field(gt=0, allow_inf_nan=False)] | Literal["model_clock"]


class InitialCorrelationTransformSpec(Value):
    """Constrain an initial-state correlation to its scientific support [-1, 1]."""

    kind: Literal[PriorAuthoringTransform.INITIAL_STATE_CORRELATION] = (
        PriorAuthoringTransform.INITIAL_STATE_CORRELATION
    )


type ParameterTransformSpec = Annotated[
    IdentityTransformSpec
    | PersistenceTransformSpec
    | IntervalEffectTransformSpec
    | InitialCorrelationTransformSpec,
    Field(discriminator="kind"),
]


class ParameterSpec(Value):
    """A named uncertain quantity and the scientific reasoning for its place in the model."""

    model_config = ConfigDict(revalidate_instances="always")

    id: ParameterId
    name: str = Field(description="Authored parameter label; relationships use its persistent ID")
    reasoning: ReasoningSpec = Field(
        description="What this quantity means and, for an informative prior, where its values come from. Cite references with [n] in the text."
    )
    transform: ParameterTransformSpec = Field(default_factory=IdentityTransformSpec)
    distribution: DistributionId | None = Field(
        default=None,
        description=(
            "Membership in a native law in DynamicalModelSpec.distributions; may be joint. "
            "None means the law has not been assigned yet."
        ),
    )

    def conditioned(self, identity: DistributionId) -> ParameterSpec:
        """Select a conditioned parameter distribution and reset its transform to the quantity scale."""
        return self._revised(distribution=identity, transform=IdentityTransformSpec())
