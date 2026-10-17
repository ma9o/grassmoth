"""Predeclared action recipes, assigned treatments, and comparison tolerances."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.preparation import ExtractionSpec
from nof1_causal_lab.artifacts.reports.checks import NotEvaluatedReason
from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.json_types import JsonValue


class CaseFile(Value):
    """A recipe-relative file whose bytes must match before any action is submitted."""

    kind: Literal["file"] = "file"
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class StudyModel(Value):
    """A read-only model result in a local study, selected by immutable tree and hash."""

    kind: Literal["study_model"] = "study_model"
    repository: str
    revision: GitOid
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


type ModelSource = Annotated[CaseFile | StudyModel, Field(discriminator="kind")]


class GeneratedData(Value):
    """The production simulator generates a pinned replicate from a literal truth model."""

    kind: Literal["simulate"] = "simulate"
    truth_model: CaseFile
    simulation: SimulationSpec


class RecordedData(Value):
    """The production preparation action imports immutable source files."""

    kind: Literal["prepare_data"] = "prepare_data"
    files: tuple[CaseFile, ...] = Field(min_length=1)
    provenance_files: tuple[CaseFile, ...] = ()
    extraction: Mapping[IndicatorId, ExtractionSpec]
    context: str = ""


class CaseRecipe(Value):
    """Case content is authored through actions; paths are relative to the recipe."""

    name: str
    question: QuestionSpec
    fit_model: ModelSource
    data: Annotated[GeneratedData | RecordedData, Field(discriminator="kind")]
    expected_identified: tuple[str, ...] = ()
    allowed_findings: Mapping[str, str] = Field(default_factory=dict)
    unavailable_findings: Mapping[NotEvaluatedReason, str] = Field(default_factory=dict)
    truth: Mapping[str, float] = Field(default_factory=dict)
    provenance: str


class SeedPair(Value):
    """Independent dataset generation and sampler replication controls."""

    generation: int = Field(ge=0)
    sampler: int = Field(ge=0)


class Arm(Value):
    """Only declared factors may override the base; runtime factors name expected values."""

    name: str
    settings: Mapping[str, JsonValue] = Field(default_factory=dict)
    runtime: Mapping[str, JsonValue] = Field(default_factory=dict)
    endpoint: str | None = None


class Reference(Value):
    """A saved fit is a reference only after stricter convergence and precision qualification."""

    workspace: str
    case_manifest: str
    call_id: str
    endpoint: str | None = None
    max_r_hat: float = Field(default=1.01, gt=1, le=1.05)
    max_mcse_sd: float = Field(default=0.05, gt=0)


class Tolerances(Value):
    """Equivalence requires a complete MCSE uncertainty band inside the assigned bound."""

    mean_sd: float = Field(gt=0)
    sd_relative: float = Field(gt=0)
    mcse_multiplier: float = Field(default=2, gt=0)


class PosteriorComparison(Value):
    """References are pinned separately for every generated dataset seed."""

    references: Mapping[int, Reference]
    tolerance: Tolerances


class Experiment(Value):
    """A single factor or explicit interaction, paired across seeds and timing repeats."""

    name: str
    case: str
    base: SamplerSpec
    factors: tuple[str, ...] = Field(min_length=1)
    interaction: bool = False
    arms: tuple[Arm, ...] = Field(min_length=2)
    seeds: tuple[SeedPair, ...] = Field(min_length=1)
    repeats: int = Field(default=1, ge=1)
    concurrent_arms: int = Field(default=1, ge=1)
    reservation_usd_per_fit: float = Field(default=0, ge=0)
    posterior: PosteriorComparison | None = None
    probe_samples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def assigned_factors(self) -> Self:
        """Reject ambiguous contrasts and accidental changes before any action runs."""
        if len(set(self.factors)) != len(self.factors) or len({a.name for a in self.arms}) != len(
            self.arms
        ):
            raise ValueError("Factor paths and arm names must be unique")
        if (len(self.factors) > 1) != self.interaction:
            raise ValueError("Multiple factors require an explicitly declared interaction")
        if any(
            left != right and right.startswith(left + ".")
            for left in self.factors
            for right in self.factors
        ):
            raise ValueError("Factor paths must not overlap")
        if any(path in {"seed", "num_chains", "retain_latent_paths"} for path in self.factors):
            raise ValueError("Seeds, chain layout and draw retention are paired controls")
        for arm in self.arms:
            paths = set(arm.settings) | {"runtime." + key for key in arm.runtime}
            if paths != set(self.factors):
                raise ValueError("Every arm must assign exactly the declared factors")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use repeats for timing noise; seed pairs must be unique")
        if self.base.hardware == "local" and self.concurrent_arms != 1:
            raise ValueError(
                "Local timing arms must run serially to avoid shared-device contention"
            )
        if not self.base.retain_latent_paths:
            raise ValueError("Benchmark fits retain joint parameter and state draws")
        if self.probe_samples is not None and self.base.num_samples_per_chain == 0:
            raise ValueError("A probe requires a base fit with retained draws")
        if self.probe_samples is not None and any(
            path.startswith("algorithm.laps") for path in self.factors
        ):
            raise ValueError("Initializer comparisons require fresh fits, not continued probes")
        return self
