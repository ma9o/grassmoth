"""Complete scientific outputs of the seven library actions.

The service supplies HTTP request and polling envelopes; these values contain
scientific results independently of their transport or persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec, ModelEditResult
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.history import ObservationData
from grassmoth.artifacts.observations.preparation import (
    DataPreparationResult,
    PreparedDataMetadata,
)
from grassmoth.artifacts.reports.data_comparison import DataComparisonReport
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.artifacts.reports.inference import FitCheckReport, InferenceReport
from grassmoth.artifacts.reports.model_checks import ModelCheckReport
from grassmoth.artifacts.simulation import SimulationReport

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np
    import polars as pl

    from grassmoth.artifacts.probability.joint_layout import JointLawLayout
    from grassmoth.artifacts.reports.inference import FitSegment

# edit_question


class EditQuestionOutput(Value):
    """The question saved by this action."""

    question: QuestionSpec = Field(
        description="The saved question, including its outcome and intervention queries."
    )


# edit_model


class EditModelOutput(Value):
    """The produced model and its recorded scientific findings."""

    dynamical_model_spec: DynamicalModelSpec = Field(description="Saved scientific definition.")
    checks: ModelCheckReport = Field(
        description="Combined findings retained by the action that produced the model."
    )
    identification: IdentificationReport = Field(description="Causal identification findings.")
    pruning: ModelEditResult = Field(
        description="Entities removed because they do not serve the question."
    )


# prepare_data


class PrepareDataOutput(Value):
    """Prepared observations, their resolved metadata, and data-quality findings."""

    data: ObservationData = Field(
        description=(
            "Full observation histories keyed by indicator ID, including measurement support "
            "intervals and missing values."
        )
    )
    metadata: PreparedDataMetadata = Field(
        description=(
            "Preparation recipe, resolved observation schema, and the "
            "calendar origin used to interpret model time."
        )
    )
    profile: DataProfileReport = Field(
        description="Empirical statistics and data-quality findings for the prepared panel."
    )
    extraction: DataPreparationResult = Field(
        description="Retained extraction workers, failed chunks and reuse counts."
    )

    def observation_frame(self) -> pl.DataFrame:
        """Reconstruct the canonical table retained by this preparation."""
        import polars as pl

        from grassmoth.artifacts.arrays import resolve_vector
        from grassmoth.artifacts.observations.dataset import ObservationDataset
        from grassmoth.artifacts.observations.time import ModelTime, ObservationInstant

        rows = []
        for variable in self.metadata.variables:
            history = self.data[variable.id]
            origin = ObservationInstant(self.metadata.time_origin)

            def instant(day: float | None, origin: ObservationInstant = origin) -> datetime | None:
                return (
                    ModelTime(day).at(origin).value.replace(tzinfo=None)
                    if day is not None
                    else None
                )

            starts = resolve_vector(history.support_start)
            ends = resolve_vector(history.support_end)
            values = resolve_vector(history.values)
            rows.extend(
                {
                    "indicator_id": variable.id,
                    "value": value,
                    "anchor_time": instant(day),
                    "support_start": instant(start),
                    "support_end": instant(end),
                    "support_kind": variable.support_kind.value,
                    "summary_operator": variable.summary_operator.value,
                    "anchor_policy": variable.anchor_policy.value,
                    "observation_window": str(variable.observation_window),
                }
                for day, start, end, value in zip(history.times, starts, ends, values, strict=True)
            )
        schema = ObservationDataset.row_schema() | {
            "value": pl.Float64,
            **{name: pl.Datetime("us") for name in ("anchor_time", "support_start", "support_end")},
        }
        return pl.DataFrame(rows, schema=schema).sort("indicator_id", "anchor_time")


# fit


class FitOutput(Value):
    """A conditioned model, its completed checks, and self-contained inference evidence."""

    dynamical_model_spec: DynamicalModelSpec | None = Field(
        description="The combined conditioned model, or null when this fit and its parents have no retained draws. Zero-draw continuations reuse their parent's model without producing a revision."
    )
    checks: FitCheckReport = Field(
        description="Completed compatibility and question findings for the selected history."
    )
    inference: InferenceReport = Field(
        description="Run provenance, native sampler evidence, diagnostics and parameter summaries."
    )

    def posterior_draws(
        self, segment: FitSegment | None = None
    ) -> tuple[JointLawLayout, np.ndarray] | None:
        """Read aligned scientific coordinates, optionally selecting one interval per chain."""
        from grassmoth.artifacts.probability.serialization import empirical_atoms

        metadata = self.inference.core.inference_metadata
        model = self.dynamical_model_spec
        if model is None or metadata.distribution is None:
            return None
        layout = model.law_layouts[metadata.distribution]
        atoms = empirical_atoms(model.distributions[metadata.distribution])
        if segment is not None:
            atoms = atoms.reshape(metadata.num_chains, -1, atoms.shape[-1])[
                :, segment.start : segment.stop
            ].reshape(-1, atoms.shape[-1])
        return layout, atoms


# simulate


class SimulateOutput(Value):
    """Generated histories and their complete scientific report, including owned numerical evidence."""

    data: tuple[ObservationData, ...] = Field(
        min_length=1,
        description="One observation history per retained replicate, in replicate order.",
    )
    report: SimulationReport = Field(description="Generation evidence and evaluated findings.")


# data_diff


class DataDiffOutput(Value):
    """The computed comparison report; original histories remain in their producing results."""

    report: DataComparisonReport = Field(
        description=(
            "Exact history selections, point changes, per-history statistics, compatibility "
            "findings and predictive checks."
        )
    )


# model_diff


class ModelDiffOutput(Value):
    """Directional changes between saved specs, expressed in the model's editing language.

    DynamicalModelSpec.changes_from owns the document comparison. Applying its patch with
    merge_fields to the before spec reconstructs the after spec; equal specs yield
    an empty document. Only saved specification fields participate, including their
    exact law-buffer references. No execution findings or statistical comparisons
    are computed here; those remain owned by their producing actions.
    """

    model_config = ConfigDict(json_schema_mode_override="validation")

    changes: DynamicalModelSpec = Field(
        description=(
            "Merge this document into the before spec to obtain the after spec. Omitted fields "
            "are unchanged, supplied fields are added or updated, and null map entries delete "
            "their identities. A question revision denotes the empty spec."
        )
    )
