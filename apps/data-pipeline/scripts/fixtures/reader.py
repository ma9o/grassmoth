"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, cast

from nof1_causal_lab.actions.contracts.bodies import (
    EditModelOutput,
    FitOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.actions.input_resolution import input_state
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model
from nof1_causal_lab.study.records import Applied, StudyRevision, inference_record
from nof1_causal_lab.study.snapshot import ModelSnapshot

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, EntityRef, GitOid, IndicatorId
    from nof1_causal_lab.artifacts.model.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.model.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_lab.artifacts.observations.history import ObservationHistory
    from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
    from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
    from nof1_causal_lab.artifacts.reports.inference import (
        FitCheckReport,
        InferenceReport,
        InferenceReportCore,
    )
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.study.state import StudyState


class ModelReader:
    """Compose saved action results through their pinned scientific inputs."""

    def __init__(self, workspace_id: str, *, at: GitOid, state: StudyState | None = None) -> None:
        """Pin reads to a successful study commit and optionally an explicitly composed input state."""
        self.repository = StudyRepository(workspace_id)
        self.commit_id = self.repository.resolve(at=at)
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        self._state = state
        self.selected = cache(self._selected)

    @cached_property
    def state(self) -> StudyState:
        """Artifact and data selections reconstructed for the pinned action, including its input context."""
        from nof1_causal_lab.study.state import apply_effects

        if self._state is not None:
            return self._state
        checkpoint = self.repository.state(self.commit_id)
        if not self.records:
            return checkpoint
        attempt = self.records[-1].record.attempt
        if attempt.request is None:
            return checkpoint
        inputs = input_state(self.repository, attempt.request)
        context = checkpoint.with_artifacts(tuple(inputs.current.values()))
        if attempt.action in {"fit", "edit_model"}:
            context = context.revised(data=inputs.data)
        if not isinstance(attempt.outcome, Applied):
            raise RuntimeError("A result reader requires an applied call")
        return apply_effects(
            context, attempt.outcome.effects.produced, attempt.outcome.effects.retracted
        )

    @cached_property
    def records(self) -> list[StudyRevision]:
        """Retained journal records reachable from the pinned study commit."""
        return self.repository.records(self.commit_id)

    @cached_property
    def seq(self) -> int:
        """Sequence number at the selected checkpoint, or zero at the study root."""
        return self.records[-1].record.seq if self.records else 0

    def _selected(self, artifact_id: ArtifactId) -> object:
        from pydantic import TypeAdapter

        from nof1_causal_lab.study.artifact_files import artifact_file_spec
        from scripts.artifact_contracts import ARTIFACT_CONTRACTS

        revision = self.state.current[artifact_id].revision
        if artifact_id == "model":
            return read_model(self.store, revision)
        filename = next(iter(artifact_file_spec(artifact_id).json_files.values()))
        return TypeAdapter(ARTIFACT_CONTRACTS[artifact_id]).validate_python(
            self.store.read_json_file(artifact_id, revision, filename),
            context={"distribution_array_loader": cache(self.store.read_array)},
        )

    @cached_property
    def question(self) -> QuestionSpec | None:
        """Selected authored study question, or ``None`` before a question is available."""
        return (
            cast("QuestionSpec", self.selected("question")) if self.state.has("question") else None
        )

    @cached_property
    def dynamical_model_spec(self) -> DynamicalModelSpec | None:
        """Selected scientific model definition, or ``None`` before a model is available."""
        return (
            cast("DynamicalModelSpec", self.selected("model")) if self.state.has("model") else None
        )

    def constructs(self) -> tuple[ConstructSpec, ...]:
        """Return the selected model's constructs, or an empty tuple without a model."""
        return self.dynamical_model_spec.constructs if self.dynamical_model_spec else ()

    def edges(self) -> tuple[CausalEdgeSpec, ...]:
        """Return the selected model's causal edges, or an empty tuple without a model."""
        return self.dynamical_model_spec.edges if self.dynamical_model_spec else ()

    def indicators(self) -> tuple[IndicatorSpec, ...]:
        """Return the selected model's observation indicators, or an empty tuple without a model."""
        return self.dynamical_model_spec.indicators if self.dynamical_model_spec else ()

    def parameters(self, owner: EntityRef | None = None) -> tuple[ParameterSpec, ...]:
        """Return model parameters, optionally restricted to those owned by a scientific entity."""
        if self.dynamical_model_spec is None:
            return ()
        return (
            self.dynamical_model_spec.parameters
            if owner is None
            else self.dynamical_model_spec.parameters_for(owner.id)
        )

    @cached_property
    def _fit_record(self) -> StudyRevision | None:
        model = self.state.get("model")
        return inference_record(self.records, model.revision) if model is not None else None

    @cached_property
    def fit_result(self) -> FitOutput | None:
        """The saved fit that produced the selected model."""
        record = self._fit_record
        if record is None:
            return None
        assert isinstance(record.record.attempt.outcome, Applied)
        return self.store.read_result(record.record.attempt.outcome.result, FitOutput)

    @cached_property
    def fit_for_selected_data(self) -> FitOutput | None:
        """Select model/data findings only while their conditioning history is selected."""
        record = self._fit_record
        if record is None:
            return None
        attempt = record.record.attempt
        assert attempt.action == "fit"
        assert attempt.request is not None
        selected = attempt.request.input.data_ref
        return self.fit_result if selected == self.state.data else None

    def model_output(self) -> EditModelOutput | None:
        """Follow model inputs to the edit that owns structural and authoring findings."""
        selected = self.state.get("model")
        while selected is not None:
            record = next(
                (
                    item
                    for item in reversed(self.records)
                    if item.record.attempt.action == "edit_model"
                    and isinstance(item.record.attempt.outcome, Applied)
                    and any(
                        artifact.revision == selected.revision
                        for artifact in item.record.attempt.outcome.effects.produced
                    )
                ),
                None,
            )
            if record is not None:
                assert isinstance(record.record.attempt.outcome, Applied)
                return self.store.read_result(record.record.attempt.outcome.result, EditModelOutput)
            parent = selected.derived_from.get("model")
            selected = self.store.read_meta("model", parent) if parent is not None else None
        return None

    @cached_property
    def prepared_result(self) -> PrepareDataOutput | None:
        """The preparation result owning the selected panel, with no lookup in unrelated calls."""
        if self.state.data is None:
            return None
        from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
        from nof1_causal_lab.study.queries.data import panel_revision, read_data_source

        if not isinstance(
            read_data_source(self.store, self.state.data.revision), PreparedDataMetadata
        ):
            return None
        panel = panel_revision(self.store, self.state.data.revision)
        record = next(
            (
                item
                for item in reversed(self.records)
                if item.record.attempt.action == "prepare_data"
                and isinstance(item.record.attempt.outcome, Applied)
                and any(
                    info.artifact_id == "panel" and info.revision == panel
                    for info in item.record.attempt.outcome.effects.produced
                )
            ),
            None,
        )
        if record is None:
            return None
        assert isinstance(record.record.attempt.outcome, Applied)
        return self.store.read_result(record.record.attempt.outcome.result, PrepareDataOutput)

    @property
    def data_metadata(self) -> PreparedDataMetadata | None:
        """Preparation recipe and calendar coordinates retained with its result."""
        return self.prepared_result.metadata if self.prepared_result is not None else None

    @property
    def data_profile(self) -> DataProfileReport | None:
        """Data-quality findings owned by the selected preparation."""
        return self.prepared_result.profile if self.prepared_result is not None else None

    @property
    def inference_report(self) -> InferenceReport | None:
        """The complete diagnostic report saved by the owning fit."""
        return self.fit_result.inference if self.fit_result is not None else None

    @property
    def fit_checks(self) -> FitCheckReport | None:
        """Model/data findings retained by the owning fit."""
        return self.fit_for_selected_data.checks if self.fit_for_selected_data is not None else None

    def identification(self) -> IdentificationReport | None:
        """Identification findings retained by the model edit."""
        result = self.model_output()
        return result.identification if result is not None else None

    def fit(self) -> InferenceReportCore | None:
        """Select the owning fit's recorded scientific summaries."""
        return self.fit_result.inference.core if self.fit_result is not None else None

    @cached_property
    def simulation_result(self) -> SimulateOutput | None:
        """The latest explicit simulation in the selected action history."""
        for record in reversed(self.records):
            attempt = record.record.attempt
            if attempt.action == "simulate" and isinstance(attempt.outcome, Applied):
                return self.store.read_result(attempt.outcome.result, SimulateOutput)
        return None

    def simulation(self) -> SimulationReport | None:
        """Findings saved by the selected simulation."""
        return self.simulation_result.report if self.simulation_result is not None else None

    def observation_history(self, indicator_id: IndicatorId) -> ObservationHistory | None:
        """Select one saved observation series from the exact input history."""
        if self.state.data is None:
            return None
        if self.prepared_result is not None:
            return self.prepared_result.data.get(indicator_id)
        attempt = self.repository.record(self.state.data.revision).record.attempt
        assert attempt.action == "simulate"
        assert isinstance(attempt.outcome, Applied)
        result = self.store.read_result(attempt.outcome.result, SimulateOutput)
        return result.data[self.state.data.replicate_index].get(indicator_id)

    def snapshot(self) -> ModelSnapshot:
        """Compose stored values by their action ownership for the workbench."""
        model_output = self.model_output()
        fit = self.fit_for_selected_data
        return ModelSnapshot(
            workspace_id=self.workspace_id,
            commit_id=self.commit_id,
            selected_seq=self.seq,
            state=self.state,
            question=self.question,
            dynamical_model_spec=self.dynamical_model_spec,
            metadata=self.data_metadata,
            profile=self.data_profile,
            identification=model_output.identification if model_output else None,
            fit_checks=self.fit_checks,
            fit=self.fit(),
            specification=model_output.checks.specification
            if model_output and model_output.checks
            else None,
            question_checks=fit.checks.question
            if fit
            else model_output.checks.question
            if model_output and model_output.checks
            else None,
            simulation=self.simulation(),
        )
