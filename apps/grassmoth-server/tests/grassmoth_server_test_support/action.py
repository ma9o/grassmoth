"""Execute model staging and action-owned checks in local contract tests."""

from grassmoth_server.actions.contracts.inputs import EditQuestionInput
from grassmoth_server.actions.edit_model import edit_model
from grassmoth_test_support.helpers import fixture_entity_id


def edit_and_check(workspace_id, request, state):
    return edit_model(workspace_id, request)


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as edit_question does."""
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth_server.actions.contracts.requests import EditQuestionRequest
    from grassmoth_server.actions.edit_question import edit_question
    from grassmoth_server.study.persistence.history import StudyRepository

    request = EditQuestionRequest(
        input=EditQuestionInput(
            question=question
            if question is not None
            else QuestionSpec(
                text="What does it answer?",
                outcome=fixture_entity_id("construct", "unmeasured_outcome"),
            )
        )
    )
    journal = StudyRepository(workspace_id)
    return journal.append(
        applied_record(
            workspace_id,
            edit_question(workspace_id, request),
            request=request,
            seq=journal.latest_seq() + 1,
        )
    )


def applied_record(
    workspace_id, result, *, seq, request=None, ts="2026-01-01T00:00:00Z", **metadata
):
    """A successful test publication composes the actual owned action payload."""
    from pydantic import TypeAdapter

    from grassmoth.actions.checks import model_check_report
    from grassmoth.actions.contracts.bodies import (
        EditModelOutput,
        EditQuestionOutput,
        FitOutput,
        SimulateOutput,
    )
    from grassmoth.actions.prepare_data.action import prepare_data
    from grassmoth.artifacts.model.dynamical_model_spec import ModelEditResult
    from grassmoth.artifacts.observations.preparation import DataPreparationResult
    from grassmoth.artifacts.reports.identification import IdentificationReport
    from grassmoth.artifacts.reports.inference import (
        FitCheckReport,
        InferenceEvidence,
        InferenceReport,
    )
    from grassmoth.artifacts.reports.model_checks import ModelCheckReport
    from grassmoth.artifacts.simulation import SimulationEvidence, SimulationReport
    from grassmoth.models.structure import StructuralSelection
    from grassmoth.study.projections.visuals import simulation_observation_histories
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.persistence.store import ArtifactStore, read_model, read_question
    from grassmoth_server.study.records import ActionAttempt, Applied, AttemptRecord
    from grassmoth_server.study.state import apply_effects

    store = ArtifactStore(workspace_id)
    state = apply_effects(
        StudyRepository(workspace_id).state(), result.effects.produced, result.effects.retracted
    )
    reports = result.effects.reports
    if isinstance(result.result, str):
        identity = result.result
        action = (
            request.action
            if request is not None
            else next(item.produced_by for item in result.effects.produced)
        )
    else:
        if not state.has("question"):
            from grassmoth_server_test_support.helpers import write_question

            state = state.with_artifacts((write_question(store),))
        question = read_question(store, state.current["question"].revision)
        match result.result:
            case DataPreparationResult():
                from grassmoth.artifacts.observations.dataset import ObservationDataset
                from grassmoth_server.study.queries.lineage import read_data_metadata

                revision = state.current["panel"].revision
                data = read_data_metadata(store, revision)
                frame = store.read_parquet_file("panel", revision, "panel.parquet")
                output = prepare_data(
                    ObservationDataset.from_frame(
                        frame, data.variables, time_origin=data.time_origin
                    ),
                    data,
                    result.result,
                )
                if "data-profile" in reports:
                    from grassmoth.artifacts.reports.data_quality import DataProfileReport

                    output = output.revised(
                        profile=store.read_result(reports["data-profile"], DataProfileReport)
                    )
                action = "prepare_data"
            case InferenceEvidence():
                from grassmoth.artifacts.reports.data_quality import DataProfileReport
                from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
                from grassmoth_test_support.inference import _report

                model = read_model(store, state.current["model"].revision)
                output = FitOutput(
                    dynamical_model_spec=model,
                    checks=store.read_result(reports["checks"], FitCheckReport)
                    if "checks" in reports
                    else FitCheckReport(
                        data=DataProfileReport(indicators={}, findings=()),
                        preflight=(),
                        question=QuestionCheckReport(findings=()),
                    ),
                    inference=store.read_result(reports["inference"], InferenceReport)
                    if "inference" in reports
                    else _report(model).revised(evidence=result.result),
                )
                action = "fit"
            case SimulationEvidence():
                report = store.read_result(reports["simulation"], SimulationReport)
                evidence = report.evidence
                output = SimulateOutput(
                    report=report,
                    data=simulation_observation_histories(
                        evidence,
                        evidence.arms.action.observations.values,
                        evidence.observation_layout.mask.values,
                    ),
                )
                action = "simulate"
            case _:
                if state.has("model"):
                    model = read_model(store, state.current["model"].revision)
                    selection = StructuralSelection.for_question(model, question)
                    output = EditModelOutput(
                        dynamical_model_spec=model,
                        checks=store.read_result(reports["checks"], ModelCheckReport)
                        if "checks" in reports
                        else model_check_report(selection, question),
                        identification=store.read_result(
                            reports["identification"], IdentificationReport
                        )
                        if "identification" in reports
                        else selection.identification,
                        pruning=result.result
                        if isinstance(result.result, ModelEditResult)
                        else ModelEditResult(),
                    )
                    action = "edit_model"
                else:
                    output = EditQuestionOutput(question=question)
                    action = "edit_question"
        identity = store.write_result(output)
    saved = TypeAdapter(ActionAttempt).validate_python(
        {
            "action": action,
            "request": request,
            "outcome": Applied(result=identity, effects=result.effects.revised(reports={})),
        }
    )
    return AttemptRecord(seq=seq, ts=ts, attempt=saved, **metadata)


def empty_simulation_summary():
    """Reports used only for routing have no plotted reductions."""
    from grassmoth.artifacts.simulation import SimulationSummary

    return SimulationSummary(
        state_frames={},
        indicator_frames={},
        action_category_probabilities={},
        reference_category_probabilities={},
    )


def model_findings(workspace_id, state, *, action):
    """Evaluate the selected scientific values in a storage-backed fixture."""
    from grassmoth.actions.checks import (
        fit_check_report,
        model_check_report,
    )
    from grassmoth.models.structure import StructuralSelection
    from grassmoth_server.study.persistence.store import ArtifactStore, read_model, read_question
    from grassmoth_server.study.queries.data import read_data_history

    store = ArtifactStore(workspace_id)
    model = read_model(store, state.current["model"].revision)
    question = read_question(store, state.current["question"].revision)
    selection = StructuralSelection.for_question(model, question)
    if action == "fit":
        history = read_data_history(store, state.data)
        return fit_check_report(selection, question, history.observations)
    return model_check_report(selection, question), selection.identification


def prepared_profile(workspace_id, state, applied):
    """Read a fixture's produced panel and exercise the pure profiler."""
    from grassmoth.artifacts.observations.reference import DataRef
    from grassmoth_server.study.persistence.store import ArtifactStore
    from grassmoth_server.study.state import apply_effects

    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    return data_profile(
        ArtifactStore(workspace_id),
        DataRef(revision=selected.current["panel"].revision, replicate_index=0),
    )


def data_profile(store, source):
    from grassmoth.observations.quality.profile import profile_data
    from grassmoth_server.study.queries.data import read_data_history

    history = read_data_history(store, source)
    return profile_data(
        history.observations.recorded.frame,
        definitions=history.variables,
        metadata=history.metadata,
    )
