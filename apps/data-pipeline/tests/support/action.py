"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_service.actions.contracts.inputs import EditQuestionInput
from nof1_causal_service.actions.edit_model import edit_model
from tests.support.helpers import fixture_entity_id


def edit_and_check(workspace_id, request, state):
    return edit_model(workspace_id, request)


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as edit_question does."""
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_service.actions.contracts.requests import EditQuestionRequest
    from nof1_causal_service.actions.edit_question import edit_question
    from nof1_causal_service.study.persistence.history import StudyRepository

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

    from nof1_causal_lab.actions.checks import model_check_report
    from nof1_causal_lab.actions.contracts.bodies import (
        EditModelOutput,
        EditQuestionOutput,
        FitOutput,
        SimulateOutput,
    )
    from nof1_causal_lab.actions.prepare_data.action import prepare_data
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
    from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
    from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
    from nof1_causal_lab.artifacts.reports.inference import (
        FitCheckReport,
        InferenceEvidence,
        InferenceReport,
    )
    from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport
    from nof1_causal_lab.artifacts.simulation import SimulationEvidence, SimulationReport
    from nof1_causal_lab.models.structure import StructuralSelection
    from nof1_causal_lab.study.projections.visuals import simulation_observation_histories
    from nof1_causal_service.study.persistence.history import StudyRepository
    from nof1_causal_service.study.persistence.store import ArtifactStore, read_model, read_question
    from nof1_causal_service.study.records import ActionAttempt, Applied, AttemptRecord
    from nof1_causal_service.study.state import apply_effects

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
            from tests.support.helpers import write_question

            state = state.with_artifacts((write_question(store),))
        question = read_question(store, state.current["question"].revision)
        match result.result:
            case DataPreparationResult():
                from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
                from nof1_causal_service.study.queries.lineage import read_data_metadata

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
                    from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport

                    output = output.revised(
                        profile=store.read_result(reports["data-profile"], DataProfileReport)
                    )
                action = "prepare_data"
            case InferenceEvidence():
                from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
                from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
                from tests.support.inference import _report

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
    from nof1_causal_lab.artifacts.simulation import SimulationSummary

    return SimulationSummary(
        state_frames={},
        indicator_frames={},
        action_category_probabilities={},
        reference_category_probabilities={},
    )


def model_findings(workspace_id, state, *, action):
    """Evaluate the selected scientific values in a storage-backed fixture."""
    from nof1_causal_lab.actions.checks import (
        check_model_data,
        fit_check_report,
        model_check_report,
    )
    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model
    from nof1_causal_lab.models.structure import StructuralSelection
    from nof1_causal_service.study.persistence.store import ArtifactStore, read_model, read_question
    from nof1_causal_service.study.queries.data import read_data_history

    store = ArtifactStore(workspace_id)
    model = read_model(store, state.current["model"].revision)
    question = read_question(store, state.current["question"].revision)
    selection = StructuralSelection.for_question(model, question)
    if action == "fit":
        history = read_data_history(store, state.data)
        findings = check_model_data(
            compile_ssm_inputs_from_model(selection),
            history.observations,
            time_origin=history.time_origin,
        )
        return fit_check_report(selection, question, history.observations, preflight=findings)
    return model_check_report(selection, question), selection.identification


def prepared_profile(workspace_id, state, applied):
    """Read a fixture's produced panel and exercise the pure profiler."""
    from nof1_causal_lab.artifacts.observations.reference import DataRef
    from nof1_causal_service.study.persistence.store import ArtifactStore
    from nof1_causal_service.study.state import apply_effects

    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    return data_profile(
        ArtifactStore(workspace_id),
        DataRef(revision=selected.current["panel"].revision, replicate_index=0),
    )


def data_profile(store, source):
    from nof1_causal_lab.observations.quality.profile import profile_data
    from nof1_causal_service.study.queries.data import read_data_history

    history = read_data_history(store, source)
    return profile_data(
        history.observations.recorded.frame,
        definitions=history.variables,
        metadata=history.metadata,
    )
