"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.contracts.bodies import EditQuestionInput
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.model_checks import evaluate_model_checks
from tests.support.helpers import fixture_entity_id


def edit_and_check(workspace_id, request, state):
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from nof1_causal_lab.study.records import Applied

    staged = edit_model(workspace_id, request)
    assert isinstance(staged, Applied)
    checks, identification = evaluate_model_checks(workspace_id, state, staged, action="edit_model")
    store = ArtifactStore(workspace_id)
    reports = {"checks": checks, "identification": identification}
    return staged.revised(
        effects=staged.effects.revised(
            reports={name: store.write_report(report) for name, report in reports.items()}
        )
    )


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as edit_question does."""
    from nof1_causal_lab.actions.contracts.requests import EditQuestionRequest
    from nof1_causal_lab.actions.edit_question import edit_question
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_lab.study.persistence.history import StudyRepository

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

    from nof1_causal_lab.actions.output_builder import build_output, staged_state
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
    from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
    from nof1_causal_lab.study.persistence.history import StudyRepository
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from nof1_causal_lab.study.records import (
        ActionAttempt,
        Applied,
        AttemptRecord,
        StagedEditAttempt,
        StagedEditQuestionAttempt,
        StagedFitAttempt,
        StagedPrepareAttempt,
        StagedSimulateAttempt,
        applied_attempt,
    )

    if result.result is None and (
        getattr(request, "action", None) == "edit_model"
        or any(artifact.artifact_id == "model" for artifact in result.effects.produced)
    ):
        result = result.revised(result=ModelEditResult())
    if request is not None:
        attempt = applied_attempt(request, result)
    else:
        match result.result:
            case None:
                attempt = StagedEditQuestionAttempt(
                    action="edit_question", request=None, outcome=result
                )
            case ModelEditResult():
                attempt = StagedEditAttempt(action="edit_model", request=None, outcome=result)
            case DataPreparationResult():
                attempt = StagedPrepareAttempt(action="prepare_data", request=None, outcome=result)
            case InferenceEvidence():
                attempt = StagedFitAttempt(
                    action="fit", request=None, outcome=Applied(result=None, effects=result.effects)
                )
            case ModelSimulationResult():
                attempt = StagedSimulateAttempt(
                    action="simulate",
                    request=None,
                    outcome=Applied(result=None, effects=result.effects),
                )
            case _:
                raise TypeError("A publication fixture needs an owned action result")
    repository, store = StudyRepository(workspace_id), ArtifactStore(workspace_id)
    state = staged_state(repository.state(repository.head()), result.effects)
    if not state.has("question"):
        from tests.support.helpers import write_question

        state = state.with_artifacts((write_question(store),))
    if isinstance(result.result, InferenceEvidence):
        assert request is not None
        assert request.action == "fit"
        state = state.revised(data=repository.fit_source(request.input.target).data_ref)
    if isinstance(result.result, DataPreparationResult):
        from nof1_causal_lab.actions.data_checks import evaluate_data_checks

        profile = evaluate_data_checks(workspace_id, state, result)
        attempt = attempt.revised(
            outcome=result.revised(
                effects=result.effects.revised(
                    reports={**result.effects.reports, "data-profile": store.write_report(profile)}
                )
            )
        )
    if (
        isinstance(result.result, ModelSimulationResult)
        and "simulation" not in result.effects.reports
    ):
        from nof1_causal_lab.actions.simulate.action import read_simulation_report

        assert request is not None
        report = read_simulation_report(
            store, result.result.evidence, request.input, state.current["question"].revision
        )
        attempt = attempt.revised(
            outcome=attempt.outcome.revised(
                effects=result.effects.revised(
                    reports={**result.effects.reports, "simulation": store.write_report(report)}
                )
            )
        )
    if isinstance(result.result, InferenceEvidence):
        from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
        from nof1_causal_lab.artifacts.reports.inference import FitCheckReport
        from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
        from nof1_causal_lab.study.persistence.store import read_model
        from tests.support.inference import _report

        assert isinstance(attempt.outcome, Applied)
        reports = dict(attempt.outcome.effects.reports)
        if "inference" not in reports:
            report = _report(read_model(store, state.current["model"].revision)).revised(
                evidence=result.result
            )
            reports["inference"] = store.write_report(report)
        if "checks" not in reports:
            reports["checks"] = store.write_report(
                FitCheckReport(
                    data=DataProfileReport(indicators={}, findings=()),
                    preflight=(),
                    question=QuestionCheckReport(findings=()),
                )
            )
        attempt = attempt.revised(
            outcome=attempt.outcome.revised(
                effects=attempt.outcome.effects.revised(reports=reports),
            )
        )
    assert isinstance(attempt.outcome, Applied)
    if (
        isinstance(result.result, ModelEditResult)
        and "checks" not in attempt.outcome.effects.reports
    ):
        checks, identification = evaluate_model_checks(
            workspace_id, state, result, action="edit_model"
        )
        attempt = attempt.revised(
            outcome=attempt.outcome.revised(
                effects=attempt.outcome.effects.revised(
                    reports={
                        **attempt.outcome.effects.reports,
                        "checks": store.write_report(checks),
                        "identification": store.write_report(identification),
                    }
                )
            )
        )
    output = build_output(workspace_id, attempt, state)
    # Storage-focused fixtures may retain physical input artifacts. The result itself
    # is always the same complete body used by production publication.
    saved = TypeAdapter(ActionAttempt).validate_python(
        {
            "action": attempt.action,
            "request": request,
            "outcome": Applied(
                result=store.write_result(output), effects=result.effects.revised(reports={})
            ),
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
