"""Git publication and commit-local evidence across process restarts."""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult, FileSourceRef
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.simulation import (
    SimulationArm,
    SimulationEvidence,
    SingleArmSimulation,
)
from nof1_causal_service.actions.contracts.effects import ActionEffects
from nof1_causal_service.actions.contracts.inputs import (
    DataDiffInput,
    EditModelInput,
    FitInput,
    ModelDiffInput,
    SimulateInput,
)
from nof1_causal_service.actions.contracts.requests import EditModelRequest, PrepareDataRequest
from nof1_causal_service.storage import paths as data_module
from nof1_causal_service.study.errors import StudyLookupError
from nof1_causal_service.study.persistence.history import StudyRepository
from nof1_causal_service.study.persistence.store import ArtifactStore
from nof1_causal_service.study.records import (
    Applied,
    AttemptRecord,
    PrepareAttempt,
    Raised,
    Rejected,
)
from tests.support.action import applied_record, empty_simulation_summary

pytestmark = pytest.mark.contract


@pytest.fixture
def study(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return StudyRepository("STUDY"), ArtifactStore("STUDY")


def _record(seq, *, outcome=None, attempt_id=None, **effects):
    if outcome is not None:
        return AttemptRecord(
            seq=seq,
            attempt_id=attempt_id,
            ts="2026-09-23T12:00:00+00:00",
            attempt=PrepareAttempt(request=None, outcome=outcome, action="prepare_data"),
        )
    return applied_record(
        "STUDY",
        Applied(result=None, effects=ActionEffects(**effects)),
        seq=seq,
        attempt_id=attempt_id,
        ts="2026-09-23T12:00:00+00:00",
    )


def _model(store, clock, parent=0):
    return store.write_artifact(
        "model",
        produced_by=None,
        derived_from={"model": parent} if parent else {},
        json_files={"model.json": {"measurement_clock": clock}},
    )


def test_failed_attempt_and_invalid_publication_do_not_advance_scientific_state(study):
    repository, _ = study
    attempt_id = uuid4()
    accepted = _record(1, attempt_id=attempt_id)
    head = repository.append(accepted).commit_id
    assert repository.append(accepted).commit_id == head
    persisted = StudyRepository("STUDY")
    assert str(persisted.repo.references[f"refs/actions/{attempt_id}"].target) == head
    with pytest.raises(FileExistsError):
        repository.append(_record(2, attempt_id=attempt_id))
    assert repository.read_attempt(2) is None
    failed = repository.append(
        _record(2, outcome=Raised(error_type="SavedError", error_message="failure")),
    ).commit_id
    assert repository.head() == head
    assert repository.record(failed).record.attempt.outcome.status == "raised"
    with pytest.raises(StudyLookupError):
        repository.resolve(at=failed)
    assert [record.record.seq for record in StudyRepository("STUDY").attempts()] == [1, 2]


def test_out_of_order_publication_selects_submission_order_and_retains_input_parents(study):
    from tests.support.action import question_root

    repository, store = study
    root = question_root("STUDY")
    earlier, later = _model(store, "1d"), _model(store, "2d")
    records = []
    for seq, model, clock in ((2, earlier, "1d"), (3, later, "2d")):
        record = _record(seq, produced=(model,))
        records.append(
            record.revised(
                attempt=record.attempt.revised(
                    request=EditModelRequest[GitOid](
                        input=EditModelInput[GitOid](
                            parent_ref=repository.question().revision,
                            dynamical_model_spec=DynamicalModelSpec(measurement_clock=clock),
                        )
                    )
                )
            )
        )
    first, second = records
    published_second = repository.append(second)
    published_first = repository.append(first)
    assert repository.head() == published_second.commit_id
    assert repository.state().current["model"] == later
    assert repository.state(published_first.commit_id).current["model"] == earlier
    assert [item.record.seq for item in repository.records()] == [1, 2, 3]
    for publication in (published_first, published_second):
        assert publication.parent_ids == (root.commit_id,)
        assert "artifacts" not in repository.repo[publication.commit_id].tree
    assert "refs/heads/main" not in repository.repo.references

    from nof1_causal_service.actions.contracts.requests import DataDiffRequest
    from nof1_causal_service.study.records import failed_attempt
    from tests.support.git import git_oid

    compared = repository.append(
        AttemptRecord(
            seq=4,
            ts="2026-10-21T00:00:00Z",
            attempt=failed_attempt(
                DataDiffRequest[GitOid](
                    input=DataDiffInput[GitOid](
                        left_ref=tuple(
                            DataRef[GitOid, int](revision=identity, replicate_index=0)
                            for identity in (later.revision, earlier.revision, later.revision)
                        ),
                        right_ref=(
                            DataRef[GitOid, int](revision=root.commit_id, replicate_index=0),
                            DataRef[GitOid, int](revision=git_oid(999), replicate_index=0),
                        ),
                    )
                ),
                Rejected(
                    code="INPUT_UNAVAILABLE", subject="inputs", detail="Not observation inputs"
                ),
            ),
        )
    )
    assert compared.parent_ids == (
        published_second.commit_id,
        published_first.commit_id,
        root.commit_id,
    )


def test_question_publication_admits_one_call_and_reuses_its_result(study):
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from tests.support.action import question_root

    repository, _ = study
    first = question_root("STUDY", QuestionSpec(text="First?", outcome="construct:outcome"))
    second = question_root("STUDY", QuestionSpec(text="Second?", outcome="construct:outcome"))
    assert isinstance(first.record.attempt.outcome, Applied)
    assert first.parent_ids == ()
    assert isinstance(second.record.attempt.outcome, Rejected)
    assert second.record.messages[-1].code == "INPUT_UNAVAILABLE"
    assert str(repository.repo.references["refs/question"].target) == first.commit_id
    assert (
        repository.state().current["question"] == first.record.attempt.outcome.effects.produced[0]
    )
    assert question_root("STUDY", QuestionSpec(text="First?", outcome="construct:outcome")) == first


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("comparison_action", ["data_diff", "model_diff"])
def test_comparison_is_a_saved_leaf_of_the_journal(study, monkeypatch, fails, comparison_action):
    from temporalio.exceptions import ActivityError, ApplicationError

    from nof1_causal_lab.actions.contracts.bodies import DataDiffOutput, ModelDiffOutput
    from nof1_causal_lab.artifacts.identity import IndicatorRef
    from nof1_causal_lab.artifacts.reports.checks import Evaluated
    from nof1_causal_lab.artifacts.reports.data_comparison import (
        DataComparisonReport,
        DescriptiveIndicatorComparison,
    )
    from nof1_causal_service.actions import data_diff
    from nof1_causal_service.actions import model_diff as revisions
    from nof1_causal_service.actions.contracts.requests import DataDiffRequest, ModelDiffRequest
    from nof1_causal_service.actions.temporal.study import workflow as study_workflow
    from nof1_causal_service.actions.temporal.study.activities import (
        journal_activity,
        read_inputs_activity,
        run_action_activity,
    )
    from nof1_causal_service.actions.temporal.study.messages import ActionRequest, StudyInit
    from tests.support.helpers import run_async

    repository, store = study
    from tests.support.action import question_root

    root = question_root("STUDY").commit_id
    head = repository.append(
        applied_record(
            "STUDY",
            Applied(result=None, effects=ActionEffects(produced=(_model(store, "2d"),))),
            seq=2,
        )
    ).commit_id
    left = DataRef[GitOid, int](replicate_index=0, revision=root)
    right = DataRef[GitOid, int](replicate_index=0, revision=head)
    intent = "Compare the candidates before choosing the next fit."
    request = (
        DataDiffRequest[GitOid](
            input=DataDiffInput[GitOid](left_ref=(left,), right_ref=(right,)), reasoning=intent
        )
        if comparison_action == "data_diff"
        else ModelDiffRequest[GitOid](
            reasoning=intent, input=ModelDiffInput[GitOid](before_ref=root, after_ref=head)
        )
    )
    finding = Evaluated(
        code="schedule_mismatch",
        subject=IndicatorRef(id="indicator:sleep"),
        outcome="failed",
        evidence="The saved histories use different observation schedules.",
    )
    report = (
        DataDiffOutput(
            report=DataComparisonReport(
                left=(left,),
                right=(right,),
                variables=(
                    DescriptiveIndicatorComparison(
                        indicator_id="indicator:sleep",
                        changes=(),
                        statistics=(),
                        findings=(finding,),
                    ),
                ),
            )
        )
        if comparison_action == "data_diff"
        else ModelDiffOutput(changes=DynamicalModelSpec())
    )

    reads = []

    def compare(workspace, *selection):
        reads.append((workspace, selection))
        if fails:
            raise StudyLookupError("The selected panel has no saved observations")
        return report

    async def execute(name, payload, **kwargs):
        name = name if isinstance(name, str) else name.__name__
        if name == "read_inputs_activity":
            return await read_inputs_activity(payload)
        if name == "run_action_activity":
            return await run_action_activity(payload)
        if name == "complete_result_activity":
            from nof1_causal_service.actions.temporal.study.activities import (
                complete_result_activity,
            )

            return await complete_result_activity(payload)
        if name == "journal_activity":
            try:
                # A journal activity retry reads the saved report rather than comparing again.
                commit = await journal_activity(payload)
                assert await journal_activity(payload) == commit
                return commit
            except ApplicationError as exc:
                raise ActivityError(
                    "comparison failed",
                    scheduled_event_id=1,
                    started_event_id=2,
                    identity="test",
                    activity_type=name,
                    activity_id="test",
                    retry_state=None,
                ) from exc
        if name == "collect_completed_runs_activity":
            return None
        raise AssertionError(f"A comparison must not execute scientific actions: {name}")

    monkeypatch.setattr(
        data_diff if comparison_action == "data_diff" else revisions,
        "read_data_diff" if comparison_action == "data_diff" else "model_diff",
        compare,
    )
    monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 9, 30, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    worker = study_workflow.StudyWorkflow(StudyInit(workspace_id="STUDY", initial_seq=2))
    action = ActionRequest(request=request)
    run_async(study_workflow.StudyWorkflow.execute_action(worker, action))
    leaf = repository.read_attempt(3)
    assert leaf is not None
    assert leaf.parent_ids == (root, head)
    assert "artifacts" not in repository.repo[leaf.commit_id].tree
    assert leaf.record.attempt.action == comparison_action
    assert leaf.record.attempt.request == request
    outcome = leaf.record.attempt.outcome
    assert outcome.status == ("rejected" if fails else "applied")
    assert repository.state(leaf.commit_id) == repository.state(head)
    assert repository.head() == head
    assert len(reads) == 1
    if fails:
        assert isinstance(outcome, Rejected)
        assert "no saved observations" in outcome.detail
    else:
        assert isinstance(outcome, Applied)
        assert len(outcome.result) == 40
        assert outcome.effects.produced == outcome.effects.retracted == ()
        assert leaf.record.attempt.request == request
        assert outcome.effects.reports == {}
        if comparison_action == "data_diff":
            announcements = tuple(
                message for message in leaf.record.messages if message.code == finding.code
            )
            assert len(announcements) == 1
            assert announcements[0].severity == "warning"
            assert announcements[0].subject == finding.subject
            assert announcements[0].detail == finding.evidence
        from nof1_causal_service.study.persistence.codec import unpack_result

        assert (
            type(report).model_validate(
                unpack_result(repository.read_file(leaf.commit_id, "result.msgpack"))
            )
            == report
        )

        # Replaying the call after cache loss and a code change reads its Git blob.
        cache = Path(data_module.cache_dir("STUDY"))
        cache.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache)

        def unexpected_comparison(*_args, **_kwargs):
            pytest.fail("A saved comparison must not recompute")

        monkeypatch.setattr(data_diff, "read_data_diff", unexpected_comparison)
        monkeypatch.setattr(revisions, "model_diff", unexpected_comparison)
        restarted = StudyRepository("STUDY")
        assert restarted.read_file(leaf.commit_id, "result.msgpack") == repository.read_file(
            leaf.commit_id, "result.msgpack"
        )
        from nof1_causal_service.api.app import app

        monkeypatch.setenv("READ_ONLY_FACADE", "1")
        with TestClient(app) as client:
            response = client.post(
                f"/api/studies/STUDY/{comparison_action}",
                json=request.model_dump(mode="json"),
            )
        assert response.status_code == 200, response.text
        assert unpack_result(response.content)["body"] == report.model_dump(mode="json")
        assert len(restarted.attempts()) == 3
        repeated = run_async(
            study_workflow.StudyWorkflow.execute_action(
                worker, ActionRequest(request=request.revised(reasoning=None))
            )
        )
        assert repeated.commit_id == leaf.commit_id
        assert len(reads) == 1


def test_timeline_links_only_declared_refs_to_published_outputs():
    from nof1_causal_service.study.records import StudyRevision, record_dependencies
    from nof1_causal_service.study.state import ArtifactRecord

    def oid(n):
        return f"{n:040x}"

    from datetime import date

    from nof1_causal_lab.artifacts.simulation import (
        SimulationReport,
        SimulationSpec,
    )
    from nof1_causal_service.actions.contracts.requests import (
        DataDiffRequest,
        FitRequest,
        ModelDiffRequest,
        SimulateRequest,
    )
    from nof1_causal_service.study.records import FitAttempt

    def dependency_record(applied, *, seq, request=None, **metadata):
        from nof1_causal_service.study.records import applied_attempt

        assert request is not None
        return AttemptRecord(
            seq=seq,
            ts="2026-10-01T00:00:00Z",
            attempt=applied_attempt(
                request, Applied(result=oid(500 + seq), effects=applied.effects)
            ),
        )

    def revision(seq, action, inputs, *produced, status="applied"):
        artifacts = tuple(
            ArtifactRecord(
                artifact_id=artifact_id,
                revision=oid(n),
                derived_from={key: oid(value) for key, value in derived.items()},
            )
            for artifact_id, n, derived in produced
        )
        if action == "edit_model":
            from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult

            result = Applied(
                result=ModelEditResult(),
                effects=ActionEffects(produced=artifacts),
            )
            record = dependency_record(
                result,
                seq=seq,
                request=EditModelRequest[GitOid](
                    input=EditModelInput[GitOid](
                        parent_ref=inputs["parent_ref"],
                        dynamical_model_spec=DynamicalModelSpec(),
                    )
                ),
            )
        elif action == "prepare_data":
            record = dependency_record(
                Applied(
                    result=DataPreparationResult(),
                    effects=ActionEffects(produced=artifacts),
                ),
                seq=seq,
                request=PrepareDataRequest[GitOid, FileSourceRef](input=inputs["input"]),
            )
        elif action == "simulate":
            ref = GitRef(
                workspace_id="STUDY", revision=inputs["dynamical_model_spec_ref"], path="model.json"
            )
            report = SimulationReport(
                summary=empty_simulation_summary(),
                fit_reliability="not_fitted",
                law=AuthoredLawProvenance(),
                evidence=SimulationEvidence(
                    time_origin=datetime(2026, 1, 1, tzinfo=UTC),
                    times=(0, 10),
                    draws=1,
                    seed=0,
                    state_ids=(),
                    parameter_draws={},
                    arms=SingleArmSimulation(
                        action=SimulationArm(
                            latent_paths=NumericalArray.from_numpy(np.zeros((1, 2, 0))),
                            observations=NumericalArray.from_numpy(np.zeros((1, 2, 0))),
                        ),
                    ),
                    observation_layout={
                        "variables": [],
                        "support_start_times": NumericalArray.from_numpy(np.zeros((2, 0))),
                        "support_end_times": NumericalArray.from_numpy(np.zeros((2, 0))),
                        "mask": NumericalArray.from_numpy(np.zeros((1, 2, 0), dtype=bool)),
                    },
                    assignments=SimulationSpec(start=date(2026, 1, 1), horizon="10d").assignments(
                        datetime(2026, 1, 1, tzinfo=UTC)
                    ),
                ),
            )
            record = dependency_record(
                Applied(
                    result=(report).evidence,
                    effects=ActionEffects(),
                ),
                seq=seq,
                request=SimulateRequest[GitOid](
                    input=SimulateInput[GitOid](
                        simulation=SimulationSpec(start=date(2026, 1, 1), horizon="10d"),
                        dynamical_model_spec_ref=ref.revision,
                    )
                ),
            )
        elif action == "fit":
            request = FitRequest[GitOid](
                input=FitInput[GitOid].model_validate(
                    {
                        "target": {
                            **inputs["target"],
                            "data_ref": {
                                "revision": inputs["target"]["data_ref"],
                                "replicate_index": 0,
                            },
                        }
                    }
                )
            )
            if status == "raised":
                record = AttemptRecord(
                    seq=seq,
                    ts="2026-10-01T00:00:00Z",
                    attempt=FitAttempt(
                        action="fit",
                        request=request,
                        outcome=Raised(error_type="WorkerError", error_message="failed"),
                    ),
                )
            else:
                from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence

                result = InferenceEvidence()
                record = dependency_record(
                    Applied(result=result, effects=ActionEffects(produced=artifacts)),
                    seq=seq,
                    request=request,
                )
        else:
            request = (
                ModelDiffRequest[GitOid](input=ModelDiffInput[GitOid](**inputs))
                if action == "model_diff"
                else DataDiffRequest[GitOid](input=DataDiffInput[GitOid](**inputs))
            )
            record = dependency_record(
                Applied(
                    result=None,
                    effects=ActionEffects(),
                ),
                seq=seq,
                request=request,
            )
        return StudyRevision(record=record, commit_id=oid(100 + seq), parent_ids=(oid(99 + seq),))

    records = [
        revision(1, "edit_model", {"parent_ref": oid(0)}, ("model", 1, {})),
        revision(2, "simulate", {"dynamical_model_spec_ref": oid(1)}),
        revision(
            3,
            "prepare_data",
            {
                "input": {
                    "dynamical_model_spec_ref": oid(1),
                    "source": {"files": ["data.csv"]},
                    "extraction": {
                        "indicator:00000000000000000000": {
                            "kind": "semantic",
                            "how_to_measure": "Read observations",
                        }
                    },
                }
            },
            ("raw_data", 3, {}),
            ("panel", 4, {"raw_data": 3}),
        ),
        revision(
            4,
            "fit",
            {"target": {"kind": "new", "dynamical_model_spec_ref": oid(1), "data_ref": oid(4)}},
            ("model", 5, {"model": 1, "panel": 4}),
        ),
        revision(
            5,
            "edit_model",
            {"parent_ref": oid(5)},
            ("model", 6, {"model": 5, "panel": 4}),
            # Carried forward unchanged: its first producer keeps it.
            ("panel", 4, {"raw_data": 3}),
        ),
        revision(
            6,
            "fit",
            {"target": {"kind": "new", "dynamical_model_spec_ref": oid(6), "data_ref": oid(4)}},
            status="raised",
        ),
        revision(8, "edit_model", {"parent_ref": oid(6)}, ("model", 8, {"model": 6})),
        revision(
            7,
            "data_diff",
            {
                "left_ref": [
                    {"revision": oid(4), "replicate_index": 0},
                    {"revision": oid(4), "replicate_index": 0},
                    {"revision": oid(102), "replicate_index": 0},
                ],
                "right_ref": [{"revision": oid(102), "replicate_index": None}],
            },
        ),
    ]
    records.extend(
        [
            # Failed commits and unknown revisions are never output producers.
            revision(
                9,
                "fit",
                {
                    "target": {
                        "kind": "new",
                        "dynamical_model_spec_ref": oid(106),
                        "data_ref": oid(999),
                    }
                },
                status="raised",
            ),
            revision(10, "model_diff", {"before_ref": oid(5), "after_ref": oid(6)}),
        ]
    )
    assert [
        (item.seq, item.source_seq, item.argument) for item in record_dependencies(records)
    ] == [
        (2, 1, "dynamical_model_spec"),
        (3, 1, "dynamical_model_spec"),
        (4, 1, "dynamical_model_spec"),
        (4, 3, "data"),
        (5, 4, "parent"),
        (6, 5, "dynamical_model_spec"),
        (6, 3, "data"),
        (8, 5, "parent"),
        (7, 3, "left"),
        (7, 2, "left"),
        (7, 2, "right"),
        (10, 4, "before"),
        (10, 5, "after"),
    ]
