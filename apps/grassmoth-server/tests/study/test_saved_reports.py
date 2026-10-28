"""Action findings survive cache eviction and are never evaluated by result readers."""

import asyncio
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import get_ident

import numpy as np
import polars as pl
import pygit2
import pytest

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import DistributionId, GitOid
from grassmoth.artifacts.model.dynamical_model_spec import ModelEditResult
from grassmoth.artifacts.observations.preparation import DataPreparationResult
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.reports.checks import Evaluated
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.artifacts.reports.inference import (
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from grassmoth.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport
from grassmoth.artifacts.reports.posterior_diagnostics import ConvergenceReport
from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from grassmoth.artifacts.simulation import (
    SimulationArm,
    SimulationEvidence,
    SimulationReport,
    SimulationSpec,
    SingleArmSimulation,
)
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import (
    EditModelInput,
    FitInput,
    NewFitTarget,
    PrepareDataInput,
    SimulateInput,
)
from grassmoth_server.actions.contracts.preparation import FileSourceRef
from grassmoth_server.actions.contracts.requests import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from grassmoth_server.actions.temporal.study.activities import (
    evaluate_data_checks_activity,
    evaluate_model_checks_activity,
    journal_activity,
)
from grassmoth_server.actions.temporal.study.messages import (
    AttemptPublication,
    EvaluateChecksInput,
)
from grassmoth_server.storage import paths as data
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.queries.lineage import fitted_law_report
from grassmoth_server.study.records import Applied
from grassmoth_server_test_support.action import (
    applied_record,
    empty_simulation_summary,
    question_root,
)
from grassmoth_server_test_support.runner_fixtures import panel_frame, panel_metadata, seed_model
from grassmoth_test_support.model import x_y_model

pytestmark = pytest.mark.contract


def test_action_reports_are_published_once_and_loaded_without_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    workspace = "SAVED"
    question_root(workspace)
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={
            "panel.parquet": panel_frame(n_days=2).with_columns(
                pl.col("anchor_time", "support_start", "support_end").str.to_datetime(
                    time_zone="UTC"
                )
            )
        },
    )
    profile = DataProfileReport(indicators={}, findings=())

    def publish(applied, *, request=None, model_checks=None, data_profile=None):
        assert request is not None
        reports = dict(applied.effects.reports)
        if model_checks is not None:
            if isinstance(model_checks, tuple):
                reports.update(
                    checks=store.write_result(model_checks[0]),
                    identification=store.write_result(model_checks[1]),
                )
            else:
                reports["checks"] = store.write_result(model_checks)
        if data_profile is not None:
            reports["data-profile"] = store.write_result(data_profile)
        record = applied_record(
            workspace,
            applied.revised(effects=applied.effects.revised(reports=reports)),
            seq=repository.latest_seq() + 1,
            request=request,
        )
        evaluated = None
        assert isinstance(record.attempt.outcome, Applied)
        if model_checks is not None or data_profile is not None:
            activity_input = EvaluateChecksInput(
                workspace_id=workspace,
                state=repository.state(),
                applied=record.attempt.outcome,
                request=request,
            )
            evaluated = asyncio.run(
                evaluate_data_checks_activity(activity_input)
                if data_profile is not None
                else evaluate_model_checks_activity(activity_input)
            )
            assert len(evaluated.model_dump_json().encode()) < 10_000
            record = record.revised(messages=evaluated.messages)
        publication = AttemptPublication(
            workspace_id=workspace,
            record=record,
        )
        assert len(publication.model_dump_json().encode()) < 20_000
        from grassmoth_server.study.queries import results

        read_result = results.read_result
        event_loop_thread = get_ident()

        def read_off_loop(*args, **kwargs):
            assert get_ident() != event_loop_thread
            return read_result(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(results, "read_result", read_off_loop)
            return publication, asyncio.run(journal_activity(publication))

    _, prepared = publish(
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=(panel,))),
        request=PrepareDataRequest[GitOid, FileSourceRef](
            input=PrepareDataInput[GitOid, FileSourceRef](
                dynamical_model_spec_ref=seed_model(store).revision,
                source=FileSourceRef(files=("input/test.csv",), hashes={}),
                extraction={
                    variable.observation.id: variable.extraction
                    for variable in panel_metadata().preparation.variables
                },
            )
        ),
        data_profile=profile,
    )
    dynamical_model_spec = x_y_model()
    authored = store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={"question": repository.question().revision},
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json", round_trip=True)},
    )
    checks = ModelCheckReport(
        specification=(
            Evaluated(
                subject="model_execution",
                outcome="passed",
                evidence="Saved" * 500_000,
                code="model_execution",
            ),
        ),
        question=QuestionCheckReport(
            findings=(),
        ),
    )
    assert checks.question is not None
    edit_checks = checks

    identification = IdentificationReport(outcome=None)
    from grassmoth.artifacts.reports.inference import FitCheckReport

    validation = FitCheckReport(
        question=QuestionCheckReport(findings=()),
        data=profile,
        preflight=(
            Evaluated(
                subject="fit_preflight",
                outcome="failed",
                evidence="Saved failure",
                code="fit_preflight",
            ),
        ),
    )
    _, edited = publish(
        Applied(result=ModelEditResult(), effects=ActionEffects(produced=(authored,))),
        request=EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=repository.question().revision,
                dynamical_model_spec=dynamical_model_spec,
            )
        ),
        model_checks=(edit_checks, identification),
    )
    fitted = store.write_artifact(
        "model",
        produced_by="fit",
        derived_from={"model": authored.revision, "panel": panel.revision},
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json", round_trip=True)},
    )
    from grassmoth.artifacts.reports.posterior_diagnostics import MCMCEvidence

    initial_delta = NumericalArray.from_numpy(np.asarray([[1.0, 2.0]]))
    divergent = NumericalArray.from_numpy(np.asarray([False, True]))
    evidence = InferenceEvidence(
        initial_latent_delta=initial_delta,
        chain_extra_fields={"diverging": divergent},
    )
    inference = InferenceReport(
        evidence=evidence,
        core=InferenceReportCore(
            inference_metadata=InferenceMetadata(
                distribution=DistributionId("distribution:retained"),
                num_samples_total=2,
                num_chains=1,
                duration_seconds=3,
                engine=MCMCEvidence(),
                sampler_diagnostics=None,
            ),
            inference_diagnostics=None,
            convergence=ConvergenceReport(findings=()),
            posterior_marginals=(),
            prior_densities={},
        ),
        detail=InferenceReportDetail(),
    )

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("Saved result reads and publication must not run checks")

    monkeypatch.setattr("grassmoth.actions.fit.action.inference_report", unexpected_evaluation)
    fit_publication, fit_revision = publish(
        Applied(
            result=evidence,
            effects=ActionEffects(
                produced=(fitted,),
                reports={
                    "inference": store.write_result(inference),
                    "checks": store.write_result(validation),
                },
            ),
        ),
        request=FitRequest[GitOid](
            input=FitInput[GitOid](
                target=NewFitTarget(
                    dynamical_model_spec_ref=authored.revision,
                    data_ref=DataRef(revision=panel.revision, replicate_index=0),
                ),
            )
        ),
        model_checks=validation,
    )
    zeroes = NumericalArray.from_numpy(np.zeros((2, 2, 0)))
    assert dynamical_model_spec.measurement_clock is not None
    variables = tuple(
        item.observation.resolved(dynamical_model_spec.measurement_clock)
        for item in dynamical_model_spec.indicators
    )
    support_ends = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(variables)))
    observations = np.zeros((2, 2, len(variables)))
    observations[1] = 7
    observations[1, 0, 0] = np.inf
    mask = np.ones_like(observations, dtype=bool)
    mask[1, 1, 0] = False
    evidence = SimulationEvidence(
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 1),
        draws=2,
        seed=0,
        state_ids=(),
        parameter_draws={},
        arms=SingleArmSimulation(
            action=SimulationArm(
                latent_paths=zeroes, observations=NumericalArray.from_numpy(observations)
            ),
        ),
        observation_layout={
            "variables": variables,
            "support_start_times": NumericalArray.from_numpy(support_ends - 1),
            "support_end_times": NumericalArray.from_numpy(support_ends),
            "mask": NumericalArray.from_numpy(mask),
        },
        assignments=SimulationSpec(start="2026-01-01", horizon="1d").assignments(
            datetime(2026, 1, 1, tzinfo=UTC)
        ),
    )
    simulation = SimulationReport(
        summary=empty_simulation_summary(),
        evidence=evidence,
        law=AuthoredLawProvenance(),
        fit_reliability="not_fitted",
    )
    monkeypatch.setattr(
        "grassmoth.actions.simulate.action.simulation_report", unexpected_evaluation
    )
    simulation_publication, simulated = publish(
        Applied(
            result=evidence,
            effects=ActionEffects(reports={"simulation": store.write_result(simulation)}),
        ),
        request=SimulateRequest[GitOid](
            input=SimulateInput[GitOid](
                simulation=SimulationSpec(start="2026-01-01", horizon="1d"),
                dynamical_model_spec_ref=fitted.revision,
            )
        ),
    )
    cache = Path(data.cache_dir(workspace))
    cache.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cache)
    monkeypatch.setattr(
        "grassmoth_server.actions.output_builder.complete_attempt", unexpected_evaluation
    )

    for owner in (
        "checks.model_check_report",
        "prepare_data.action.prepare_data",
        "fit.action.inference_report",
        "simulate.action.simulation_report",
    ):
        monkeypatch.setattr(f"grassmoth.actions.{owner}", unexpected_evaluation)
    assert asyncio.run(journal_activity(fit_publication)) == fit_revision
    assert asyncio.run(journal_activity(simulation_publication)) == simulated
    from pydantic import TypeAdapter, ValidationError

    from grassmoth_server.actions.contracts.polling import ActionPoll
    from grassmoth_server.study.persistence.codec import unpack_result
    from grassmoth_server.study.queries.action_outputs import completed_call_msgpack

    def completed_call(workspace_id, revision):
        return TypeAdapter(ActionPoll).validate_python(
            unpack_result(completed_call_msgpack(workspace_id, revision))
        )

    prepared_response = completed_call(workspace, prepared)
    assert prepared_response.status == "success"
    assert prepared_response.action == "prepare_data"
    assert prepared_response.body.profile == profile
    edited_response = completed_call(workspace, edited)
    assert edited_response.status == "success"
    assert edited_response.action == "edit_model"
    assert edited_response.body.checks == edit_checks
    assert edited_response.body.identification == identification
    fitted_response = completed_call(workspace, fit_revision)
    assert fitted_response.status == "success"
    assert fitted_response.action == "fit"
    assert fitted_response.body.checks.question == checks.question
    assert fitted_response.body.checks == validation
    saved_inference = fitted_response.body.inference
    assert saved_inference == inference
    assert saved_inference.evidence.initial_latent_delta == initial_delta
    assert saved_inference.evidence.chain_extra_fields["diverging"] == divergent
    np.testing.assert_array_equal(initial_delta.values, [[1.0, 2.0]])
    np.testing.assert_array_equal(divergent.values, [False, True])
    assert set(fitted_response.body.model_dump()) == {
        "dynamical_model_spec",
        "checks",
        "inference",
        "profile",
    }
    assert fitted_response.body.revised(dynamical_model_spec=None).dynamical_model_spec is None
    for field in ("checks", "inference"):
        with pytest.raises(ValidationError):
            fitted_response.body.revised(**{field: None})
    assert (
        fitted_law_report(store, repository.records(simulated.commit_id), fitted.revision)
        == inference.core
    )
    response = completed_call(workspace, simulated)
    assert response.status == "success"
    assert response.action == "simulate"
    assert response.body.report == simulation
    saved_simulation = response.body.report
    assert set(response.body.model_dump()) == {"report", "data"}
    assert isinstance(prepared_response.body.data, Mapping)
    prepared_history = prepared_response.body.data[panel_metadata().variables[0].id]
    assert isinstance(response.body.data, tuple)
    assert len(response.body.data) == 2
    from grassmoth.artifacts.arrays import resolve_vector

    def values(value):
        return resolve_vector(value)

    first, second = response.body.data
    assert type(first[variables[0].id]) is type(prepared_history)
    assert values(first[variables[0].id].values) == (0.0, 0.0)
    assert first[variables[0].id].times == (0.0, 1.0)
    assert values(first[variables[0].id].support_start) == (-1.0, 0.0)
    assert values(second[variables[0].id].values) == (None, None)
    assert values(second[variables[1].id].values) == (7.0, 7.0)
    np.testing.assert_array_equal(
        saved_simulation.evidence.arms.action.latent_paths.values, np.zeros((2, 2, 0))
    )
    assert (
        repository.read_file(fit_revision.commit_id, "result.msgpack")
        == store.repo[pygit2.Oid(hex=fit_revision.record.attempt.outcome.result)]
        .peel(pygit2.Blob)
        .data
    )
    assert all(record.record.trace_ids == () for record in repository.attempts())
