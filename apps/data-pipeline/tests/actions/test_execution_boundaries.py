"""Partial models stay inspectable; numerical operations validate their own inputs."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts.bodies import EditModelOutput, FitInput, NewFitTarget
from nof1_causal_lab.actions.contracts.requests import EditModelRequest, FitRequest
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS, GitOid
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec, InferenceHardware
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.state import StudyState
from tests.support.action import applied_record, edit_and_check, question_root
from tests.support.data import metadata_for_model
from tests.support.git import artifact_revision, artifact_revisions
from tests.support.helpers import fixture_entity_id, make_model, run_async, write_question
from tests.support.model import x_y_model

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import backend as storage
    from nof1_causal_lab.storage import paths as data_module

    monkeypatch.setenv("DEPLOYMENT_ENV", "development")
    monkeypatch.setattr(storage, "is_remote", lambda: False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return "EXECUTION"


def test_partial_model_revisions_remain_readable_with_capability_findings(workspace):
    partial = make_model(["X", "Y"], [("X", "Y")])
    complete = x_y_model()
    missing_law = complete.with_entities(
        distributions={},
        parameters=tuple(p.revised(distribution=None) for p in complete.parameters),
    )
    journal = StudyRepository(workspace)
    from nof1_causal_lab.artifacts.model.question import QuestionSpec

    question = QuestionSpec(text="Does X change Y?", outcome=fixture_entity_id("construct", "Y"))
    state = journal.state(question_root(workspace, question).commit_id)
    store = ArtifactStore(workspace)
    saved_results = []
    for revision, model in (
        (1, partial),
        (2, complete),
        (3, missing_law),
    ):
        effects = edit_and_check(
            workspace,
            EditModelRequest[GitOid].model_validate(
                {
                    "input": {
                        "parent_ref": state.current["model"].revision
                        if state.has("model")
                        else state.current["question"].revision,
                        "dynamical_model_spec": model.model_dump(mode="json"),
                    }
                }
            ),
            state,
        )
        state = state.with_artifacts(effects.effects.produced)
        published = journal.append(
            applied_record(
                workspace, effects, seq=revision + 1, ts="2026-09-14T12:00:00Z", trace_ids=[]
            )
        )
        assert published.record.attempt.outcome.status == "applied"
        result_ref = published.record.attempt.outcome.result
        output = store.read_result(result_ref, EditModelOutput)
        assert output.dynamical_model_spec == model
        execution = output.checks.specification[0]
        assert execution.kind == ("evaluated" if revision == 2 else "not_evaluated")
        if revision == 2:
            assert execution.kind == "evaluated"
            assert execution.outcome == "passed"
        assert "execution" not in output.model_dump()
        assert "execution_readiness" not in output.dynamical_model_spec.model_dump()
        saved_results.append((result_ref, output))
    assert "compiled_ssm" not in ARTIFACT_IDS
    for result_ref, output in saved_results:
        assert store.read_result(result_ref, EditModelOutput) == output


def test_incomplete_model_is_rejected_off_loop_before_inference(workspace, monkeypatch):
    from threading import get_ident

    from nof1_causal_lab.actions.fit import action as flow
    from nof1_causal_lab.study.persistence import store as store_module

    event_loop_thread = get_ident()
    read_model = store_module.read_model

    def read_model_off_loop(*args, **kwargs):
        assert get_ident() != event_loop_thread
        return read_model(*args, **kwargs)

    monkeypatch.setattr(store_module, "read_model", read_model_off_loop)

    local = Mock()
    monkeypatch.setattr(flow, "fit", local)
    store = ArtifactStore(workspace)
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )
    state = StudyState().with_artifacts([write_question(store), model, panel])
    with pytest.raises(IncompleteModelError):
        run_async(
            run_action(
                workspace,
                FitRequest[GitOid](
                    input=FitInput[GitOid](
                        target=NewFitTarget(
                            dynamical_model_spec_ref=model.revision,
                            data_ref=DataRef(revision=panel.revision, replicate_index=0),
                        ),
                    )
                ),
                state,
            )
        )
    local.assert_not_called()
    assert artifact_revisions(store, "model") == [model.revision]


@pytest.mark.parametrize("hardware", [InferenceHardware.LOCAL, InferenceHardware.L40S])
def test_refit_after_an_edit_uses_selected_model_and_preserves_the_edit(
    workspace, monkeypatch, hardware
):
    from dataclasses import replace
    from threading import get_ident

    from nof1_causal_lab import config
    from nof1_causal_lab.actions.fit import action as flow
    from nof1_causal_lab.actions.fit import remote
    from nof1_causal_lab.actions.runners import _run_fit
    from nof1_causal_lab.study.persistence.store import read_model
    from nof1_causal_lab.study.queries import data

    event_loop_thread = get_ident()
    read_history = data.read_data_history

    def read_history_off_loop(*args, **kwargs):
        assert get_ident() != event_loop_thread
        return read_history(*args, **kwargs)

    monkeypatch.setattr(data, "read_data_history", read_history_off_loop)

    prior = x_y_model()
    fitted = prior
    edited = fitted.with_entities(
        edges=(fitted.edges[0].revised(description="X changes Y within a day"),)
    )
    store = ArtifactStore(workspace)
    for revision, (model, producer) in enumerate(
        ((prior, "edit_model"), (fitted, "fit"), (edited, None)), 1
    ):
        store.write_artifact(
            "model",
            derived_from={"model": artifact_revision(workspace, "model", revision - 1)}
            if revision > 1
            else {},
            produced_by=producer,
            json_files={"model.json": model.model_dump(mode="json")},
        )
    store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": metadata_for_model(edited).model_dump(mode="json")},
        parquet_files={
            "panel.parquet": pl.DataFrame(
                [
                    {
                        "indicator_id": variable.id,
                        "value": 1.0,
                        "anchor_time": "2024-01-02",
                        "support_start": "2024-01-01",
                        "support_end": "2024-01-02",
                        "support_kind": variable.support_kind.value,
                        "summary_operator": variable.summary_operator.value,
                        "anchor_policy": variable.anchor_policy.value,
                        "observation_window": variable.observation_window.source,
                    }
                    for variable in metadata_for_model(edited).variables
                ]
            )
        },
    )

    import jax.numpy as jnp

    from nof1_causal_lab.actions.fit.conditioning import condition_model
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from tests.support.inference import (
        compile_model_fixture,
        inference_metadata,
        parameter_draws,
        particle_posterior,
    )

    def fit(**kwargs):
        model = kwargs["selection"].dynamical_model_spec
        assert model == edited
        conditioned, _ = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 4), jnp.zeros((4, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            array_writer=store.write_array,
            array_loader=store.read_array,
        )
        return {
            "_dynamical_model_spec": conditioned,
            "evidence": InferenceEvidence(),
            "metadata": inference_metadata(conditioned),
        }

    configured = config.get_config()
    monkeypatch.setattr(
        config,
        "get_config",
        lambda: replace(
            configured,
            inference=replace(
                configured.inference,
                sampler=configured.inference.sampler.revised(hardware=hardware),
            ),
        ),
    )
    local_fit = Mock(side_effect=fit)
    modal_fit = Mock(side_effect=lambda _hardware, **kwargs: fit(**kwargs))
    monkeypatch.setattr(flow, "fit", local_fit)
    monkeypatch.setattr(remote, "fit_on_modal", modal_fit)
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision(workspace, "model", 3),
        "panel": artifact_revision(workspace, "panel", 1),
        "question": write_question(store).revision,
    }
    effects = run_async(
        _run_fit(
            store,
            pins,
            FitInput(
                target=NewFitTarget(
                    dynamical_model_spec_ref=pins["model"],
                    data_ref=DataRef[GitOid, int](revision=pins["panel"], replicate_index=0),
                ),
                settings=FitSettingsSpec(),
            ),
        )
    )
    assert effects.effects.produced[0].derived_from == pins
    if hardware is InferenceHardware.LOCAL:
        local_fit.assert_called_once()
        modal_fit.assert_not_called()
    else:
        local_fit.assert_not_called()
        modal_fit.assert_called_once()
        assert modal_fit.call_args.args == (hardware,)
    assert (
        read_model(store, effects.effects.produced[0].revision).edges[0].description
        == edited.edges[0].description
    )
