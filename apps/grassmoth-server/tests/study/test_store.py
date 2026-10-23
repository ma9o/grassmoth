"""Versioned artifact store + transition log."""

import polars as pl
import pytest

from grassmoth.artifacts.observations.preparation import DataPreparationResult
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import (
    Applied,
    AttemptRecord,
    EditAttempt,
    FitAttempt,
    PrepareAttempt,
    Raised,
    Rejected,
)
from grassmoth_server.study.state import RetractedArtifact
from grassmoth_server_test_support.git import artifact_revisions
from grassmoth_test_support.git import git_oid
from grassmoth_test_support.helpers import make_model

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


class TestArtifactStore:
    def test_storage_format_requires_explicit_conversion(self, workspace):
        repository = ArtifactStore(workspace).repo
        assert repository.config.get_int("grassmoth.format") == 30
        repository.config["grassmoth.format"] = 29
        with pytest.raises(ValueError, match="format 30"):
            ArtifactStore(workspace)

    def test_artifacts_are_immutable_git_trees(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"measurement_clock": "1d"}},
        )
        second = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"measurement_clock": "2d"}},
        )
        assert first.revision != second.revision
        assert all(len(info.revision) == 40 for info in (first, second))
        assert not list(__import__("pathlib").Path(store._root).glob("*/v*"))
        assert artifact_revisions(store, "model") == [first.revision, second.revision]
        # Old revision stays readable — nothing is overwritten.
        assert store.read_json_file("model", first.revision, "model.json") == {
            "measurement_clock": "1d"
        }
        assert store.read_json_file("model", second.revision, "model.json") == {
            "measurement_clock": "2d"
        }

    def test_meta_roundtrip(self, workspace):
        store = ArtifactStore(workspace)
        info = store.write_artifact(
            "model",
            derived_from={"raw_data": git_oid(10)},
            produced_by="edit_model",
            json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
        )
        loaded = store.read_meta("model", info.revision)
        assert loaded == info
        assert loaded.derived_from["raw_data"] == git_oid(10)
        assert loaded.created_at

    def test_parquet_payload(self, workspace):
        store = ArtifactStore(workspace)
        df = pl.DataFrame({"indicator": ["mood"], "value": [3.5]})
        info = store.write_artifact(
            "panel",
            derived_from={},
            produced_by="prepare_data",
            parquet_files={"panel.parquet": df},
        )
        loaded_df = store.read_parquet_file("panel", info.revision, "panel.parquet")
        assert loaded_df.equals(df)

    def test_empty_artifact_has_no_versions(self, workspace):
        store = ArtifactStore(workspace)
        assert artifact_revisions(store, "model") == []


class TestStudyRepository:
    def _record(self, workspace, seq, action, outcome=None, **kwargs):
        if outcome is not None:
            variant = {
                "edit_model": EditAttempt,
                "prepare_data": PrepareAttempt,
                "fit": FitAttempt,
            }[action]
            return AttemptRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                attempt=variant(action=action, request=None, outcome=outcome),
            )
        result = Applied(
            result=None if action == "edit_model" else DataPreparationResult(),
            effects=ActionEffects(**kwargs),
        )
        from grassmoth.artifacts.identity import GitOid

        store = ArtifactStore(workspace)
        import msgpack

        identity = GitOid(str(store.repo.create_blob(msgpack.packb({}))))
        variant = {"edit_model": EditAttempt, "prepare_data": PrepareAttempt, "fit": FitAttempt}[
            action
        ]
        return AttemptRecord(
            seq=seq,
            ts="2026-07-03T00:00:00+00:00",
            attempt=variant(
                action=action,
                request=None,
                outcome=Applied(result=identity, effects=result.effects),
            ),
        )

    def test_append_and_read_back_in_order(self, workspace):
        journal = StudyRepository(workspace)
        journal.append(self._record(workspace, 1, "edit_model"))
        journal.append(
            self._record(
                workspace,
                2,
                "edit_model",
                outcome=Rejected(
                    code="MODEL_SPEC_INVALID",
                    subject="model",
                    detail=(
                        "measurement_structure requires artifacts that do not exist: "
                        "raw_data, latent_structure"
                    ),
                ),
            )
        )
        journal.append(
            self._record(
                workspace,
                3,
                "fit",
                outcome=Raised(
                    error_type="ModelFitError",
                    error_message="sampler diverged",
                    details=('{"rhat_max": 2.4}',),
                ),
            )
        )
        records = journal.attempts()
        assert [r.record.seq for r in records] == [1, 2, 3]
        assert records[1].record.attempt.outcome.status == "rejected"
        assert records[1].record.attempt.outcome.detail is not None
        assert "raw_data" in records[1].record.attempt.outcome.detail
        assert records[2].record.attempt.outcome.status == "raised"
        assert records[2].record.attempt.outcome.error_type == "ModelFitError"
        assert records[2].record.attempt.outcome.details == ('{"rhat_max": 2.4}',)
        # Scientific action identifiers round-trip.
        assert records[0].record.attempt.action == "edit_model"
        assert records[2].record.attempt.action == "fit"

    def test_identical_duplicate_seq_is_idempotent(self, workspace):
        journal = StudyRepository(workspace)
        record = self._record(workspace, 1, "edit_model")
        journal.append(record)
        journal.append(record)
        assert [item.record.model_dump() for item in journal.attempts()] == [record.model_dump()]

    def test_different_duplicate_seq_refused(self, workspace):
        journal = StudyRepository(workspace)
        journal.append(self._record(workspace, 1, "edit_model"))
        with pytest.raises(FileExistsError):
            journal.append(self._record(workspace, 1, "prepare_data"))

    def test_latest_seq_reads_max_entry_without_state_manifest(self, workspace):
        journal = StudyRepository(workspace)
        assert journal.latest_seq() == 0
        journal.append(self._record(workspace, 3, "edit_model"))
        assert journal.latest_seq() == 3


class TestDerivedCurrentState:
    def _append(self, workspace, seq, action, *, produced=None, retracted=None, status="applied"):
        if status == "applied":
            result = Applied(
                result=None if action == "edit_model" else DataPreparationResult(),
                effects=ActionEffects(produced=produced or (), retracted=retracted or ()),
            )
            record = TestStudyRepository()._record(
                workspace, seq, action, **result.effects.model_dump()
            )
        else:
            # A failed outcome has no artifact payload, even if staging wrote a tree.
            variant = {"edit_model": EditAttempt, "prepare_data": PrepareAttempt}[action]
            outcome = (
                Rejected(code="MODEL_SPEC_INVALID", subject="model", detail="Saved rejection")
                if status == "rejected"
                else Raised(error_type="SavedError", error_message="Saved failure")
            )
            record = AttemptRecord(
                seq=seq,
                ts="2026-07-03T00:00:00Z",
                attempt=variant(action=action, request=None, outcome=outcome),
            )
        StudyRepository(workspace).append(record)

    def test_current_state_replays_only_applied_versions(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"measurement_clock": "1d"}},
        )
        self._append(
            workspace,
            1,
            "edit_model",
            produced=[first],
        )
        second = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"measurement_clock": "2d"}},
        )

        # Persisting a revision is not the commit boundary. Until an applied
        # transition records it, readers continue to see the prior state.
        assert StudyRepository(workspace).state().get("model") == first

        self._append(
            workspace,
            2,
            "edit_model",
            produced=[second],
        )

        assert StudyRepository(workspace).state().get("model") == second

    def test_rejected_and_raised_effects_are_not_current(self, workspace):
        store = ArtifactStore(workspace)
        rejected = store.write_artifact(
            "model",
            derived_from={},
            produced_by=None,
            json_files={"model.json": {"measurement_clock": "3d"}},
        )
        raised = store.write_artifact(
            "raw_data",
            derived_from={},
            produced_by="prepare_data",
        )
        self._append(
            workspace,
            1,
            "edit_model",
            produced=[rejected],
            status="rejected",
        )
        self._append(
            workspace,
            2,
            "prepare_data",
            produced=[raised],
            status="raised",
        )

        assert StudyRepository(workspace).state().current == {}

    def test_applied_retraction_removes_optional_output(self, workspace):
        store = ArtifactStore(workspace)
        panel_v1 = store.write_artifact(
            "panel",
            derived_from={},
            produced_by="prepare_data",
            parquet_files={"panel.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
        )
        self._append(
            workspace,
            1,
            "prepare_data",
            produced=[panel_v1],
        )

        state_with_panel = StudyRepository(workspace).state()
        assert state_with_panel.get("panel") == panel_v1

        self._append(
            workspace,
            2,
            "prepare_data",
            produced=[],
            retracted=[
                RetractedArtifact(
                    artifact_id="panel",
                    reason_ref="measurements.produces_optional.panel",
                )
            ],
        )

        state_without_panel = StudyRepository(workspace).state()
        assert state_without_panel.get("panel") is None
