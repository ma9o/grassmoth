"""Pinned external inputs, action revisions and reproducible observation masking."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from evaluation.client import ActionClient
from evaluation.declarations import CaseFile, StudyModel
from evaluation.fixtures.recorded import ObservationMask, mask_csv
from evaluation.sources import model_document, read_source
from grassmoth.artifacts.identity import GitOid

pytestmark = pytest.mark.contract


def test_case_sources_pin_file_bytes_and_saved_action_models(tmp_path, monkeypatch):
    payload = b'{"measurement_clock": "3h"}'
    (tmp_path / "model.json").write_bytes(payload)
    source = CaseFile(path="model.json", sha256=hashlib.sha256(payload).hexdigest())
    document = model_document(read_source(source, tmp_path))
    assert str(document.measurement_clock) == "3h"
    (tmp_path / "model.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_source(source, tmp_path)

    from grassmoth.actions.contracts.bodies import EditModelOutput
    from grassmoth_server.actions.contracts.identity import CallId
    from grassmoth_server.actions.contracts.polling import SuccessfulPoll
    from grassmoth_server.public.contracts import ActionResult

    call_id = CallId("call:" + "1" * 64)
    stored = ActionResult.from_poll(
        SuccessfulPoll(
            action="edit_model",
            call_id=call_id,
            commit_id=GitOid("2" * 40),
            body=EditModelOutput.model_validate(
                {
                    "dynamical_model_spec": {"measurement_clock": "3h"},
                    "checks": {"specification": [], "question": {"findings": []}},
                    "identification": {"outcome": None},
                    "pruning": {},
                }
            ),
            messages=(),
        )
    )
    urls = []

    def get(request, **_kwargs):
        urls.append(request.full_url)
        return io.BytesIO(stored.payload)

    monkeypatch.setattr("grassmoth_server.public.client.urlopen", get)
    external = StudyModel(
        endpoint="http://example.invalid",
        workspace="SOURCE",
        call_id=call_id,
        sha256=hashlib.sha256(document.model_dump_json(round_trip=True).encode()).hexdigest(),
    )
    assert model_document(read_source(external, tmp_path)) == document
    assert urls == [f"http://example.invalid/api/studies/SOURCE/edit_model/{call_id}"]
    with pytest.raises(ValueError, match="hash mismatch"):
        read_source(external.revised(sha256="0" * 64), tmp_path)


def test_action_artifact_revision_is_not_its_journal_commit(tmp_path, monkeypatch):
    payload = json.dumps(
        {
            "attempts": [
                {
                    "call_id": "call:" + "1" * 64,
                    "commit_id": "2" * 40,
                    "parent_ids": [],
                    "record": {
                        "seq": 1,
                        "ts": "2026-10-17T00:00:00Z",
                        "messages": [],
                        "trace_ids": [],
                        "attempt": {
                            "action": "edit_question",
                            "request": {
                                "action": "edit_question",
                                "input": {"question": {"text": "Case", "outcome": "construct:y"}},
                            },
                            "outcome": {
                                "status": "applied",
                                "result": None,
                                "effects": {
                                    "produced": [
                                        {
                                            "artifact_id": "question",
                                            "revision": "3" * 40,
                                            "derived_from": {},
                                            "produced_by": "edit_question",
                                            "created_at": "2026-10-17T00:00:00Z",
                                            "source": {"kind": "result", "result": "4" * 40},
                                        }
                                    ],
                                    "retracted": [],
                                    "reports": {},
                                },
                            },
                        },
                    },
                }
            ],
            "dependencies": [],
            "running": [],
        }
    ).encode()
    monkeypatch.setattr(
        "grassmoth_server.public.client.urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )
    client = ActionClient("http://example.invalid", "BENCH", tmp_path)
    assert client.artifact_revision("call:" + "1" * 64, "question") == "3" * 40


def test_t512_mask_reproduces_committed_data_without_changing_support(tmp_path):
    root = Path(__file__).parents[2] / "evaluation/cases/assets/t512"
    mask = ObservationMask.model_validate_json((root / "mask.json").read_text())
    masked = tmp_path / "observations.csv"
    mask_csv(root / "generated.csv", masked, mask)
    assert masked.read_bytes() == (root / "observations.csv").read_bytes()
    with (root / "generated.csv").open() as stream:
        original = list(csv.DictReader(stream))
    with masked.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 512
    assert sum(bool(row["accumulator"]) for row in rows) == 17
    assert sum(bool(row["accumulator_origin"]) for row in rows) == 1
    assert sum(bool(row["driver_exact"]) for row in rows) == 1
    for before, after in zip(original, rows, strict=True):
        for column, value in before.items():
            if column == "timestamp" or column.endswith(("_start", "_end")) or after[column]:
                assert after[column] == value


@pytest.mark.parametrize("variant", ["precise", "noisy"])
def test_dense_case_pins_sources_and_declares_only_sparse_value_masking(variant, tmp_path):
    from evaluation.declarations import CaseRecipe, RecordedData

    root = Path(__file__).parents[2] / "evaluation/cases"
    recipe = CaseRecipe.model_validate_json((root / f"dense_{variant}.json").read_text())
    assert isinstance(recipe.data, RecordedData)
    for source in (recipe.fit_model, *recipe.data.files, *recipe.data.provenance_files):
        read_source(source, root)
    assets = root / "assets/dense"
    mask = ObservationMask.model_validate_json((assets / "mask.json").read_text())
    output = tmp_path / "masked.csv"
    mask_csv(assets / variant / "generated.csv", output, mask)
    assert output.read_bytes() == (assets / variant / "observations.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(output.read_text())))
    assert len(rows) == 129
    assert sum(bool(row["dense"]) for row in rows) == 129
    assert sum(bool(row["sparse"]) for row in rows) == 17
    generation = json.loads((assets / variant / "generation.json").read_text())
    assert generation["simulate"]["input"]["num_replicates"] == 1
    assert generation["simulate"]["input"]["seed"] == 8029
