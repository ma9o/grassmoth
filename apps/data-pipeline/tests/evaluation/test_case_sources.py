"""Pinned external inputs, action revisions and reproducible observation masking."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pygit2
import pytest
from evaluation.client import ActionClient
from evaluation.declarations import CaseFile, StudyModel
from evaluation.fixtures.t512 import ObservationMask, mask_csv
from evaluation.sources import model_document, read_source

from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.study.persistence.codec import pack_payload
from nof1_causal_lab.study.persistence.git_objects import write_tree

pytestmark = pytest.mark.contract


def test_case_sources_pin_file_bytes_and_read_only_study_objects(tmp_path):
    payload = b'{"measurement_clock": "3h"}'
    (tmp_path / "model.json").write_bytes(payload)
    source = CaseFile(path="model.json", sha256=hashlib.sha256(payload).hexdigest())
    document = model_document(source, read_source(source, tmp_path))
    assert str(document.measurement_clock) == "3h"
    (tmp_path / "model.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_source(source, tmp_path)

    repository = pygit2.init_repository(str(tmp_path / "history.git"), bare=True)
    stored = pack_payload({"dynamical_model_spec": {"measurement_clock": "3h"}})
    revision = write_tree(repository, {"result.msgpack": stored})
    external = StudyModel(
        repository="history.git",
        revision=GitOid(str(revision)),
        sha256=hashlib.sha256(stored).hexdigest(),
    )
    assert model_document(external, read_source(external, tmp_path)) == document
    assert not tuple(repository.references)
    assert "nof1.format" not in repository.config
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
            "running": None,
        }
    ).encode()
    monkeypatch.setattr("evaluation.client.urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
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
