"""Negotiated JSON preserves complete action results and the viewer's binary contract."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth_server.actions.contracts.polling import ActionResult, StudyError
from grassmoth_server.api.app import app
from grassmoth_server.study import calls
from grassmoth_server.study.persistence.codec import json_result, pack_payload, pack_result
from scripts.codegen.export_api import export_openapi
from scripts.codegen.export_story_scenarios import examples

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("accept", "media_type"),
    [
        (None, "application/msgpack"),
        ("*/*", "application/msgpack"),
        ("application/msgpack, application/json", "application/msgpack"),
        ("application/json;q=0.5, application/msgpack;q=0.5", "application/msgpack"),
        ("application/json;q=0.2, application/msgpack;q=0.8", "application/msgpack"),
        ("application/json;q=0, application/msgpack;q=0", "application/msgpack"),
        ("text/html", "application/msgpack"),
        ("application/json;q=invalid", "application/msgpack"),
        ("application/json", "application/json"),
        ("application/json;q=0.8, application/msgpack;q=0.2", "application/json"),
        ("application/json, */*;q=0.5", "application/json"),
        ("application/json;q=0.5, */*", "application/msgpack"),
        ("application/*;q=0.8, application/msgpack;q=0.2", "application/json"),
        ("Application/JSON;Q=1.000, application/msgpack;q=0.9", "application/json"),
    ],
)
def test_post_and_poll_negotiate_all_actions_and_statuses(monkeypatch, accept, media_type):
    client = TestClient(app)
    client.headers.clear()
    headers = {} if accept is None else {"Accept": accept}
    for example in examples():
        payload = pack_result(example.poll)

        async def result(*_args, payload=payload):
            return ActionResult(payload=payload)

        monkeypatch.setattr(calls, "submit_action", result)
        monkeypatch.setattr(calls, "poll_action", result)
        path = f"/api/studies/SYNTHETIC/{example.request.action}"
        request_body = example.request.model_dump(mode="json")
        if example.request.action == "prepare_data":
            request_body["input"]["source"] = "input"
        responses = (
            client.post(path, json=request_body, headers=headers),
            client.get(f"{path}/{example.poll.call_id}", headers=headers),
        )
        for response in responses:
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == media_type
            assert "Accept" in response.headers["vary"].split(", ")
            assert response.content == (
                json_result(payload) if media_type == "application/json" else payload
            )


def test_json_expands_shared_buffers_and_preserves_shapes_scalars_and_nulls():
    values = {
        "float": np.array([[1.5, np.nan], [np.inf, -np.inf]], dtype=np.float32),
        "bool": np.array([[True, False]], dtype=bool),
        "integer": np.array([-(2**63), 2**63 - 1], dtype=np.int64),
        "unsigned": np.array([2**64 - 1], dtype=np.uint64),
        "scalar": np.array(7),
        "empty": np.empty((2, 0, 3)),
    }
    arrays = {name: NumericalArray.from_numpy(value).model_dump() for name, value in values.items()}
    payload = pack_payload({"body": arrays, "repeated": arrays, "messages": []})
    assert payload.count(arrays["float"]["npy"]) == 1
    rendered = json_result(payload)
    assert b"NaN" not in rendered
    assert b"Infinity" not in rendered
    result = json.loads(rendered)
    assert (
        result["body"]
        == result["repeated"]
        == {
            "float": [[1.5, None], [None, None]],
            "bool": [[True, False]],
            "integer": [-(2**63), 2**63 - 1],
            "unsigned": [2**64 - 1],
            "scalar": 7,
            "empty": [[], []],
        }
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ],
)
def test_json_round_trips_all_owned_dtypes_and_scalar_arrays(dtype: str):
    selected = np.dtype(dtype)
    if selected.kind == "b":
        values = np.array([False, True], dtype=selected)
    elif selected.kind in "iu":
        limits = np.iinfo(dtype)
        values = np.array([limits.min, 1, limits.max], dtype=selected)
    else:
        limits = np.finfo(dtype)
        values = np.array(
            [
                -limits.max,
                1.3508236,
                -0.0,
                limits.smallest_subnormal,
                limits.max,
                np.nan,
                np.inf,
                -np.inf,
            ],
            dtype=selected,
        )
    for array in (values.reshape(1, -1), values[1].reshape(())):
        owner = NumericalArray.from_numpy(array)
        assert not owner.values.flags.writeable
        result = json.loads(json_result(pack_payload({"array": owner.model_dump()})))["array"]
        restored = np.asarray(result, dtype=object)
        assert restored.shape == array.shape
        finite = np.isfinite(array)
        assert all(value is None for value in restored[~finite].flat)
        np.testing.assert_array_equal(
            np.asarray(restored[finite].tolist(), dtype=selected), array[finite]
        )
        if selected.kind == "f":
            np.testing.assert_array_equal(
                np.signbit(np.asarray(restored[finite].tolist(), dtype=selected)),
                np.signbit(array[finite]),
            )


@pytest.mark.parametrize("openapi", [app.openapi, export_openapi], ids=["live", "exported"])
def test_json_openapi_validates_owned_examples_and_has_no_binary_arrays(openapi):
    schema = openapi()
    for path, operations in schema["paths"].items():
        for operation in operations.values():
            content = operation["responses"]["200"]["content"]
            if "application/msgpack" not in content:
                continue
            response_schema = content["application/json"]["schema"]
            validator = Draft202012Validator(
                {**response_schema, "components": schema["components"]}
            )
            for example in examples():
                if path.endswith(example.request.action) or path.endswith("{call_id}"):
                    validator.validate(json.loads(json_result(pack_result(example.poll))))
            pending = [response_schema]
            visited = set()
            while pending:
                node = pending.pop()
                if isinstance(node, dict):
                    assert "npy" not in node.get("properties", {})
                    reference = node.get("$ref")
                    if reference is not None and reference not in visited:
                        visited.add(reference)
                        pending.append(
                            schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]
                        )
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)


def test_http_errors_stay_json(monkeypatch):
    async def unknown(*_args):
        return StudyError(code="unknown_call", detail="No saved call")

    monkeypatch.setattr(calls, "poll_action", unknown)
    response = TestClient(app).get(
        f"/api/studies/MISSING/fit/call:{'0' * 64}", headers={"Accept": "application/msgpack"}
    )
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "No saved call"}
