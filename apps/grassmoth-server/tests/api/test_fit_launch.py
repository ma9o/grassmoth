"""Public fit requests reject kernels outside the production launch contract."""

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from grassmoth_server.api.app import app
from grassmoth_server.study import calls

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "algorithm",
    [
        {"method": "nuts_da"},
        {"method": "mpgibbs", "path_update": {"method": {"kind": "twisted_particle_agrad"}}},
        {"method": "mpgibbs", "path_update": {"method": {"kind": "auxiliary_kalman"}}},
        {"method": "mpgibbs", "path_update": {"update": {"kind": "blocked"}}},
        {"method": "mpgibbs", "path_update": {"proposal": {"kind": "parameter_mean"}}},
        {"method": "mpgibbs", "path_update": {"proposal": {"kind": "conditional_parameter"}}},
        {
            "method": "mpgibbs",
            "path_update": {"proposal": {"kind": "parameter_mixture", "weights": "prior"}},
        },
    ],
    ids=["nuts", "twisted-agrad", "kalman", "blocked", "mean", "conditional", "prior-weights"],
)
def test_http_and_openapi_reject_unsupported_fit_launches(monkeypatch, algorithm):
    async def unexpected_submit(*_args):
        pytest.fail("An invalid launch must be rejected before action submission")

    monkeypatch.setattr(calls, "submit_action", unexpected_submit)
    body = {
        "action": "fit",
        "input": {
            "target": {
                "kind": "new",
                "dynamical_model_spec_ref": "1" * 40,
                "data_ref": {"revision": "2" * 40, "replicate_index": 0},
            },
            "settings": {"algorithm": algorithm},
        },
    }
    schema = app.openapi()
    request_schema = schema["paths"]["/api/studies/{workspace_id}/fit"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]
    validator = Draft202012Validator({**request_schema, "components": schema["components"]})
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(body)
    response = TestClient(app).post("/api/studies/SYNTHETIC/fit", json=body)
    assert response.status_code == 422
    assert any("algorithm" in issue["loc"] for issue in response.json()["detail"])
