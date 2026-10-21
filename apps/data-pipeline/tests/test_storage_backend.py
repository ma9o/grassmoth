"""Deployment storage roots remain independent of scientific requests."""

import pytest

from nof1_causal_service.storage.backend import get_base_uri

pytestmark = pytest.mark.contract


def test_local_data_root_is_explicit_and_does_not_override_remote_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setenv("NOF1_DATA_ROOT", str(tmp_path))
    assert get_base_uri() == str(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.setenv("R2_PREFIX", "test-prefix")
    assert get_base_uri() == "s3://test-bucket/test-prefix"
