"""Shared fixtures for pipeline integration tests."""

from __future__ import annotations

import pytest

from nof1_causal_service.study.persistence.store import ArtifactStore


@pytest.fixture
def integration_workspace(monkeypatch, tmp_path) -> str:
    from nof1_causal_service.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "fixture_workspace"


@pytest.fixture
def artifact_store(integration_workspace: str) -> ArtifactStore:
    return ArtifactStore(integration_workspace)
