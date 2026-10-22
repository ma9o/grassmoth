"""Server runtime policy and isolated hardware routing for server tests."""

from dataclasses import replace
from typing import NoReturn

import modal
import pytest

from grassmoth.artifacts.sampling import InferenceHardware
from grassmoth_server import config
from grassmoth_server.bootstrap import initialize_numerical_runtime


def pytest_configure() -> None:  # noqa: V103 - pytest calls this configuration hook before collection.
    """Initialize the server numerical policy before collecting its tests."""
    initialize_numerical_runtime()


@pytest.fixture(autouse=True)
def _fit_on_local_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests fit locally; hardware routing contracts must mock the Modal boundary."""
    loaded = config.load_config()
    local = replace(
        loaded,
        inference=replace(
            loaded.inference,
            sampler=loaded.inference.sampler.revised(hardware=InferenceHardware.LOCAL),
        ),
    )
    monkeypatch.setattr(config, "get_config", lambda: local)

    def reject_modal_run(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail(
            "Live Modal execution is disabled in pytest; mock the Modal boundary.", pytrace=False
        )

    monkeypatch.setattr(modal.App, "run", reject_modal_run)
