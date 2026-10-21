"""Require explicit concern owners before pytest applies marker selection."""

from typing import NoReturn

import pytest

from nof1_causal_service.bootstrap import initialize_numerical_runtime


def pytest_configure() -> None:
    """Initialize numerical policy before collecting scientific fixtures."""
    initialize_numerical_runtime()


pytest_plugins = ["pytester"]

INFERENCE_CONCERNS = ("sampling", "warmup", "simulation", "predictive", "recovery")
TEST_CONCERNS = frozenset({"contract", "inference", "workflow"})


@pytest.fixture(autouse=True)
def _fit_on_local_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests fit locally; hardware routing contracts must mock the Modal boundary."""
    from dataclasses import replace

    import modal

    from nof1_causal_service import config
    from nof1_causal_lab.artifacts.sampling import InferenceHardware

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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate ownership and inference subcategories before pytest deselects tests."""
    unowned = [
        item.nodeid
        for item in items
        if not TEST_CONCERNS.intersection(mark.name for mark in item.iter_markers())
    ]
    invalid_inference = [
        item.nodeid
        for item in items
        if any(
            mark.args
            or set(mark.kwargs) != {"concern"}
            or mark.kwargs["concern"] not in INFERENCE_CONCERNS
            for mark in item.iter_markers("inference")
        )
    ]
    if unowned or invalid_inference:
        details = []
        if unowned:
            details.append("Unowned tests:\n" + "\n".join(f"  {nodeid}" for nodeid in unowned))
        if invalid_inference:
            details.append(
                "Invalid inference concerns:\n"
                + "\n".join(f"  {nodeid}" for nodeid in invalid_inference)
            )
        raise pytest.UsageError(
            "Every test must declare at least one concern owner via pytest.mark: "
            + ", ".join(sorted(TEST_CONCERNS))
            + ". Use inference(concern=...) with one of: "
            + ", ".join(INFERENCE_CONCERNS)
            + ".\n"
            + "\n".join(details)
        )
