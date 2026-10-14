"""Require explicit concern owners before pytest applies marker selection."""

import pytest

from nof1_causal_lab.bootstrap import initialize_numerical_runtime


def pytest_configure() -> None:
    """Initialize numerical policy before collecting scientific fixtures."""
    initialize_numerical_runtime()


pytest_plugins = ["pytester"]

INFERENCE_CONCERNS = ("sampling", "warmup", "simulation", "predictive", "recovery")
TEST_CONCERNS = frozenset({"contract", "inference", "workflow"})


@pytest.fixture(autouse=True)
def _fit_on_local_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests fit locally whatever config.yaml selects; a test may opt into Modal itself."""
    from dataclasses import replace

    from nof1_causal_lab import config

    loaded = config.load_config()
    local = replace(loaded, inference=replace(loaded.inference, compute_backend="local"))
    monkeypatch.setattr(config, "get_config", lambda: local)


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
