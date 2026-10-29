"""Require explicit concern owners before pytest applies marker selection."""

from pathlib import Path

import jax
import pytest

INFERENCE_CONCERNS = ("sampling", "warmup", "simulation", "predictive", "recovery")
TEST_CONCERNS = frozenset({"contract", "inference", "workflow"})


def pytest_configure() -> None:  # noqa: V103 - pytest calls this configuration hook before collection.
    """Share compiled programs across workers and runs; compilation dominates numerical tests.

    JAX keys each entry by the lowered program, compile options and JAX version, so a
    code change misses instead of reusing a stale executable. Delete the directory to
    time a cold run or reclaim disk.
    """
    if not jax.config.jax_compilation_cache_dir:
        cache = Path.home() / ".cache" / "grassmoth" / "jax-tests"
        jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


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
