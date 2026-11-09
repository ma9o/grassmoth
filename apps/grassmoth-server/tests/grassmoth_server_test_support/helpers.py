"""Async execution and study setup for server tests."""

import asyncio

from grassmoth_server_test_support.storage import seed_record
from grassmoth_test_support.helpers import fixture_entity_id


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def write_initial_model(store, model=None):
    """Save a complete initial model for a test lineage."""
    from grassmoth_test_support.purpose import initial_model

    value = (
        model
        if model is not None
        else initial_model(outcome=fixture_entity_id("construct", "unmeasured_outcome"))
    )
    return seed_record(
        store,
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": value.model_dump(mode="json")},
    )
