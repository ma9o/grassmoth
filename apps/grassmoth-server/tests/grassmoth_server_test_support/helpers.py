"""Async execution and study setup for server tests."""

import asyncio

from grassmoth_test_support.helpers import fixture_entity_id


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def write_question(store, question=None):
    """Write the study question that roots a test lineage and return its artifact record."""
    from grassmoth.artifacts.model.question import QuestionSpec

    value = (
        question
        if question is not None
        else QuestionSpec(
            text="What does the model answer?",
            outcome=fixture_entity_id("construct", "unmeasured_outcome"),
        )
    )
    return store.write_artifact(
        "question",
        derived_from={},
        produced_by="edit_question",
        json_files={"question.json": value.model_dump(mode="json")},
    )
