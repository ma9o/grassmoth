"""Temporal client wiring shared by the facade and the worker.

Workflow metadata uses memos so local and CI namespaces need no preregistered
custom Search Attribute schema.
"""

from __future__ import annotations

import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

STUDY_TASK_QUEUE = os.environ.get("TEMPORAL_TASK_QUEUE", "nof1-studies")
MODEL_CHECKS_TASK_QUEUE = os.environ.get(
    "TEMPORAL_MODEL_CHECKS_TASK_QUEUE",
    "nof1-model-checks",
)
OPENROUTER_TASK_QUEUE = os.environ.get("TEMPORAL_OPENROUTER_TASK_QUEUE", "nof1-openrouter")


def study_workflow_id(workspace_id: str) -> str:
    """One entity workflow per study."""
    return f"study-{workspace_id}"


# The study workflow keeps its executing attempt in this memo key. The server returns
# memos with the workflow's description, so status reads need no worker.
RUNNING_ACTION_MEMO = "running_action"


async def connect_client() -> Client:
    """Connect to the configured Temporal address and namespace using Pydantic payloads."""
    return await Client.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
        namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
        data_converter=pydantic_data_converter,
    )
