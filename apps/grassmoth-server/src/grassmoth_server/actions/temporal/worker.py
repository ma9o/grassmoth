"""Study worker entrypoint.

Run with::

    uv run python -m grassmoth_server.actions.temporal.worker
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from grassmoth_server.actions.temporal.client import (
    MODEL_CHECKS_TASK_QUEUE,
    OPENROUTER_TASK_QUEUE,
    STUDY_TASK_QUEUE,
    connect_client,
)
from grassmoth_server.actions.temporal.extraction.activities import OpenRouterActivities
from grassmoth_server.actions.temporal.extraction.workflow import (
    ExtractionChunkWorkflow,
    MeasurementsWorkflow,
)
from grassmoth_server.actions.temporal.llm.workflow import LLMSubroutineWorkflow
from grassmoth_server.actions.temporal.study.activities import (
    ALL_ACTIVITIES,
    evaluate_data_checks_activity,
    evaluate_model_checks_activity,
)
from grassmoth_server.actions.temporal.study.workflow import StudyWorkflow

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from temporalio.client import Client

logger = logging.getLogger(__name__)


def study_workflow_runner() -> SandboxedWorkflowRunner:
    """Sandbox runner with the library and service packages passed through.

    The process configures JAX before starting workers; re-importing jaxlib
    inside the sandbox aborts the process. Passing both
    packages through is safe here because workflow determinism is carried
    by construction (the workflow only calls the pure machine functions).
    Typed model requests run deterministic NetworkX graph validation during
    payload decoding; its import-time backend configuration stays outside replay.
    Native prior constructors also lazily import JAX/NumPyro while decoding;
    encoding their array dtypes lazily imports NumPy. These numerical modules
    must retain the worker's initialized instances in both directions.
    """
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "grassmoth",
            "grassmoth_server",
            "pydantic",
            "networkx",
            "jax",
            "jaxlib",
            "numpyro",
            "numpy",
            "sympy",
        )
    )


def build_worker(client: Client, task_queue: str = STUDY_TASK_QUEUE) -> Worker:
    """Build the study worker with scientific workflows and their orchestration activities."""
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[
            StudyWorkflow,
            MeasurementsWorkflow,
            ExtractionChunkWorkflow,
            LLMSubroutineWorkflow,
        ],
        activities=ALL_ACTIVITIES,
        workflow_runner=study_workflow_runner(),
    )


def build_openrouter_worker(
    client: Client,
    openrouter_client: AsyncOpenAI,
    task_queue: str = OPENROUTER_TASK_QUEUE,
) -> Worker:
    """Build the provider worker with the configured request-per-minute queue limit."""
    from grassmoth_server.config import get_config

    max_rpm = get_config().extraction_workers.max_rpm
    return Worker(
        client,
        task_queue=task_queue,
        activities=[OpenRouterActivities(openrouter_client).call_openrouter_activity],
        max_task_queue_activities_per_second=(max_rpm / 60) if max_rpm else None,
    )


def build_model_checks_worker(
    client: Client,
    task_queue: str = MODEL_CHECKS_TASK_QUEUE,
) -> Worker:
    """Serialize automatic exact check batches independently of extraction workers."""
    return Worker(
        client,
        task_queue=task_queue,
        activities=[evaluate_model_checks_activity, evaluate_data_checks_activity],
        max_concurrent_activities=1,
    )


async def run_worker() -> None:
    """Run study, provider, and model-check workers until shutdown or worker failure."""
    from grassmoth_server.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    from grassmoth_server.config import configure_jax_persistent_cache
    from grassmoth_server.llm.openrouter import create_openrouter_client

    configure_jax_persistent_cache()
    client = await connect_client()
    async with create_openrouter_client() as openrouter_client:
        study_worker = build_worker(client)
        openrouter_worker = build_openrouter_worker(client, openrouter_client)
        model_checks_worker = build_model_checks_worker(client)
        logger.info("Study worker started on task queue %s", STUDY_TASK_QUEUE)
        logger.info("OpenRouter worker started on task queue %s", OPENROUTER_TASK_QUEUE)
        logger.info(
            "Model checks worker started on task queue %s",
            MODEL_CHECKS_TASK_QUEUE,
        )
        async with asyncio.TaskGroup() as tasks:
            for worker in (study_worker, openrouter_worker, model_checks_worker):
                tasks.create_task(worker.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # The SDK reports update decoding/validation rejections only at DEBUG.
    logging.getLogger("temporalio.worker._workflow_instance").setLevel(logging.DEBUG)
    asyncio.run(run_worker())
