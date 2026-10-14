"""Temporal activities for measurements extraction.

The workflow layer owns the durable control flow. These activities own all I/O:
artifact reads/writes, transient run files, OpenRouter calls, validation, and
live progress events.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from temporalio import activity

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.actions.prepare_data.context import extraction_context
from nof1_causal_lab.actions.prepare_data.worker_output import EXTRACTION_ROW_SCHEMA, ExtractionRow
from nof1_causal_lab.actions.temporal.errors import as_non_retryable_application_error
from nof1_causal_lab.actions.temporal.extraction.messages import (
    CompletedExtractionChunk,
    ExtractionChunkFinalizeInput,
    MeasurementChunkContext,
    MeasurementChunkRef,
    MeasurementsFile,
    MeasurementsFinalizeInput,
    MeasurementsPlan,
    MeasurementsWorkflowInput,
    ProgressEventInput,
)
from nof1_causal_lab.actions.temporal.extraction.settings import first_config_value
from nof1_causal_lab.actions.temporal.llm.messages import (
    OpenRouterCallInput,
    OpenRouterCallResult,
    ToolCallSummary,
)
from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.artifacts.observations.preparation import (
    DataPreparationResult,
    PreparedDataMetadata,
)
from nof1_causal_lab.artifacts.observations.records import ObservationRecord
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.llm.specs import EmbeddedLLMSpec
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _run_root(workspace_id: str, run_id: str) -> str:
    return storage.join(data_module.scratch_run_dir(workspace_id, run_id), "extraction")


def _write_json(path: str, value: object) -> None:
    storage.write_text(path, json.dumps(value))


def _read_json(path: str) -> JsonObject:
    return storage.read_json(path)


@activity.defn
async def emit_progress_event_activity(activity_input: ProgressEventInput) -> None:
    """Persist a workflow progress event in the designated workspace."""
    from nof1_causal_lab.actions.progress import emit_event

    emit_event(activity_input.workspace_id, activity_input.event)


@activity.defn
async def plan_measurements_activity(activity_input: MeasurementsWorkflowInput) -> MeasurementsPlan:
    """Compute deterministic measurements and plan cached or LLM-backed extraction chunks."""
    import polars as pl

    from nof1_causal_lab.actions.prepare_data.planning import prepare_semantic_chunks
    from nof1_causal_lab.actions.prepare_data.worker_output import WorkerOutput
    from nof1_causal_lab.config import get_config
    from nof1_causal_lab.observations.aggregations import compute_indicators

    store = ArtifactStore(activity_input.workspace_id)
    store.read_meta("raw_data", activity_input.raw_data_revision)
    pins: dict[ArtifactId, GitOid] = {
        "raw_data": activity_input.raw_data_revision,
        "model": activity_input.preparation.dynamical_model_spec_ref,
    }
    run_id = f"seq-{activity_input.seq:06d}"
    root = _run_root(activity_input.workspace_id, run_id)

    raw_table = store.read_parquet_table(
        "raw_data",
        pins["raw_data"],
        parquet_filename("raw_data", "raw"),
    )
    raw_df = pl.DataFrame(raw_table)
    from nof1_causal_lab.actions.prepare_data.resolve import resolve_preparation

    resolved = resolve_preparation(store, activity_input.preparation)
    question = resolved.definition.context
    measurement_structure = extraction_context(resolved)

    config = get_config()
    extraction_workers = config.extraction_workers
    time_col = "timestamp"

    computed_dicts: list[ExtractionRow] = []
    if any(ind.extraction.kind == "computed" for ind in measurement_structure.indicators):
        computed_df = compute_indicators(
            raw_df,
            measurement_structure,
            time_col,
        )
        computed_dicts = TypeAdapter(list[ExtractionRow]).validate_python(computed_df.to_dicts())

    extraction_llm = extraction_workers.llm
    embedded_defaults = config.llm.embedded
    llm = EmbeddedLLMSpec(
        harness="none",
        model=extraction_llm.model,
        max_tokens=first_config_value(extraction_llm.max_tokens, embedded_defaults.max_tokens),
        timeout=extraction_workers.worker_timeout,
        reasoning_effort=first_config_value(
            extraction_llm.reasoning_effort,
            embedded_defaults.reasoning_effort,
        ),
    )

    chunks: list[MeasurementChunkRef] = []
    empty_output = WorkerOutput()
    if any(ind.extraction.kind == "semantic" for ind in measurement_structure.indicators):
        chunk_texts, chunk_window_starts, chunk_contexts, empty_output = prepare_semantic_chunks(
            raw_df=raw_df,
            measurement_structure=measurement_structure,
            time_col=time_col,
            max_events_per_window=extraction_workers.max_events_per_window,
        )
        for worker_id, (chunk_text, window_starts, chunk_context) in enumerate(
            zip(chunk_texts, chunk_window_starts, chunk_contexts, strict=True)
        ):
            spec_ref = storage.join(root, "chunks", f"worker-{worker_id:06d}.json")
            _write_json(
                spec_ref,
                {
                    "worker_id": worker_id,
                    "question": question,
                    "window_text": chunk_text,
                    "window_starts": window_starts,
                    "measurement_structure": chunk_context.model_dump(mode="json"),
                },
            )
            from nof1_causal_lab.actions.temporal.extraction.cache import preparation_cache_path
            from nof1_causal_lab.storage.cache import read

            chunk_spec = dict(_read_json(spec_ref))
            cache_ref = preparation_cache_path(
                spec_ref,
                llm,
                extraction_workers.max_tool_turns,
                {"window_starts": window_starts},
            )
            chunk_spec["cache_ref"] = cache_ref
            _write_json(spec_ref, chunk_spec)
            cached_result_ref = None
            cached = read(cache_ref)
            if cached is not None:
                cached_result_ref = storage.join(root, "reused", f"worker-{worker_id:06d}.json")
                storage.write_text(cached_result_ref, cached.decode())
            chunks.append(
                MeasurementChunkRef(
                    worker_id=worker_id,
                    n_windows=len(window_starts),
                    spec_ref=spec_ref,
                    cached_result_ref=cached_result_ref,
                )
            )

    plan_ref = storage.join(root, "plan.json")
    _write_json(
        plan_ref,
        {
            "workspace_id": activity_input.workspace_id,
            "run_id": run_id,
            "pins": pins,
            "question": question,
            "measurement_structure": measurement_structure.model_dump(mode="json"),
            "preparation": resolved.model_dump(mode="json"),
            "computed_dicts": computed_dicts,
            "empty_output": empty_output.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        },
    )

    return MeasurementsPlan(
        workspace_id=activity_input.workspace_id,
        run_id=run_id,
        plan_ref=plan_ref,
        pins=pins,
        chunks=chunks,
        max_concurrent_workers=extraction_workers.max_concurrent_workers,
        max_rpm=extraction_workers.max_rpm,
        max_tool_turns=extraction_workers.max_tool_turns,
        llm=llm,
    )


class OpenRouterActivities:
    """Bind OpenRouter activities to their worker-owned transport client."""

    def __init__(self, client: AsyncOpenAI) -> None:
        """Bind the provider client shared by this worker's model-call activities."""
        self._client = client

    @activity.defn
    async def call_openrouter_activity(
        self, activity_input: OpenRouterCallInput
    ) -> OpenRouterCallResult:
        """Persist one provider response, reusing the recorded result on activity retries."""
        if storage.exists(activity_input.call_ref):
            return OpenRouterCallResult.model_validate(
                _read_json(activity_input.call_ref)["result"]
            )

        from nof1_causal_lab.llm.openrouter import GenerateConfig, Tool, call_model

        async def _unused_tool(**kwargs: str) -> str:
            del kwargs
            return ""

        conversation = _read_json(activity_input.conversation_ref)
        messages = list(TypeAdapter(list[JsonObject]).validate_python(conversation["messages"]))
        tools = [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=dict(tool.parameters),
                execute=_unused_tool,
                stop_on_success=tool.kind == "terminal",
                success_output="VALID" if tool.kind == "terminal" else None,
            )
            for tool in activity_input.tools
        ] or None
        output = await call_model(
            activity_input.llm.model,
            messages,
            client=self._client,
            tools=tools,
            config=GenerateConfig(
                max_tokens=activity_input.llm.max_tokens,
                timeout=activity_input.llm.timeout,
                reasoning_effort=activity_input.llm.reasoning_effort,
            ),
            log_label=activity_input.log_label,
        )

        _write_json(activity_input.assistant_ref, output)

        next_messages = [*messages, output["message"]]
        _write_json(activity_input.next_conversation_ref, {"messages": next_messages})

        tool_calls = [
            ToolCallSummary(
                index=index,
                id=str(tool_call.get("id", "")),
                name=str(
                    (tool_call.get("function") or {}).get("name") or tool_call.get("name", "")
                ),
            )
            for index, tool_call in enumerate(output["message"].get("tool_calls") or [])
        ]
        result = OpenRouterCallResult(
            conversation_ref=activity_input.next_conversation_ref,
            assistant_ref=activity_input.assistant_ref,
            model=output["model"],
            stop_reason=output.get("stop_reason"),
            time=float(output.get("time") or 0.0),
            usage=output.get("usage"),
            tool_calls=tool_calls,
        )
        _write_json(activity_input.call_ref, {"result": result.model_dump(mode="json")})
        return result


@activity.defn
async def finalize_extraction_chunk_activity(
    activity_input: ExtractionChunkFinalizeInput,
) -> CompletedExtractionChunk:
    """Validate a chunk's worker output, publish its cache entry, and retain observation rows."""
    from nof1_causal_lab.actions.prepare_data.worker_output import (
        WorkerOutput,
        validate_worker_output,
    )
    from nof1_causal_lab.storage.cache import publish

    data = _read_json(activity_input.result_ref)
    spec = MeasurementChunkContext.model_validate(_read_json(activity_input.spec_ref))
    output, errors = validate_worker_output(
        data,
        spec.measurement_structure,
        spec.window_starts,
    )
    if output is None:
        raise ValueError("; ".join(errors))
    assert spec.cache_ref is not None
    output = WorkerOutput.model_validate_json(
        publish(spec.cache_ref, output.model_dump_json().encode())
    )
    dataframe = output.to_dataframe()

    result_ref = storage.join(
        _run_root(activity_input.workspace_id, activity_input.run_id),
        "results",
        f"worker-{activity_input.worker_id:06d}.json",
    )
    _write_json(
        result_ref,
        {
            "dataframe": dataframe.to_dicts(),
            "n_extractions": len(output.extractions),
            "status": "completed",
        },
    )
    return CompletedExtractionChunk(
        worker_id=activity_input.worker_id,
        status="completed",
        n_extractions=len(output.extractions),
        n_windows=activity_input.n_windows,
        n_llm_calls=activity_input.n_llm_calls,
        result_ref=result_ref,
        reused=activity_input.reused,
    )


@activity.defn
@execution_failure_handler
async def finalize_measurements_activity(
    activity_input: MeasurementsFinalizeInput,
) -> Applied[DataPreparationResult]:
    """Assemble computed and extracted rows into a validated panel with preparation metadata."""
    import polars as pl

    from nof1_causal_lab.actions.prepare_data.materialization import materialize_panel
    from nof1_causal_lab.artifacts.observations.rows import validate_observation_rows
    from nof1_causal_lab.observations.rows import annotate_observation_rows, prepared_time_origin

    try:
        plan = MeasurementsFile.model_validate(_read_json(activity_input.plan_ref))
        measurement_structure = plan.measurement_structure
        computed_dicts = plan.computed_dicts
        chunk_specs = plan.chunks
        results_by_worker = {result.worker_id: result for result in activity_input.chunk_results}

        semantic_dicts: list[ExtractionRow] = TypeAdapter(list[ExtractionRow]).validate_python(
            plan.empty_output.to_dataframe().to_dicts()
        )

        for chunk_spec in chunk_specs:
            worker_id = chunk_spec.worker_id
            result = results_by_worker[worker_id]
            if result.status == "completed":
                chunk_payload = _read_json(result.result_ref)
                semantic_dicts.extend(
                    TypeAdapter(list[ExtractionRow]).validate_python(chunk_payload["dataframe"])
                )

        preparation = plan.preparation
        variables = preparation.definition.observation_schema()
        all_dicts = computed_dicts + semantic_dicts
        observation_rows = TypeAdapter(list[ObservationRecord]).validate_python(
            annotate_observation_rows(
                pl.DataFrame(all_dicts, schema=EXTRACTION_ROW_SCHEMA), variables
            ).to_dicts()
            if all_dicts
            else [],
        )
        panel = materialize_panel(observation_rows, measurement_structure)
        if len(panel) == 0:
            raise ValueError("Extraction produced no observations")
        panel = validate_observation_rows(panel, variables)
        metadata = PreparedDataMetadata(
            preparation=preparation.definition,
            time_origin=prepared_time_origin(panel, preparation.source.start),
        )
        store = ArtifactStore(activity_input.workspace_id)
        return Applied(
            result=DataPreparationResult(
                workers=tuple(results_by_worker[spec.worker_id] for spec in chunk_specs),
                extraction_reused=sum(
                    result.reused is True for result in activity_input.chunk_results
                ),
            ),
            effects=ActionEffects(
                produced=(
                    store.write_artifact(
                        "panel",
                        derived_from=activity_input.pins,
                        produced_by="prepare_data",
                        parquet_files={parquet_filename("panel", "panel"): panel},
                        json_files={
                            json_filename("panel", "metadata"): metadata.model_dump(mode="json")
                        },
                    ),
                )
            ),
        )
    except Exception as exc:
        raise as_non_retryable_application_error(exc) from exc


MEASUREMENT_ACTIVITIES = [
    emit_progress_event_activity,
    plan_measurements_activity,
    finalize_extraction_chunk_activity,
    finalize_measurements_activity,
]
