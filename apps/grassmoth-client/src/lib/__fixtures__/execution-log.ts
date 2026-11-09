import type { ExecutionMessage } from "@grassmoth/api-types";

const timestamp = "2026-10-05T12:00:00Z";
const attempt_id = "00000000-0000-0000-0000-000000000001";

export const executionMessages: readonly ExecutionMessage[] = [
  {
    kind: "log",
    timestamp,
    severity: "info",
    code: "FIT_CAPACITY_ESTIMATE",
    subject: "fit",
    detail: "Estimated sampling memory: 128 MiB.",
  },
  {
    kind: "progress",
    timestamp,
    progress: {
      event: "grassmoth.step",
      attempt_id,
      cursor: "1-0",
      status: "running",
      error: null,
    },
  },
  {
    kind: "progress",
    timestamp,
    progress: {
      event: "grassmoth.extraction.plan",
      attempt_id,
      cursor: "2-0",
      total_workers: 2,
      max_concurrent_workers: 1,
    },
  },
  {
    kind: "progress",
    timestamp,
    progress: {
      event: "grassmoth.extraction.worker",
      attempt_id,
      cursor: "3-0",
      worker_id: 7,
      state: "completed",
      n_windows: 4,
      n_extractions: 8,
      n_llm_calls: 3,
      error: null,
    },
  },
  {
    kind: "progress",
    timestamp,
    progress: {
      event: "grassmoth.extraction.snapshot",
      attempt_id,
      cursor: "4-0",
      total_workers: 2,
      pending_workers: 0,
      running_workers: 1,
      completed_workers: 1,
      failed_workers: 0,
    },
  },
  {
    kind: "trace",
    timestamp,
    trace_id: "worker-7",
    trace: {
      messages: [
        {
          role: "assistant",
          content: "Inspecting retained observations.",
          reasoning: "Check the recorded inputs.",
          tool_calls: null,
          tool_call_id: null,
          tool_name: null,
          tool_result: null,
          tool_is_error: false,
        },
      ],
      model: "unit-model",
      total_time_seconds: 1,
      usage: { input_tokens: 3, output_tokens: 5, reasoning_tokens: 2 },
    },
  },
];

export const executionFailure: ExecutionMessage = {
  kind: "failure",
  timestamp,
  failure: {
    status: "raised",
    error_type: "ExtractionError",
    error_message: "Worker failed",
    details: ["Retained provider error details."],
  },
};
