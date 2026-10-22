import type { ActionPoll, LLMTrace, TimelineRevision } from "@grassmoth/api-types";
import { apiClient } from "./client";

/** GET reads the recorded call; the viewer never submits scientific work. */
async function readCall(
  workspaceId: string,
  call: TimelineRevision,
  signal?: AbortSignal,
): Promise<ActionPoll> {
  if (call.call_id === null)
    throw new Error("This historical attempt has no retained call identity");
  const response = await apiClient.GET("/api/studies/{workspace_id}/{action}/{call_id}", {
    params: {
      path: {
        workspace_id: workspaceId,
        action: call.record.attempt.action,
        call_id: call.call_id,
      },
    },
    ...(signal === undefined ? {} : { signal }),
  });
  if (response.data === undefined)
    throw new Error(
      `Cannot read ${call.record.attempt.action} (${response.response.status}): ${JSON.stringify(response.error)}`,
    );
  return response.data;
}

export async function readActionResult(
  workspaceId: string,
  call: TimelineRevision,
  signal?: AbortSignal,
) {
  const result = await readCall(workspaceId, call, signal);
  if (result.status !== "success") throw new Error("The saved successful call is unavailable");
  return result;
}

/** The result owns all traces; merging conversation envelopes is presentation only. */
export async function getLLMTraceForAction(
  workspaceId: string,
  call: TimelineRevision,
  traceIds: readonly string[],
): Promise<LLMTrace> {
  const result = await readCall(workspaceId, call);
  const traces = traceIds.map((identity) => {
    const trace = result.messages.find(
      (message) => message.kind === "trace" && message.trace_id === identity,
    );
    if (trace?.kind !== "trace") throw new Error(`Saved trace is unavailable: ${identity}`);
    return trace.trace;
  });
  return traces.reduce<LLMTrace>(
    (merged, trace) => ({
      messages: [...merged.messages, ...trace.messages],
      model: trace.model || merged.model,
      total_time_seconds: merged.total_time_seconds + trace.total_time_seconds,
      usage: {
        input_tokens: merged.usage.input_tokens + trace.usage.input_tokens,
        output_tokens: merged.usage.output_tokens + trace.usage.output_tokens,
        reasoning_tokens:
          merged.usage.reasoning_tokens === null && trace.usage.reasoning_tokens === null
            ? null
            : (merged.usage.reasoning_tokens ?? 0) + (trace.usage.reasoning_tokens ?? 0),
      },
    }),
    {
      messages: [],
      model: "",
      total_time_seconds: 0,
      usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: null },
    },
  );
}
