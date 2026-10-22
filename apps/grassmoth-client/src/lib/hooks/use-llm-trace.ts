"use client";

import type { TimelineRevision } from "@grassmoth/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";
import { getLLMTraceForAction } from "../api/endpoints";

const LLM_TRACE_QUERY_VERSION = 2;

export function getLLMTraceForActionQueryKey(workspaceId: string | null, commitId: string | null) {
  return [
    "pipeline",
    workspaceId,
    "llm-trace",
    "action",
    commitId,
    `v${LLM_TRACE_QUERY_VERSION}`,
  ] as const;
}

/** Merged LLM trace of one recorded action: the conversation turn behind a scrubber tick. */
export function useLLMTraceForAction(
  workspaceId: string | null,
  call: TimelineRevision | undefined,
  traceIds: readonly string[],
  enabled: boolean,
) {
  return useQuery({
    queryKey: getLLMTraceForActionQueryKey(workspaceId, call?.commit_id ?? null),
    queryFn:
      workspaceId !== null && call !== undefined && call.call_id !== null && traceIds.length > 0
        ? () => getLLMTraceForAction(workspaceId, call, traceIds)
        : skipToken,
    // An action without traces (e.g. submitted directly through the API) has nothing to fetch.
    enabled:
      !!workspaceId &&
      call !== undefined &&
      call.call_id !== null &&
      traceIds.length > 0 &&
      enabled,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}
