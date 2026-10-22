"use client";

import type { ModelSnapshot, TimelineRevision } from "@grassmoth/api-types";
import { queryOptions, skipToken, useQueries, useQuery } from "@tanstack/react-query";
import { readActionResult } from "@/lib/api/endpoints";
import { callDependencies, producingCall } from "@/lib/model-asset/call-dependencies";
import { composeCallView } from "@/lib/model-asset/compose-call-view";
import { useStudyJournal } from "./use-study-journal";

function callQuery(workspaceId: string, call: TimelineRevision | undefined, enabled: boolean) {
  return queryOptions({
    queryKey: ["action-result", workspaceId, call?.call_id],
    queryFn: call
      ? ({ signal }: { signal: AbortSignal }) => readActionResult(workspaceId, call, signal)
      : skipToken,
    enabled: enabled && call !== undefined,
    staleTime: Infinity,
    retry: false,
  });
}

/** A selected call or explicitly named input revision resolves to its applied producer. */
export function useActionResult(workspaceId: string, identity: string | undefined, enabled = true) {
  const journal = useStudyJournal(workspaceId).data;
  const call = producingCall(journal?.attempts ?? [], identity);
  return useQuery(callQuery(workspaceId, call, enabled));
}

/** Locate the simulation selected through the viewed call's recorded dependencies. */
export function useViewedSimulationResult(modelSnapshot: ModelSnapshot, enabled: boolean) {
  const journal = useStudyJournal(modelSnapshot.workspace_id).data;
  const selected = producingCall(journal?.attempts ?? [], modelSnapshot.commit_id);
  const calls =
    selected && journal ? callDependencies(selected, journal.attempts, journal.dependencies) : [];
  const call = calls.findLast((entry) => entry.record.attempt.action === "simulate");
  return useQuery(callQuery(modelSnapshot.workspace_id, call, enabled));
}

export function useModelSnapshot(workspaceId: string, identity?: string, enabled = true) {
  const journal = useStudyJournal(workspaceId);
  const selected = producingCall(journal.data?.attempts ?? [], identity);
  const dependencies = journal.data?.dependencies ?? [];
  const calls =
    selected && journal.data ? callDependencies(selected, journal.data.attempts, dependencies) : [];
  const queries = useQueries({
    queries: calls.map((call) => callQuery(workspaceId, call, enabled)),
  });
  const results = new Map(
    calls.flatMap((call, index) => {
      const result = queries[index]?.data;
      return result ? [[call.record.seq, result] as const] : [];
    }),
  );
  return {
    data:
      selected && journal.data && results.size === calls.length
        ? composeCallView(workspaceId, selected, journal.data.attempts, dependencies, results)
        : undefined,
    error: queries.find((query) => query.error)?.error ?? journal.error,
    result: selected ? results.get(selected.record.seq) : undefined,
  };
}
