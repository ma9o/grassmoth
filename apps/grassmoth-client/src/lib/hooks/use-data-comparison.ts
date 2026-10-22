"use client";

import type { DataComparisonReport, GitOid, TimelineRevision } from "@grassmoth/api-types";
import { useQueries } from "@tanstack/react-query";
import { readActionResult } from "@/lib/api/endpoints";
import { producingCall } from "@/lib/model-asset/call-dependencies";
import { dataComparisonView, type DataSourceResult } from "@/lib/model-asset/data-comparison";

/** Resolve each immutable source once, including artifact refs and explicitly selected replicates. */
export function useDataComparison(
  workspaceId: string,
  report: DataComparisonReport | null,
  attempts: readonly TimelineRevision[],
) {
  const revisions = report
    ? [...new Set([...report.left, ...report.right].map((source) => source.revision))]
    : [];
  const producers = revisions.map((revision) => producingCall(attempts, revision));
  const calls = [
    ...new Map(producers.flatMap((call) => (call ? [[call.call_id, call] as const] : []))).values(),
  ];
  const queries = useQueries({
    queries: calls.map((call) => ({
      queryKey: ["action-result", workspaceId, call.call_id],
      queryFn: ({ signal }: { signal: AbortSignal }) => readActionResult(workspaceId, call, signal),
      staleTime: Infinity,
      retry: false,
    })),
  });
  const sources = new Map<GitOid, DataSourceResult>();
  for (const [index, revision] of revisions.entries()) {
    const queryIndex = calls.findIndex((call) => call.call_id === producers[index]?.call_id);
    const result = queries[queryIndex]?.data;
    if (!result) continue;
    if (result.action !== "prepare_data" && result.action !== "simulate")
      throw new Error("Saved comparison source is not a data-producing action");
    sources.set(revision, result);
  }
  const missing = revisions.find((_, index) => producers[index] === undefined);
  const error =
    queries.find((query) => query.error)?.error ??
    (missing ? new Error(`Missing saved comparison producer ${missing}`) : null);
  return {
    data: report && sources.size === revisions.length ? dataComparisonView(report, sources) : null,
    error,
    isLoading: queries.some((query) => query.isLoading),
  };
}
