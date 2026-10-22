import type { ModelDiffOutput, TimelineRevision } from "@grassmoth/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";
import { readActionResult } from "@/lib/api/endpoints";

export function useModelDiff(
  workspaceId: string,
  before: string | null,
  after: string | null,
  attempts: readonly TimelineRevision[],
) {
  const call = attempts.find((entry) => {
    const { request, outcome } = entry.record.attempt;
    return (
      outcome.status === "applied" &&
      request?.action === "model_diff" &&
      request.input.before_ref === before &&
      request.input.after_ref === after
    );
  });
  return useQuery({
    queryKey: ["action-result", workspaceId, call?.call_id],
    queryFn: call ? ({ signal }) => readActionResult(workspaceId, call, signal) : skipToken,
    select: (result): ModelDiffOutput | null =>
      result.action === "model_diff" ? result.body : null,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}
