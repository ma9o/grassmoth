import type { ModelDiffOutput, TimelineRevision } from "@grassmoth/api-types";
import { useQuery } from "@tanstack/react-query";
import { recordedCallQuery } from "./use-action-poll";

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
    ...recordedCallQuery(workspaceId, call),
    select: (result): ModelDiffOutput | null =>
      result.status === "success" && result.action === "model_diff" ? result.body : null,
  });
}
